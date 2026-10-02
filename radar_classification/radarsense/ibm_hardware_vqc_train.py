"""Real IBM Quantum hardware VQC TRAINING run -- unlike ibm_hardware_kernel.py
and ibm_hardware_kernel_large.py (which characterize a FIXED feature-map
kernel's fidelity on real hardware, no training involved), this actually
trains a Variational Quantum Classifier's parameters using loss evaluations
executed on real IBM hardware, one classical-optimizer iteration at a time.

Cannot reuse qiskit_machine_learning's VQC class (used by vqc.py) here: it
requires qiskit<2.0, while qiskit-ibm-runtime requires qiskit>=2.3 --
confirmed incompatible in one environment (see ibm_hardware_kernel.py's
docstring). So the training loop is hand-rolled directly against SamplerV2,
same circuits as vqc.py (ZZFeatureMap(4, reps=2) + RealAmplitudes ansatz),
same parity-based binary readout qiskit_machine_learning's VQC uses
internally for 2-class problems.

QUOTA REALITY (checked live via QiskitRuntimeService.usage() before writing
this): Open/free plan, 600 QPU-seconds per rolling period, only ~417s
remaining at the time this was scoped. Unlike the kernel-characterization
scripts (ONE batched job total), gradient-free training needs ONE hardware
job PER OPTIMIZER ITERATION -- each job evaluates the ansatz+feature-map
circuit across the whole training batch. That means the number of iterations
achievable is budget-limited, not maxiter-limited, and is NOT known in
advance -- so this script checks real remaining quota via the API before
every single job submission and stops cleanly (keeping the best parameters
found so far) rather than crashing mid-run when the plan's hard limit hits.

  STAGE 1 (main environment -- pandas, radarsense available):
    prepare_points() -- n_train points via the same radarsense.quantum
      subsample() function (drone fraction raised to 30%) and same [0,2]
      angle-encoding scaler used throughout the project's quantum work.

  STAGE 2 (.venv_ibm_quantum/ -- qiskit-ibm-runtime, NOT radarsense/pandas):
    run_on_hardware() -- COBYLA (gradient-free, same choice as vqc.py's
    noiseless run) minimizing cross-entropy loss, where every loss
    evaluation = one SamplerV2 job of n_train circuits on real hardware.
    Stops when remaining account quota drops below a safety margin.

  STAGE 3 (main environment): evaluate_and_compare() -- final held-out test
    metrics computed via noiseless statevector simulation USING THE
    HARDWARE-LEARNED PARAMETERS (disclosed plainly: conserves remaining
    quota for training rather than spending it on evaluation), plus a
    same-iteration-budget noiseless reference run of the identical circuit
    for a fair (noise vs. no-noise, same optimizer-call count) comparison.

Credentials: saved qiskit-ibm-runtime account (channel ibm_quantum_platform),
same as ibm_hardware_kernel.py / ibm_hardware_kernel_large.py.
"""

import json

import numpy as np

from radarsense import config as C

POINTS_PATH = C.REPORTS_DIR / "ibm_hw_vqc_points.json"
RAW_PATH = C.REPORTS_DIR / "ibm_hw_vqc_raw.json"
RESULTS_PATH = C.REPORTS_DIR / "ibm_hardware_vqc_results.json"

N_TRAIN = 50
ANSATZ_REPS = 2
SHOTS = 256
SAFETY_MARGIN_SECONDS = 20.0


def prepare_points(n_train=N_TRAIN, seed=C.SEED):
    """STAGE 1 -- main environment."""
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler

    from radarsense.quantum import subsample

    train = pd.read_parquet(C.DATA_DIR / "radar_train.parquet")
    X_raw, y = subsample(train, n_train, seed)
    scaler = MinMaxScaler(feature_range=(0.0, 2.0)).fit(X_raw)
    X = scaler.transform(X_raw)

    rng = np.random.default_rng(seed)
    from qiskit.circuit.library import RealAmplitudes
    n_params = RealAmplitudes(num_qubits=len(C.QUANTUM_FEATURES), reps=ANSATZ_REPS).num_parameters
    theta0 = (0.1 * rng.standard_normal(n_params)).tolist()

    POINTS_PATH.write_text(json.dumps({
        "X": X.tolist(), "y": y.tolist(), "theta0": theta0,
        "n_params": n_params, "ansatz_reps": ANSATZ_REPS, "shots": SHOTS,
    }))
    print(f"saved {n_train} training points, {n_params} initial params -> {POINTS_PATH}")


def run_on_hardware(backend_name="ibm_fez"):
    """STAGE 2 -- run with .venv_ibm_quantum/bin/python3. Deliberately does
    not import radarsense (pandas-dependent) or anything requiring
    qiskit<2.0."""
    import time

    from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from scipy.optimize import minimize

    data = json.loads(POINTS_PATH.read_text())
    X = np.array(data["X"])
    y = np.array(data["y"])
    theta0 = np.array(data["theta0"])
    n_qubits = X.shape[1]
    shots = data["shots"]

    service = QiskitRuntimeService()
    backend = service.backend(backend_name)
    print("backend:", backend.name, "pending jobs:", backend.status().pending_jobs)

    def remaining_seconds():
        u = service.usage()
        return u["usage_limit_seconds"] - u["usage_consumed_seconds"]

    start_remaining = remaining_seconds()
    print(f"quota remaining at start: {start_remaining:.1f}s "
          f"(safety margin: {SAFETY_MARGIN_SECONDS}s)")

    fmap = ZZFeatureMap(feature_dimension=n_qubits, reps=C.QUANTUM_REPS if hasattr(C, "QUANTUM_REPS") else 2)
    ansatz = RealAmplitudes(num_qubits=n_qubits, reps=data["ansatz_reps"])
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    sampler = SamplerV2(mode=backend)

    class BudgetExhausted(Exception):
        pass

    state = {
        "iteration": 0,
        "best_loss": float("inf"),
        "best_theta": theta0.tolist(),
        "loss_history": [],
        "job_ids": [],
        "quota_remaining_history": [],
        "elapsed_seconds": [],
    }

    def build_circuits(theta):
        circuits = []
        for x in X:
            qc = fmap.assign_parameters(x).compose(ansatz.assign_parameters(theta))
            qc.measure_all()
            circuits.append(pm.run(qc))
        return circuits

    def parity_prob_drone(counts):
        """Standard qiskit_machine_learning parity interpretation for binary
        classification: odd popcount -> class 1 (drone)."""
        total = sum(counts.values())
        p1 = sum(c for bits, c in counts.items() if bin(int(bits, 2)).count("1") % 2 == 1)
        return p1 / total

    def objective(theta):
        remaining = remaining_seconds()
        state["quota_remaining_history"].append(remaining)
        if remaining < SAFETY_MARGIN_SECONDS:
            print(f"  quota remaining {remaining:.1f}s < safety margin "
                  f"{SAFETY_MARGIN_SECONDS}s -- stopping before next job")
            raise BudgetExhausted()

        t0 = time.time()
        circuits = build_circuits(theta)
        job = sampler.run(circuits, shots=shots)
        result = job.result()
        elapsed = time.time() - t0

        eps = 1e-9
        p_drone = np.array([parity_prob_drone(r.data.meas.get_counts()) for r in result])
        p_drone = np.clip(p_drone, eps, 1 - eps)
        loss = float(-np.mean(y * np.log(p_drone) + (1 - y) * np.log(1 - p_drone)))

        state["iteration"] += 1
        state["loss_history"].append(loss)
        state["job_ids"].append(job.job_id())
        state["elapsed_seconds"].append(elapsed)
        if loss < state["best_loss"]:
            state["best_loss"] = loss
            state["best_theta"] = theta.tolist()

        print(f"  iter {state['iteration']}: loss={loss:.4f} job={job.job_id()} "
              f"({elapsed:.1f}s, quota remaining before job: {remaining:.1f}s)")
        return loss

    try:
        minimize(objective, theta0, method="COBYLA", options={"maxiter": 500, "rhobeg": 0.3})
    except BudgetExhausted:
        pass
    except Exception as e:
        print(f"optimizer stopped: {type(e).__name__}: {e}")

    end_remaining = remaining_seconds()
    out = {
        "backend": backend.name,
        "n_train": len(X),
        "n_params": len(theta0),
        "shots": shots,
        "ansatz_reps": data["ansatz_reps"],
        "iterations_completed": state["iteration"],
        "loss_history": state["loss_history"],
        "job_ids": state["job_ids"],
        "elapsed_seconds_per_iteration": state["elapsed_seconds"],
        "best_loss": state["best_loss"],
        "best_theta": state["best_theta"],
        "quota_seconds_used_this_run": round(start_remaining - end_remaining, 1),
        "quota_remaining_after_run": round(end_remaining, 1),
    }
    RAW_PATH.write_text(json.dumps(out, indent=2))
    print(f"\ncompleted {state['iteration']} real-hardware training iterations, "
          f"used {out['quota_seconds_used_this_run']}s of quota, "
          f"{out['quota_remaining_after_run']}s remaining")
    print(f"saved -> {RAW_PATH}")


def evaluate_and_compare():
    """STAGE 3 -- main environment. Evaluate hardware-learned parameters on a
    held-out test set via noiseless simulation (disclosed: conserves the
    remaining hardware quota for training, not evaluation), and run a
    same-iteration-count noiseless reference for a fair comparison."""
    import pandas as pd
    from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
    from qiskit.quantum_info import Statevector
    from scipy.optimize import minimize
    from sklearn.metrics import (average_precision_score, f1_score,
                                  precision_score, recall_score, roc_auc_score)
    from sklearn.preprocessing import MinMaxScaler

    from radarsense.quantum import subsample

    hw = json.loads(RAW_PATH.read_text())
    pts = json.loads(POINTS_PATH.read_text())
    n_qubits = len(C.QUANTUM_FEATURES)
    fmap = ZZFeatureMap(feature_dimension=n_qubits, reps=2)
    ansatz = RealAmplitudes(num_qubits=n_qubits, reps=hw["ansatz_reps"])

    train = pd.read_parquet(C.DATA_DIR / "radar_train.parquet")
    test = pd.read_parquet(C.DATA_DIR / "radar_test.parquet")
    Xtr_raw, ytr = subsample(train, hw["n_train"], C.SEED)
    Xte_raw, yte = subsample(test, C.QUANTUM_TEST_SIZE, C.SEED)
    scaler = MinMaxScaler(feature_range=(0.0, 2.0)).fit(Xtr_raw)
    Xte = scaler.transform(Xte_raw)

    def parity_prob_drone_statevector(theta, X):
        probs = []
        for x in X:
            qc = fmap.assign_parameters(x).compose(ansatz.assign_parameters(theta))
            sv = Statevector(qc)
            p = sv.probabilities()
            p1 = sum(p[i] for i in range(len(p)) if bin(i).count("1") % 2 == 1)
            probs.append(p1)
        return np.array(probs)

    def metrics_at(theta, X, y):
        p_drone = np.clip(parity_prob_drone_statevector(theta, X), 1e-9, 1 - 1e-9)
        pred = (p_drone > 0.5).astype(int)
        return {
            "recall_drone": float(recall_score(y, pred, zero_division=0)),
            "precision_drone": float(precision_score(y, pred, zero_division=0)),
            "f1_drone": float(f1_score(y, pred, zero_division=0)),
            "pr_auc": float(average_precision_score(y, p_drone)),
            "roc_auc": float(roc_auc_score(y, p_drone)) if len(set(y)) > 1 else None,
        }

    hw_test_metrics = metrics_at(np.array(hw["best_theta"]), Xte, yte)
    print("hardware-trained VQC, test metrics (noiseless eval of hardware-learned params):")
    print(" ", hw_test_metrics)

    X = np.array(pts["X"])
    y = np.array(pts["y"])
    theta0 = np.array(pts["theta0"])
    ref_history = []

    def ref_objective(theta):
        p_drone = np.clip(parity_prob_drone_statevector(theta, X), 1e-9, 1 - 1e-9)
        loss = float(-np.mean(y * np.log(p_drone) + (1 - y) * np.log(1 - p_drone)))
        ref_history.append(loss)
        return loss

    n_iters = max(hw["iterations_completed"], 1)
    res = minimize(ref_objective, theta0, method="COBYLA",
                    options={"maxiter": n_iters, "rhobeg": 0.3})
    ref_test_metrics = metrics_at(res.x, Xte, yte)
    print(f"noiseless reference (same n_train={hw['n_train']}, same {n_iters} "
          f"iterations), test metrics:")
    print(" ", ref_test_metrics)

    out = {
        "hardware_run": {
            "backend": hw["backend"], "n_train": hw["n_train"],
            "iterations_completed": hw["iterations_completed"],
            "job_ids": hw["job_ids"], "shots": hw["shots"],
            "quota_seconds_used": hw["quota_seconds_used_this_run"],
            "best_training_loss": hw["best_loss"],
            "test_metrics": hw_test_metrics,
        },
        "noiseless_reference_same_iteration_budget": {
            "n_train": hw["n_train"], "iterations": n_iters,
            "final_training_loss": ref_history[-1] if ref_history else None,
            "test_metrics": ref_test_metrics,
        },
        "evaluation_note": (
            "Training loss evaluations (COBYLA objective calls) executed as real "
            "circuit jobs on IBM hardware. Held-out test-set metrics for both the "
            "hardware-trained and reference parameters are computed via noiseless "
            "statevector simulation, to conserve the remaining hardware quota for "
            "training rather than spending it on evaluation. This is disclosed "
            "plainly, not a hidden simulation-only result."
        ),
        "honest_interpretation": (
            "Compares a VQC whose PARAMETERS were learned from real IBM hardware "
            "circuit executions against an identical circuit trained for the same "
            "number of optimizer iterations with no hardware noise. The gap (or "
            "lack of one) between the two isolates the effect of hardware noise on "
            "training, holding architecture, data, and iteration budget fixed."
        ),
    }
    RESULTS_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {RESULTS_PATH}")
    return out


if __name__ == "__main__":
    import sys
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("prep", "all"):
        prepare_points()
    if stage in ("hw", "all"):
        run_on_hardware()
    if stage in ("eval", "all"):
        evaluate_and_compare()
