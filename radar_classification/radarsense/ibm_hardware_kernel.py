"""Real IBM Quantum hardware run: noise characterization of the same
ZZFeatureMap(4, reps=2) fidelity kernel used throughout quantum.py, vqc.py,
diat_usat_cnn.py, raddar_cnn.py, and track_fusion.py -- every OTHER quantum
result in this project is noiseless exact statevector simulation, always
disclosed as such. This is the one real-hardware measurement.

TWO SEPARATE ENVIRONMENTS, by necessity: qiskit-ibm-runtime requires
qiskit>=2.3, but qiskit-machine-learning (used by vqc.py) requires
qiskit<2.0 -- genuinely incompatible in one environment (confirmed: pip
installing qiskit-ibm-runtime silently upgraded qiskit to 2.5.0 and broke
VQC's import chain; fixed by reverting the main environment and creating a
separate .venv_ibm_quantum/ venv, gitignored, just for this module's
hardware-submission step).

  STAGE 1 (main environment -- pandas, radarsense.quantum available):
    prepare_points()          -- 6 real subsampled points, same
                                  radarsense.quantum.subsample() function
                                  every other quantum experiment uses
    compute_noiseless_kernel() -- exact statevector fidelity for the same
                                  6 points, ground truth to compare against

  STAGE 2 (.venv_ibm_quantum/ -- qiskit-ibm-runtime available, NOT
  radarsense/pandas): run_on_hardware() -- submits the 15 unique pairwise
  fidelity circuits (inversion test: feature-map(x_i) then inverse
  feature-map(x_j), measure, P(all-zeros) estimates the fidelity) as ONE
  batched job, not 15 separate jobs.

  STAGE 3 (main environment): compare_results() -- loads both JSONs,
  computes mean absolute deviation and correlation.

Credentials: read from .claude/ibm_quantum_credentials.json (gitignored,
never committed) -- {"token": ..., "instance_crn": ...}.

Already-run result (job d9f6pskjeosc73fji2j0, backend ibm_fez, 2026-07-21):
mean absolute deviation 0.020, Pearson correlation 0.920 between real
hardware and noiseless simulation -- see artifacts/reports/
IBM_HARDWARE_EXPERIMENT.md for the full writeup and interpretation.
"""

import itertools
import json

import numpy as np
import pandas as pd

from radarsense import config as C

POINTS_PATH = C.REPORTS_DIR / "ibm_hw_points.json"
NOISELESS_PATH = C.REPORTS_DIR / "ibm_hw_noiseless_kernel.json"
HARDWARE_RESULTS_PATH = C.REPORTS_DIR / "ibm_hardware_kernel_comparison.json"


def prepare_points(n_points=6, seed=C.SEED):
    """STAGE 1a -- main environment. Same subsample() function as every
    other quantum experiment in this project."""
    from radarsense.quantum import subsample
    from sklearn.preprocessing import MinMaxScaler

    train = pd.read_parquet(C.DATA_DIR / "radar_train.parquet")
    X_raw, y = subsample(train, n_points, seed)
    scaler = MinMaxScaler(feature_range=(0.0, 2.0)).fit(X_raw)
    X = scaler.transform(X_raw)
    POINTS_PATH.write_text(json.dumps({"X": X.tolist(), "y": y.tolist()}))
    print(f"saved {n_points} points -> {POINTS_PATH}")
    return X, y


def compute_noiseless_kernel():
    """STAGE 1b -- main environment. Exact statevector fidelity, ground truth."""
    from qiskit.circuit.library import ZZFeatureMap
    from radarsense.quantum import fidelity_kernel, statevectors

    data = json.loads(POINTS_PATH.read_text())
    X = np.array(data["X"])
    fmap = ZZFeatureMap(feature_dimension=4, reps=2)
    S = statevectors(fmap, X)
    K = fidelity_kernel(S)
    NOISELESS_PATH.write_text(json.dumps({"K_noiseless": K.tolist()}))
    print(f"saved noiseless kernel -> {NOISELESS_PATH}")
    return K


def run_on_hardware(backend_name="ibm_fez", shots=2048):
    """STAGE 2 -- run with .venv_ibm_quantum/bin/python3, NOT the main env.
    Requires qiskit-ibm-runtime; deliberately does not import radarsense
    (pandas-dependent) or anything requiring qiskit<2.0."""
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    import time

    import os
    token = os.environ.get("IBM_QUANTUM_TOKEN")
    crn = os.environ.get("IBM_QUANTUM_CRN")
    data = json.loads(open(str(POINTS_PATH)).read())
    X = data["X"]
    n = len(X)

    if token and crn:
        service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token, instance=crn)
    else:
        service = QiskitRuntimeService()
    backend = service.backend(backend_name)
    print("backend:", backend.name, "pending jobs:", backend.status().pending_jobs)

    fmap = ZZFeatureMap(feature_dimension=4, reps=2)
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)

    pairs = list(itertools.combinations(range(n), 2))
    circuits = []
    for i, j in pairs:
        qc = fmap.assign_parameters(X[i]).compose(fmap.assign_parameters(X[j]).inverse())
        qc.measure_all()
        circuits.append(pm.run(qc))

    print(f"submitting {len(circuits)} circuits as one batched job, {shots} shots each...")
    sampler = SamplerV2(mode=backend)
    t0 = time.time()
    job = sampler.run(circuits, shots=shots)
    print("job id:", job.job_id())
    result = job.result()
    elapsed = time.time() - t0
    print(f"job completed in {elapsed:.1f}s")

    fidelities = {}
    for (i, j), res in zip(pairs, result):
        counts = res.data.meas.get_counts()
        p_zero = counts.get("0" * 4, 0) / sum(counts.values())
        fidelities[f"{i}_{j}"] = p_zero
        print(f"  pair ({i},{j}): fidelity estimate = {p_zero:.4f}")

    raw_path = str(C.REPORTS_DIR / "ibm_hw_raw_results.json")
    json.dump({"backend": backend.name, "job_id": job.job_id(), "shots_per_circuit": shots,
              "n_circuits": len(circuits), "elapsed_seconds": elapsed, "pairs": pairs,
              "fidelities_real_hardware": fidelities}, open(raw_path, "w"), indent=2)
    print(f"saved -> {raw_path}")


def compare_results():
    """STAGE 3 -- main environment. Load both JSONs, compute deviation/correlation."""
    hw = json.loads((C.REPORTS_DIR / "ibm_hw_raw_results.json").read_text())
    noiseless = json.loads(NOISELESS_PATH.read_text())["K_noiseless"]

    pairs = hw["pairs"]
    real_vals, sim_vals, per_pair = [], [], []
    for i, j in pairs:
        real = hw["fidelities_real_hardware"][f"{i}_{j}"]
        sim = noiseless[i][j]
        real_vals.append(real); sim_vals.append(sim)
        per_pair.append({"i": i, "j": j, "real_hardware_fidelity": real,
                         "noiseless_simulated_fidelity": sim})

    real_vals, sim_vals = np.array(real_vals), np.array(sim_vals)
    mae = float(np.mean(np.abs(real_vals - sim_vals)))
    corr = float(np.corrcoef(real_vals, sim_vals)[0, 1])
    print(f"mean absolute deviation: {mae:.4f}")
    print(f"Pearson correlation: {corr:.4f}")

    out = {"backend": hw["backend"], "job_id": hw["job_id"],
           "shots_per_circuit": hw["shots_per_circuit"], "elapsed_seconds": hw["elapsed_seconds"],
           "n_pairs": len(pairs), "mean_absolute_deviation_real_vs_noiseless": mae,
           "pearson_correlation_real_vs_noiseless": corr, "per_pair": per_pair}
    HARDWARE_RESULTS_PATH.write_text(json.dumps(out, indent=2))
    print(f"saved -> {HARDWARE_RESULTS_PATH}")
    return out


if __name__ == "__main__":
    print(__doc__)
