"""Larger, ERROR-MITIGATED IBM hardware run -- the scaled-up follow-up to
ibm_hardware_kernel.py (which ran 6 points / 15 pairwise fidelity circuits
with NO mitigation). Two upgrades:

  1. Scale: 8 points -> 28 unique pairwise fidelity circuits (a bigger slice
     of the ZZFeatureMap(4, reps=2) kernel).
  2. Error mitigation: the same circuits are run TWICE on the same backend --
     once RAW, once with dynamical decoupling (idle-qubit echo sequences) +
     measurement twirling (TREX readout-error mitigation) -- so the effect of
     mitigation is MEASURED (does it move real hardware closer to the
     noiseless ground truth?), not assumed.

Same two-environment split as the original (top-level import is only the
lightweight radarsense.config; qiskit lives inside the functions). Credentials
come from the account saved to ~/.qiskit (channel ibm_quantum_platform) -- run
the hardware stage with .venv_ibm_quantum/bin/python.
"""

import itertools
import json

import numpy as np

from radarsense import config as C

POINTS_PATH = C.REPORTS_DIR / "ibm_hw_large_points.json"
NOISELESS_PATH = C.REPORTS_DIR / "ibm_hw_large_noiseless.json"
RAW_PATH = C.REPORTS_DIR / "ibm_hw_large_raw.json"
COMPARISON_PATH = C.REPORTS_DIR / "ibm_hardware_large_comparison.json"
N_POINTS = 8


def prepare_points(n_points=N_POINTS, seed=C.SEED):
    import pandas as pd
    from radarsense.quantum import subsample
    from sklearn.preprocessing import MinMaxScaler
    train = pd.read_parquet(C.DATA_DIR / "radar_train.parquet")
    X_raw, y = subsample(train, n_points, seed)
    X = MinMaxScaler(feature_range=(0.0, 2.0)).fit_transform(X_raw)
    POINTS_PATH.write_text(json.dumps({"X": X.tolist(), "y": y.tolist()}))
    print(f"saved {n_points} points -> {POINTS_PATH}")


def compute_noiseless():
    from qiskit.circuit.library import ZZFeatureMap
    from radarsense.quantum import fidelity_kernel, statevectors
    X = np.array(json.loads(POINTS_PATH.read_text())["X"])
    fmap = ZZFeatureMap(feature_dimension=4, reps=2)
    K = fidelity_kernel(statevectors(fmap, X))
    NOISELESS_PATH.write_text(json.dumps({"K_noiseless": K.tolist()}))
    print(f"saved noiseless kernel -> {NOISELESS_PATH}")


def run_on_hardware(backend_name="ibm_fez", shots=4096):
    """Run with .venv_ibm_quantum/bin/python. Submits the 28 pairwise circuits
    twice: raw, and error-mitigated (DD + measurement twirling)."""
    import time
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

    X = json.loads(open(str(POINTS_PATH)).read())["X"]
    n = len(X)
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
    print(f"{len(circuits)} circuits x {shots} shots, raw + mitigated")

    def run(sampler, tag):
        t0 = time.time()
        job = sampler.run(circuits, shots=shots)
        print(f"  [{tag}] job {job.job_id()} submitted...")
        result = job.result()
        el = time.time() - t0
        fids = {}
        for (i, j), res in zip(pairs, result):
            counts = res.data.meas.get_counts()
            fids[f"{i}_{j}"] = counts.get("0" * 4, 0) / sum(counts.values())
        print(f"  [{tag}] done in {el:.1f}s, job {job.job_id()}")
        return {"job_id": job.job_id(), "elapsed_seconds": el, "fidelities": fids}

    raw_sampler = SamplerV2(mode=backend)
    raw = run(raw_sampler, "raw")

    mit_sampler = SamplerV2(mode=backend)
    mit_sampler.options.dynamical_decoupling.enable = True
    mit_sampler.options.dynamical_decoupling.sequence_type = "XY4"
    mit_sampler.options.twirling.enable_gates = True
    mit_sampler.options.twirling.enable_measure = True
    mitigated = run(mit_sampler, "mitigated")

    json.dump({"backend": backend.name, "shots_per_circuit": shots, "n_points": n,
               "pairs": pairs, "raw": raw, "mitigated": mitigated},
              open(str(RAW_PATH), "w"), indent=2)
    print(f"saved -> {RAW_PATH}")


def compare():
    hw = json.loads(RAW_PATH.read_text())
    noiseless = json.loads(NOISELESS_PATH.read_text())["K_noiseless"]
    pairs = hw["pairs"]

    def stats(fid_dict):
        real = np.array([fid_dict[f"{i}_{j}"] for i, j in pairs])
        sim = np.array([noiseless[i][j] for i, j in pairs])
        return (float(np.mean(np.abs(real - sim))), float(np.corrcoef(real, sim)[0, 1]))

    raw_mae, raw_corr = stats(hw["raw"]["fidelities"])
    mit_mae, mit_corr = stats(hw["mitigated"]["fidelities"])
    print(f"  RAW       : MAE={raw_mae:.4f} corr={raw_corr:.4f}")
    print(f"  MITIGATED : MAE={mit_mae:.4f} corr={mit_corr:.4f}")
    print(f"  error mitigation {'IMPROVES' if mit_mae < raw_mae else 'does NOT improve'} "
          f"fidelity agreement (MAE {raw_mae:.4f} -> {mit_mae:.4f})")

    out = {"backend": hw["backend"], "n_points": hw["n_points"], "n_pairs": len(pairs),
           "shots_per_circuit": hw["shots_per_circuit"],
           "raw": {"job_id": hw["raw"]["job_id"], "mae_vs_noiseless": raw_mae, "correlation": raw_corr},
           "mitigated": {"job_id": hw["mitigated"]["job_id"], "mae_vs_noiseless": mit_mae,
                         "correlation": mit_corr},
           "mitigation_improves_mae": bool(mit_mae < raw_mae)}
    COMPARISON_PATH.write_text(json.dumps(out, indent=2))
    print(f"saved -> {COMPARISON_PATH}")


if __name__ == "__main__":
    import sys
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("prep", "all"):
        prepare_points(); compute_noiseless()
    if stage in ("hw", "all"):
        run_on_hardware()
    if stage in ("compare", "all"):
        compare()
