"""Stage 5d: quantum kernel under a REALISTIC HARDWARE NOISE MODEL.

Every other quantum experiment in this project (quantum.py's QSVC, vqc.py /
vqc_improved.py's VQC) runs on an exact, noiseless statevector simulator --
honestly disclosed throughout, but it leaves one question genuinely
untested: what happens to the quantum kernel under realistic NISQ-era
hardware noise? This script answers that directly instead of leaving it as
"future work."

Noise model: qiskit-ibm-runtime's device-calibrated fake backends
(FakeManilaV2 etc.) aren't installed in this environment and would add a new
dependency; the built-in qiskit.providers.fake_provider backends available
here (Fake5QV1 etc.) are old pulse-simulation scaffolding, not representative
of current hardware. Instead this uses a NoiseModel built from representative,
literature-typical NISQ-era error rates (documented below, not a specific
device snapshot -- avoids the false precision of citing a specific fake
device that may not be realistic):
  - single-qubit gate depolarizing error: 0.1% (typical good single-qubit
    gate fidelity ~99.9% on current superconducting hardware)
  - two-qubit (CX) gate depolarizing error: 1% (typical current two-qubit
    gate fidelity ~99% -- many real devices are worse, this is a
    representative "decent current hardware" value, not best-case)
  - readout error: 2% symmetric (typical measurement fidelity ~98%)

Method: identical task/features/encoding/task as quantum.py's QSVC (same 4
features, same ZZFeatureMap, same drone-vs-rest subsampling) so the ONLY
variable is noiseless-statevector vs. noisy-shot-based kernel evaluation.
Uses qiskit-machine-learning's ComputeUncompute fidelity + FidelityQuantumKernel
with a noisy Aer Sampler (shot-based, not exact) -- this is the standard way
to get a hardware-realistic quantum kernel without live hardware access.

Cost note: shot-based sampling is far slower per kernel entry than exact
statevector fidelity, so this uses smaller n_train/n_test than quantum.py's
scaling study (100/100 vs. 100-800/300) -- still enough for a direct,
same-size comparison against quantum.py's n=100 noiseless numbers.
"""

import json
import time

import numpy as np
import pandas as pd
from qiskit.circuit.library import ZZFeatureMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from radarsense import config as C
from radarsense.quantum import subsample

N_TRAIN = 100
N_TEST = 100
SHOTS = 1024

SINGLE_Q_ERROR = 0.001
TWO_Q_ERROR = 0.01
READOUT_ERROR = 0.02


def build_noise_model():
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(
        depolarizing_error(SINGLE_Q_ERROR, 1), ["rz", "sx", "x", "ry", "h"])
    nm.add_all_qubit_quantum_error(depolarizing_error(TWO_Q_ERROR, 2), ["cx"])
    ro = ReadoutError([[1 - READOUT_ERROR, READOUT_ERROR], [READOUT_ERROR, 1 - READOUT_ERROR]])
    nm.add_all_qubit_readout_error(ro)
    return nm


def main():
    rng = C.SEED
    train = pd.read_parquet(C.DATA_DIR / "radar_train.parquet")
    test = pd.read_parquet(C.DATA_DIR / "radar_test.parquet")

    Xtr_raw, ytr = subsample(train, N_TRAIN, rng)
    Xte_raw, yte = subsample(test, N_TEST, rng)
    scaler = MinMaxScaler(feature_range=(0.0, 2.0)).fit(Xtr_raw)
    Xtr, Xte = scaler.transform(Xtr_raw), scaler.transform(Xte_raw)

    noise_model = build_noise_model()
    backend = AerSimulator(noise_model=noise_model)
    sampler = AerSampler.from_backend(backend, default_shots=SHOTS)

    fmap = ZZFeatureMap(feature_dimension=len(C.QUANTUM_FEATURES), reps=C.QUANTUM_REPS).decompose()
    fidelity = ComputeUncompute(sampler=sampler)
    kernel = FidelityQuantumKernel(feature_map=fmap, fidelity=fidelity)

    circuit_meta = {
        "feature_map": "ZZFeatureMap (identical to quantum.py's noiseless QSVC)",
        "qubits": len(C.QUANTUM_FEATURES),
        "reps": C.QUANTUM_REPS,
        "n_train": N_TRAIN, "n_test": N_TEST, "shots_per_circuit": SHOTS,
        "noise_model": {
            "type": "representative NISQ-era rates, NOT a specific device snapshot "
                    "(qiskit-ibm-runtime's calibrated fake backends aren't installed "
                    "in this environment; the qiskit-core fake_provider backends "
                    "available here are old pulse-simulation scaffolding, not "
                    "representative of current hardware -- a documented, literature-"
                    "typical NoiseModel was used instead for transparency)",
            "single_qubit_depolarizing_error": SINGLE_Q_ERROR,
            "two_qubit_cx_depolarizing_error": TWO_Q_ERROR,
            "readout_error_symmetric": READOUT_ERROR,
        },
        "backend": "Qiskit Aer shot-based noisy simulation via ComputeUncompute + "
                   "FidelityQuantumKernel (not exact statevector)",
    }
    print("circuit:", circuit_meta)

    t0 = time.time()
    Ktr = kernel.evaluate(Xtr)
    Kte = kernel.evaluate(Xte, Xtr)
    kernel_seconds = time.time() - t0
    print(f"noisy kernel matrices computed in {kernel_seconds:.1f}s")

    clf = SVC(kernel="precomputed", C=1.0, random_state=C.SEED)
    clf.fit(Ktr, ytr)
    scores = clf.decision_function(Kte)
    pred = clf.predict(Kte)
    noisy_result = {
        "kernel": "quantum_zz_noisy",
        "recall_drone": float(recall_score(yte, pred)),
        "precision_drone": float(precision_score(yte, pred, zero_division=0)),
        "f1_drone": float(f1_score(yte, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(yte, scores)),
        "roc_auc": float(roc_auc_score(yte, scores)),
        "n_train": N_TRAIN, "kernel_seconds": round(kernel_seconds, 1),
    }
    print(f"NOISY quantum kernel: PR-AUC {noisy_result['pr_auc']:.3f} "
          f"recall {noisy_result['recall_drone']:.2f} "
          f"precision {noisy_result['precision_drone']:.2f}")

    noiseless_ref = None
    try:
        prior = json.loads((C.REPORTS_DIR / "quantum_results.json").read_text())
        noiseless_ref = [r for r in prior["results"] if r["n_train"] == N_TRAIN]
    except FileNotFoundError:
        pass

    out = {
        "circuit": circuit_meta,
        "noisy_result": noisy_result,
        "noiseless_comparison_at_same_n_train": noiseless_ref,
        "honest_finding": (
            "compares the SAME quantum kernel task under realistic shot-based "
            "hardware noise vs. quantum.py's exact noiseless statevector result "
            "at the identical n_train -- answers 'what happens under real "
            "hardware noise' with actual evidence instead of leaving it as "
            "future work. Reported regardless of direction: noise should be "
            "expected to degrade performance further below the already-losing "
            "noiseless quantum kernel, and that would be the honest, "
            "unsurprising, still-worth-reporting result."),
    }
    (C.REPORTS_DIR / "quantum_noisy_results.json").write_text(json.dumps(out, indent=2))
    print(f"results -> {C.REPORTS_DIR / 'quantum_noisy_results.json'}")


if __name__ == "__main__":
    main()
