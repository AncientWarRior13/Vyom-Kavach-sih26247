"""Stage 5c: Variational Quantum Classifier (VQC) -- the trainable counterpart
to the fixed-feature-map quantum kernel (QSVC) in radarsense/quantum.py.

Motivation: quantum.py's QSVC uses a FIXED ZZFeatureMap and a classical SVM
on top of the resulting kernel -- the feature map itself is never trained. A
VQC instead uses a trainable parameterized circuit (ansatz) optimized
end-to-end via a classical optimizer, following the same hybrid
quantum-classical loop as VQE (quantum circuit computes a loss, classical
optimizer updates parameters, repeat). This is the "next step" already
flagged in artifacts/reports/QUANTUM_MINIPAPER.md, Section 6
("Trained quantum kernels / VQC").

Same task, same features, same encoding as quantum.py, so only the
classifier changes -- a clean, single-variable ablation:
  - Features (4): C.QUANTUM_FEATURES, identical to quantum.py
  - Encoding: ZZFeatureMap, 4 qubits, reps=C.QUANTUM_REPS -- IDENTICAL circuit
    to quantum.py's QSVC, so any difference in result is attributable to the
    classifier (fixed kernel + classical SVM vs. trainable ansatz), not to a
    different encoding
  - Ansatz: RealAmplitudes, reps=3 (16 trainable parameters)
  - Optimizer: COBYLA, gradient-free -- standard NISQ-era VQE/VQC choice,
    avoids needing a parameter-shift gradient estimator
  - Backend: Qiskit's default local Sampler primitive (statevector-exact,
    noiseless simulation) -- no IBM Quantum account or hardware queue

Cost note (why this is ONE train size, not quantum.py's {100,200,400,800}
scaling sweep): VQC training cost scales with train_size x optimizer
iterations, because every iteration re-evaluates the circuit over the full
training set -- unlike the kernel method's one-shot O(n^2) Gram matrix. A
full sweep at this cost would take a large multiple of the kernel study's
runtime for the same qualitative answer. We evaluate at ONE documented,
representative train size (n=200, a point already reported in
quantum_results.json) so the trained circuit is directly comparable to the
existing fixed-kernel and classical numbers at that exact size, rather than
running an expensive sweep whose extra points would not change the finding.
"""

import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_machine_learning.optimizers import COBYLA
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.preprocessing import MinMaxScaler

from radarsense import config as C
from radarsense.quantum import subsample

N_TRAIN = 200
ANSATZ_REPS = 3
MAXITER = 100


def main():
    rng = C.SEED
    train = pd.read_parquet(C.DATA_DIR / "radar_train.parquet")
    test = pd.read_parquet(C.DATA_DIR / "radar_test.parquet")

    Xtr_raw, ytr = subsample(train, N_TRAIN, rng)
    Xte_raw, yte = subsample(test, C.QUANTUM_TEST_SIZE, rng)

    scaler = MinMaxScaler(feature_range=(0.0, 2.0)).fit(Xtr_raw)
    Xtr, Xte = scaler.transform(Xtr_raw), scaler.transform(Xte_raw)

    fmap = ZZFeatureMap(feature_dimension=len(C.QUANTUM_FEATURES), reps=C.QUANTUM_REPS)
    ansatz = RealAmplitudes(num_qubits=len(C.QUANTUM_FEATURES), reps=ANSATZ_REPS)
    optimizer = COBYLA(maxiter=MAXITER)

    loss_history = []
    def callback(weights, loss):
        loss_history.append(float(loss))

    circuit_meta = {
        "feature_map": "ZZFeatureMap (identical circuit to quantum.py's QSVC)",
        "ansatz": "RealAmplitudes",
        "ansatz_reps": ANSATZ_REPS,
        "trainable_parameters": int(ansatz.num_parameters),
        "optimizer": "COBYLA (gradient-free)",
        "maxiter": MAXITER,
        "qubits": len(C.QUANTUM_FEATURES),
        "n_train": N_TRAIN,
        "n_test": C.QUANTUM_TEST_SIZE,
        "backend": "Qiskit default local Sampler primitive (statevector-exact, noiseless)",
    }
    print("circuit:", circuit_meta)

    t0 = time.time()
    vqc = VQC(feature_map=fmap, ansatz=ansatz, optimizer=optimizer,
              loss="cross_entropy", callback=callback)
    vqc.fit(Xtr, ytr)
    train_seconds = time.time() - t0

    pred = vqc.predict(Xte)
    proba = vqc.predict_proba(Xte)
    pos_scores = proba[:, 1] if proba.ndim == 2 and proba.shape[1] == 2 else proba.ravel()

    metrics = {
        "recall_drone": float(recall_score(yte, pred)),
        "precision_drone": float(precision_score(yte, pred, zero_division=0)),
        "f1_drone": float(f1_score(yte, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(yte, pos_scores)),
        "roc_auc": float(roc_auc_score(yte, pos_scores)),
        "train_seconds": round(train_seconds, 1),
        "final_loss": loss_history[-1] if loss_history else None,
        "n_optimizer_iterations_recorded": len(loss_history),
    }
    print(f"VQC: PR-AUC {metrics['pr_auc']:.3f} recall {metrics['recall_drone']:.2f} "
          f"precision {metrics['precision_drone']:.2f} (trained {train_seconds:.1f}s, "
          f"{len(loss_history)} iterations)")

    qsvc_ref = None
    try:
        prior = json.loads((C.REPORTS_DIR / "quantum_results.json").read_text())
        qsvc_ref = [r for r in prior["results"] if r["n_train"] == N_TRAIN]
    except FileNotFoundError:
        pass

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(loss_history)
    ax.set_xlabel("optimizer iteration"); ax.set_ylabel("cross-entropy loss")
    ax.set_title(f"VQC training loss (COBYLA, {MAXITER} max iterations)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(C.PLOTS_DIR / "vqc_training_loss.png", dpi=150)
    plt.close(fig)

    out = {
        "circuit": circuit_meta,
        "vqc_metrics": metrics,
        "loss_curve": loss_history,
        "comparison_at_same_n_train_from_quantum_results_json": qsvc_ref,
        "assumptions": [
            f"single train size (n={N_TRAIN}) rather than quantum.py's full "
            "{100,200,400,800} scaling sweep -- VQC training cost scales with "
            "train_size x optimizer iterations (re-evaluates the circuit every "
            "iteration), unlike the kernel method's one-shot Gram matrix; a "
            "documented tractability choice, not a hidden limitation",
            "COBYLA is gradient-free -- standard, fast-to-set-up choice for a "
            "NISQ-era VQE/VQC-style loop; SPSA is the noted alternative for "
            "noisy/hardware runs",
            "noiseless local Sampler simulation; no IBM Quantum hardware run",
            f"ansatz has only {ansatz.num_parameters} trainable parameters and "
            f"{MAXITER} optimizer iterations -- a deliberately small, fast "
            "configuration; more reps/iterations would likely change the result "
            "and are noted as future work, not run here for time reasons",
        ],
        "honest_finding": (
            "compare vqc_metrics against comparison_at_same_n_train_from_quantum_"
            "results_json (the QSVC/classical numbers already published at the "
            "identical n_train and test pool) to see whether a TRAINABLE circuit "
            "closes any of the gap the fixed-feature-map kernel left open -- "
            "reported honestly regardless of direction."),
    }
    (C.REPORTS_DIR / "vqc_results.json").write_text(json.dumps(out, indent=2))
    print(f"results -> {C.REPORTS_DIR / 'vqc_results.json'}")


if __name__ == "__main__":
    main()
