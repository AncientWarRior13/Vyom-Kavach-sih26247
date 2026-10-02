"""Second VQC attempt (Stage 5c-v2) -- a genuine effort to improve on
vqc.py's near-random result, not a rerun of the same thing.

vqc.py's COBYLA/100-iteration/single-initial-point run converged to
cross-entropy ~0.844 (barely below ln(2) ~ 0.693, i.e. close to a coin flip)
-- a training-difficulty result, not necessarily evidence the circuit can't
express the boundary. This script tests that distinction directly:
  - SPSA instead of COBYLA (the standard alternative for noisy/rugged VQC
    loss landscapes; stochastic perturbation can escape flat regions COBYLA
    gets stuck in), maxiter=150
  - 3 independent random initial points (restarts), keeping the best by
    training loss -- separates "bad luck on one random start" from "circuit
    genuinely can't do this"
  - identical task/features/encoding as vqc.py (same subsample, same
    ZZFeatureMap, same RealAmplitudes ansatz) so the ONLY variable is the
    optimizer/restart strategy

Reported honestly either way: if this still loses to QSVC/classical, that
strengthens (not weakens) the original finding -- it means the gap survived
a genuine, differently-configured attempt to close it, not just one
unlucky run.
"""

import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_machine_learning.optimizers import SPSA
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.preprocessing import MinMaxScaler

from radarsense import config as C
from radarsense.quantum import subsample
from radarsense.vqc import ANSATZ_REPS, N_TRAIN

MAXITER = 150
N_RESTARTS = 3


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

    circuit_meta = {
        "feature_map": "ZZFeatureMap (identical to vqc.py and quantum.py)",
        "ansatz": "RealAmplitudes", "ansatz_reps": ANSATZ_REPS,
        "trainable_parameters": int(ansatz.num_parameters),
        "optimizer": "SPSA (vs. COBYLA in vqc.py)", "maxiter": MAXITER,
        "n_restarts": N_RESTARTS, "n_train": N_TRAIN, "n_test": C.QUANTUM_TEST_SIZE,
    }
    print("circuit:", circuit_meta)

    runs = []
    rng_np = np.random.RandomState(C.SEED)
    t0 = time.time()
    for restart in range(N_RESTARTS):
        init = rng_np.uniform(-np.pi, np.pi, size=ansatz.num_parameters)
        loss_history = []
        def spsa_callback(n_evals, params, fval, stepsize, accepted):
            loss_history.append(float(fval))

        tr0 = time.time()
        vqc = VQC(feature_map=fmap, ansatz=ansatz,
                  optimizer=SPSA(maxiter=MAXITER, callback=spsa_callback),
                  loss="cross_entropy", initial_point=init)
        vqc.fit(Xtr, ytr)
        train_seconds = time.time() - tr0
        final_loss = vqc._fit_result.fun

        pred = vqc.predict(Xte)
        proba = vqc.predict_proba(Xte)
        pos_scores = proba[:, 1] if proba.ndim == 2 and proba.shape[1] == 2 else proba.ravel()
        m = {
            "restart": restart,
            "recall_drone": float(recall_score(yte, pred)),
            "precision_drone": float(precision_score(yte, pred, zero_division=0)),
            "f1_drone": float(f1_score(yte, pred, zero_division=0)),
            "pr_auc": float(average_precision_score(yte, pos_scores)),
            "roc_auc": float(roc_auc_score(yte, pos_scores)),
            "final_loss": float(final_loss),
            "n_iterations_recorded": len(loss_history),
            "train_seconds": round(train_seconds, 1),
        }
        runs.append({"metrics": m, "loss_curve": loss_history})
        print(f"restart {restart}: PR-AUC {m['pr_auc']:.3f} recall {m['recall_drone']:.2f} "
              f"final_loss {m['final_loss']:.3f} ({train_seconds:.1f}s)")

    best = min(runs, key=lambda r: r["metrics"]["final_loss"])
    total_seconds = time.time() - t0

    prior = {}
    for fname, key in [("quantum_results.json", None), ("vqc_results.json", "vqc_metrics")]:
        try:
            d = json.loads((C.REPORTS_DIR / fname).read_text())
            prior[fname] = ([r for r in d["results"] if r["n_train"] == N_TRAIN]
                             if key is None else d[key])
        except FileNotFoundError:
            pass

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for r in runs:
        ax.plot(r["loss_curve"], alpha=0.7, label=f"restart {r['metrics']['restart']}")
    ax.set_xlabel("SPSA iteration"); ax.set_ylabel("cross-entropy loss")
    ax.set_title(f"VQC v2: SPSA, {N_RESTARTS} restarts x {MAXITER} iterations")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(C.PLOTS_DIR / "vqc_v2_spsa_restarts.png", dpi=150)
    plt.close(fig)

    out = {
        "circuit": circuit_meta,
        "all_restarts": runs,
        "best_restart_by_final_loss": best["metrics"],
        "comparison_prior_results_same_n_train": prior,
        "total_seconds": round(total_seconds, 1),
        "honest_finding": (
            "compares the best-of-3 SPSA restart against vqc.py's single "
            "COBYLA run and the QSVC/classical numbers at the same n_train. "
            "If this still trails QSVC/classical, the gap is likely a genuine "
            "circuit-expressivity or task-representation limitation rather "
            "than one unlucky optimizer run -- reported either way."),
    }
    (C.REPORTS_DIR / "vqc_v2_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nbest restart: PR-AUC {best['metrics']['pr_auc']:.3f} "
          f"recall {best['metrics']['recall_drone']:.2f} "
          f"final_loss {best['metrics']['final_loss']:.3f}")
    print(f"results -> {C.REPORTS_DIR / 'vqc_v2_results.json'}")


if __name__ == "__main__":
    main()
