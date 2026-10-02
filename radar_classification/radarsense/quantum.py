"""Stage 5: quantum kernel enhancement experiment (Track C).

QSVC with a fidelity quantum kernel on the drone-vs-rest decision boundary,
using the 4 top physics features. Ablation against classical RBF and linear
SVMs on the IDENTICAL subsampled splits, plus a data-scaling study
(quantum kernel advantage is data-regime dependent — show it, don't claim it).

Backend: Qiskit statevector-based fidelity kernel (noiseless simulation).
Qubits = n_features = 4, ZZFeatureMap reps = 2 -> circuit depth reported.
Real-hardware runs are future work (queue latency); framing follows the
mini-paper in artifacts/reports/QUANTUM_MINIPAPER.md.

Subsampling: quantum kernels need O(n^2) circuit evaluations; train sizes
100-800 with a fixed 300-sample balanced-ish test pool. Drone fraction is
raised to ~30% in these subsamples (documented) — at the native 2.5% rate a
100-sample kernel study would contain ~2 drones, which measures nothing.
"""

import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from radarsense import config as C

DRONE_FRAC = 0.30


def statevectors(fmap, X):
    """Exact statevector per sample; the fidelity kernel K_ij = |<psi_i|psi_j>|^2
    is then a Gram matrix — identical to FidelityQuantumKernel on a noiseless
    statevector backend, but vectorized."""
    return np.stack([Statevector(fmap.assign_parameters(x)).data for x in X])


def fidelity_kernel(S_a, S_b=None):
    S_b = S_a if S_b is None else S_b
    return np.abs(S_a.conj() @ S_b.T) ** 2


def subsample(df, n, rng):
    """Stratified subsample with drone share raised to DRONE_FRAC."""
    n_drone = int(n * DRONE_FRAC)
    drones = df[df[C.LABEL_COLUMN] == C.MINORITY_CLASS].sample(n_drone, random_state=rng)
    others = df[df[C.LABEL_COLUMN] != C.MINORITY_CLASS].sample(n - n_drone, random_state=rng)
    out = pd.concat([drones, others]).sample(frac=1, random_state=rng)
    X = out[C.QUANTUM_FEATURES].values
    y = (out[C.LABEL_COLUMN] == C.MINORITY_CLASS).astype(int).values
    return X, y


def eval_kernel(name, K_train, K_test, y_train, y_test, extra=None):
    clf = SVC(kernel="precomputed", C=1.0, random_state=C.SEED)
    clf.fit(K_train, y_train)
    scores = clf.decision_function(K_test)
    pred = clf.predict(K_test)
    m = {
        "kernel": name,
        "recall_drone": float(recall_score(y_test, pred)),
        "precision_drone": float(precision_score(y_test, pred, zero_division=0)),
        "f1_drone": float(f1_score(y_test, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_test, scores)),
        "roc_auc": float(roc_auc_score(y_test, scores)),
    }
    if extra:
        m.update(extra)
    return m


def main():
    rng = C.SEED
    train = pd.read_parquet(C.DATA_DIR / "radar_train.parquet")
    test = pd.read_parquet(C.DATA_DIR / "radar_test.parquet")

    Xte_raw, yte = subsample(test, C.QUANTUM_TEST_SIZE, rng)

    fmap = ZZFeatureMap(feature_dimension=len(C.QUANTUM_FEATURES), reps=C.QUANTUM_REPS)
    circuit_meta = {
        "feature_map": "ZZFeatureMap",
        "qubits": len(C.QUANTUM_FEATURES),
        "reps": C.QUANTUM_REPS,
        "circuit_depth": int(fmap.decompose().depth()),
        "backend": ("Qiskit exact statevector simulation (noiseless); fidelity "
                    "kernel K_ij = |<psi_i|psi_j>|^2 computed as a Gram matrix"),
        "features": C.QUANTUM_FEATURES,
        "drone_fraction_in_subsamples": DRONE_FRAC,
    }
    print("circuit:", circuit_meta)

    results = []
    for n in C.QUANTUM_TRAIN_SIZES:
        Xtr_raw, ytr = subsample(train, n, rng)
        scaler = MinMaxScaler(feature_range=(0.0, 2.0)).fit(Xtr_raw)
        Xtr, Xte = scaler.transform(Xtr_raw), scaler.transform(Xte_raw)

        t0 = time.time()
        S_tr = statevectors(fmap, Xtr)
        S_te = statevectors(fmap, Xte)
        Ktr = fidelity_kernel(S_tr)
        Kte = fidelity_kernel(S_te, S_tr)
        q_seconds = time.time() - t0
        results.append(eval_kernel(
            "quantum_zz", Ktr, Kte, ytr, yte,
            {"n_train": n, "kernel_seconds": round(q_seconds, 1)}))

        for kname in ("rbf", "linear"):
            csvc = SVC(kernel=kname, C=1.0, random_state=C.SEED)
            csvc.fit(Xtr, ytr)
            scores = csvc.decision_function(Xte)
            pred = csvc.predict(Xte)
            results.append({
                "kernel": kname, "n_train": n,
                "recall_drone": float(recall_score(yte, pred)),
                "precision_drone": float(precision_score(yte, pred, zero_division=0)),
                "f1_drone": float(f1_score(yte, pred, zero_division=0)),
                "pr_auc": float(average_precision_score(yte, scores)),
                "roc_auc": float(roc_auc_score(yte, scores)),
            })
        last = [r for r in results if r["n_train"] == n]
        print(f"n={n}: " + " | ".join(
            f"{r['kernel']} PR-AUC {r['pr_auc']:.3f} rec {r['recall_drone']:.2f}"
            for r in last))

    y_pm = 2 * ytr - 1
    T = np.outer(y_pm, y_pm)
    def alignment(K):
        return float((K * T).sum() / (np.linalg.norm(K) * np.linalg.norm(T)))
    from sklearn.metrics.pairwise import rbf_kernel as sk_rbf
    align = {"quantum_zz": alignment(Ktr),
             "rbf": alignment(sk_rbf(Xtr))}
    print("kernel-target alignment (n=%d):" % n, align)

    df = pd.DataFrame(results)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for kernel, g in df.groupby("kernel"):
        axes[0].plot(g.n_train, g.pr_auc, "o-", label=kernel)
        axes[1].plot(g.n_train, g.recall_drone, "o-", label=kernel)
    axes[0].set_xlabel("train size"); axes[0].set_ylabel("drone PR-AUC")
    axes[0].set_title("Quantum vs classical kernels: PR-AUC scaling")
    axes[1].set_xlabel("train size"); axes[1].set_ylabel("drone recall")
    axes[1].set_title("Drone recall scaling")
    for ax in axes:
        ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(C.PLOTS_DIR / "quantum_scaling_study.png", dpi=150)
    plt.close(fig)

    out = {"circuit": circuit_meta, "results": results,
           "kernel_target_alignment_largest_n": align}
    (C.REPORTS_DIR / "quantum_results.json").write_text(json.dumps(out, indent=2))
    print(f"results -> {C.REPORTS_DIR / 'quantum_results.json'}")


if __name__ == "__main__":
    main()
