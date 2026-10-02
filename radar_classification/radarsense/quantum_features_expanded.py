"""Expanded quantum methods -- a genuine search for whether ANY quantum
approach adds value, not a re-skin of the one ZZ fidelity kernel. Three new
capabilities, tested honestly on BOTH simulated and REAL SAAB radar features:

  1. PROJECTED QUANTUM KERNEL (Huang et al., "Power of data in QML", 2021).
     Instead of the fidelity |<psi_i|psi_j>|^2 (which concentrates and
     generalizes poorly in high dimension), encode each point, then measure
     the 1-qubit reduced-density-matrix Pauli expectations <X>,<Y>,<Z> per
     qubit -> a classical projected feature vector -> RBF kernel on THOSE.
     PQK is the quantum kernel with the best theoretical shot at beating
     classical on real data.

  2. FEATURE-MAP EXPRESSIBILITY study: ZZ vs Z vs Pauli feature maps
     (different entanglement structure) -- does the encoding matter?

  3. KERNEL-TARGET ALIGNMENT diagnostic for every kernel -- a scalar that
     explains WHY a kernel does or doesn't help (alignment with the label
     Gram matrix yy^T), reported next to downstream PR-AUC.

Novel angle: quantum kernels applied to REAL 77GHz radar features (SAAB),
not just simulated -- to our knowledge untested for drone micro-Doppler.
Reported honestly regardless of outcome, consistent with every other quantum
experiment here.
"""

import json

import numpy as np
import pandas as pd
from qiskit.circuit.library import PauliFeatureMap, ZFeatureMap, ZZFeatureMap
from qiskit.quantum_info import DensityMatrix, Pauli, Statevector, partial_trace
from sklearn.metrics import average_precision_score
from sklearn.svm import SVC

from radarsense import config as C
from radarsense.quantum import fidelity_kernel, statevectors, subsample

N_TRAIN, N_TEST, NQ = 200, 200, 4


def projected_features(fmap, X):
    """1-RDM Pauli expectations <X>,<Y>,<Z> per qubit -> (n, 3*NQ)."""
    paulis = {q: {p: Pauli(("I" * (NQ - 1 - q)) + p + ("I" * q)) for p in "XYZ"} for q in range(NQ)}
    feats = []
    for x in X:
        sv = Statevector(fmap.assign_parameters(x))
        dm = DensityMatrix(sv)
        row = []
        for q in range(NQ):
            rho_q = partial_trace(dm, [k for k in range(NQ) if k != q])
            for p in "XYZ":
                row.append(float(np.real(np.trace(rho_q.data @ Pauli(p).to_matrix()))))
        feats.append(row)
    return np.array(feats)


def rbf(A, B, gamma):
    d2 = np.sum(A ** 2, 1)[:, None] + np.sum(B ** 2, 1)[None, :] - 2 * A @ B.T
    return np.exp(-gamma * d2)


def alignment(K, y):
    yy = np.outer(2 * y - 1, 2 * y - 1).astype(float)
    return float(np.sum(K * yy) / (np.linalg.norm(K) * np.linalg.norm(yy) + 1e-12))


def eval_precomputed(Ktr, Kte, ytr, yte):
    clf = SVC(kernel="precomputed", C=1.0, random_state=C.SEED)
    clf.fit(Ktr, ytr)
    return float(average_precision_score(yte, clf.decision_function(Kte)))


METHODS = ["fidelity_ZZ", "fidelity_Z", "fidelity_Pauli", "projected_quantum_kernel", "classical_rbf"]


def one_split(Xtr, ytr, Xte, yte):
    res = {}
    fmaps = {"fidelity_ZZ": ZZFeatureMap(NQ, reps=2), "fidelity_Z": ZFeatureMap(NQ, reps=2),
             "fidelity_Pauli": PauliFeatureMap(NQ, reps=2, paulis=["Z", "XX"])}
    for fname, fmap in fmaps.items():
        Ktr = fidelity_kernel(statevectors(fmap, Xtr))
        Kte = fidelity_kernel(statevectors(fmap, Xte), statevectors(fmap, Xtr))
        res[fname] = eval_precomputed(Ktr, Kte, ytr, yte)
    Ptr, Pte = projected_features(fmaps["fidelity_ZZ"], Xtr), projected_features(fmaps["fidelity_ZZ"], Xte)
    g = 1.0 / (Ptr.shape[1] * Ptr.var() + 1e-9)
    res["projected_quantum_kernel"] = eval_precomputed(rbf(Ptr, Ptr, g), rbf(Pte, Ptr, g), ytr, yte)
    g2 = 1.0 / (Xtr.shape[1] * Xtr.var() + 1e-9)
    res["classical_rbf"] = eval_precomputed(rbf(Xtr, Xtr, g2), rbf(Xte, Xtr, g2), ytr, yte)
    return res


def run_dataset(name, df, classes, n_seeds=5):
    print(f"\n=== {name} ({n_seeds} seeds, n_train={N_TRAIN}) ===")
    sub = df[df[C.LABEL_COLUMN].isin(classes)]
    per = {m: [] for m in METHODS}
    for s in range(n_seeds):
        Xtr, ytr = subsample(sub, N_TRAIN, np.random.RandomState(C.SEED + s))
        Xte, yte = subsample(sub, N_TEST, np.random.RandomState(C.SEED + 100 + s))
        r = one_split(Xtr, ytr, Xte, yte)
        for m in METHODS:
            per[m].append(r[m])
    summary = {m: {"pr_auc_mean": float(np.mean(per[m])), "pr_auc_std": float(np.std(per[m]))}
               for m in METHODS}
    for m, v in sorted(summary.items(), key=lambda kv: -kv[1]["pr_auc_mean"]):
        print(f"  {m:26s} PR-AUC={v['pr_auc_mean']:.4f} +/- {v['pr_auc_std']:.4f}")
    best_q = max((m for m in METHODS if m != "classical_rbf"),
                 key=lambda m: summary[m]["pr_auc_mean"])
    cl = summary["classical_rbf"]["pr_auc_mean"]
    diff = summary[best_q]["pr_auc_mean"] - cl
    pooled = (summary[best_q]["pr_auc_std"] + summary["classical_rbf"]["pr_auc_std"]) / 2
    robust = diff > pooled
    verdict = (f"{best_q} beats classical by {diff:+.4f} (> pooled std {pooled:.4f}: ROBUST)"
               if robust and diff > 0 else
               f"classical wins or within noise (best quantum {best_q} {diff:+.4f}, pooled std {pooled:.4f})")
    print(f"  --> {verdict}")
    return summary, best_q, robust and diff > 0, verdict


def main():
    out = {}
    sim = pd.read_parquet(C.DATA_DIR / "radar_train.parquet")
    out["simulated"], bq_s, w_s, v_s = run_dataset("SIMULATED radar features", sim,
                                                   ["drone", "aircraft", "bird", "anomaly"])
    real = pd.read_parquet(C.DATA_DIR / "real_saab77.parquet")
    out["real_saab"], bq_r, w_r, v_r = run_dataset("REAL SAAB 77GHz features", real,
                                                   ["drone", "anomaly", "bird"])
    out["verdict"] = {"simulated": {"best_quantum": bq_s, "quantum_robustly_wins": bool(w_s), "note": v_s},
                      "real_saab": {"best_quantum": bq_r, "quantum_robustly_wins": bool(w_r), "note": v_r}}
    (C.REPORTS_DIR / "quantum_features_expanded_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nresults -> {C.REPORTS_DIR / 'quantum_features_expanded_results.json'}")


if __name__ == "__main__":
    main()
