"""Quantum kernel DIAGNOSTIC -- explains the no-advantage result rigorously
rather than just reporting it. For each kernel (quantum ZZ, quantum Z,
classical RBF) on simulated and real SAAB features, measures the properties
that theory (Kubler 2021; Huang 2021; Thanasilp 2022 'exponential
concentration') says determine whether a kernel can discriminate:

  - off-diagonal concentration: mean and std of off-diagonal kernel values.
    An exponentially-concentrated kernel has off-diagonals clustered near a
    constant -> it cannot separate points -> no useful classifier. This is
    THE documented failure mode of expressive quantum kernels.
  - effective rank (participation ratio of the eigenspectrum): how many
    dimensions the kernel actually uses. Low effective rank vs sample size =>
    the kernel collapses the data.
  - kernel-target alignment: correlation of the kernel with the label Gram
    matrix (already reported elsewhere; recomputed here for completeness).

The story this tells: quantum kernels here concentrate (low off-diagonal
spread) and align poorly with the labels, which is exactly why classical
wins -- a mechanistic explanation, not just a scoreboard.
"""

import json

import numpy as np
import pandas as pd
from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap

from radarsense import config as C
from radarsense.quantum import fidelity_kernel, statevectors, subsample

N = 150


def off_diagonal(K):
    m = ~np.eye(K.shape[0], dtype=bool)
    off = K[m]
    return float(off.mean()), float(off.std())


def effective_rank(K):
    w = np.linalg.eigvalsh((K + K.T) / 2)
    w = np.clip(w, 0, None)
    if w.sum() <= 0:
        return 0.0
    return float((w.sum() ** 2) / (np.sum(w ** 2) + 1e-12))


def alignment(K, y):
    yy = np.outer(2 * y - 1, 2 * y - 1).astype(float)
    return float(np.sum(K * yy) / (np.linalg.norm(K) * np.linalg.norm(yy) + 1e-12))


def rbf(X, gamma):
    d2 = np.sum(X ** 2, 1)[:, None] + np.sum(X ** 2, 1)[None, :] - 2 * X @ X.T
    return np.exp(-gamma * d2)


def diagnose(name, X, y):
    kernels = {
        "quantum_ZZ": fidelity_kernel(statevectors(ZZFeatureMap(4, reps=2), X)),
        "quantum_Z": fidelity_kernel(statevectors(ZFeatureMap(4, reps=2), X)),
        "classical_rbf": rbf(X, 1.0 / (X.shape[1] * X.var() + 1e-9)),
    }
    res = {}
    for kname, K in kernels.items():
        mu, sd = off_diagonal(K)
        res[kname] = {"offdiag_mean": mu, "offdiag_std": sd,
                      "concentration_ratio": float(sd / (mu + 1e-9)),
                      "effective_rank": effective_rank(K),
                      "alignment": alignment(K, y)}
    print(f"\n=== {name} (n={len(X)}) ===")
    print(f"  {'kernel':14s} {'offdiag_mean':>12s} {'offdiag_std':>11s} {'conc_ratio':>10s} {'eff_rank':>9s} {'align':>7s}")
    for k, v in res.items():
        print(f"  {k:14s} {v['offdiag_mean']:>12.4f} {v['offdiag_std']:>11.4f} "
              f"{v['concentration_ratio']:>10.3f} {v['effective_rank']:>9.1f} {v['alignment']:>7.3f}")
    return res


def main():
    out = {}
    sim = pd.read_parquet(C.DATA_DIR / "radar_train.parquet")
    Xs, ys = subsample(sim, N, np.random.RandomState(C.SEED))
    out["simulated"] = diagnose("SIMULATED features", Xs, ys)

    real = pd.read_parquet(C.DATA_DIR / "real_saab77.parquet")
    real = real[real[C.LABEL_COLUMN].isin(["drone", "anomaly", "bird"])]
    Xr, yr = subsample(real, N, np.random.RandomState(C.SEED))
    out["real_saab"] = diagnose("REAL SAAB features", Xr, yr)

    print("\n--- interpretation ---")
    for ds in out:
        q = out[ds]["quantum_ZZ"]; c = out[ds]["classical_rbf"]
        print(f"  {ds}: quantum ZZ off-diagonal spread {q['offdiag_std']:.3f} vs classical {c['offdiag_std']:.3f}; "
              f"alignment {q['alignment']:.3f} vs {c['alignment']:.3f} -> "
              f"{'quantum concentrates / mis-aligns => cannot beat classical' if q['alignment'] < c['alignment'] else 'quantum competitive'}")

    (C.REPORTS_DIR / "quantum_diagnostic_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nresults -> {C.REPORTS_DIR / 'quantum_diagnostic_results.json'}")


if __name__ == "__main__":
    main()
