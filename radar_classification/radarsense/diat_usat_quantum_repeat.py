"""Multi-seed robustness check for the CNN+quantum-kernel ablation in
diat_usat_cnn.py.

The single-seed result there (quantum kernel wins drone recall/PR-AUC at all
3 train sizes on the frozen CNN embeddings) is real but thin evidence -- one
subsample draw per train size, one dataset. This script answers the
obvious follow-up question ("is that one lucky subsample?") by loading the
ALREADY-TRAINED, saved CNN (no retraining -- diat_usat_cnn.keras is frozen)
and repeating the identical kernel ablation across multiple independent
random subsample seeds per train size, reporting mean +/- std -- the same
statistical-rigor convention (multiple independent runs, mean +/- std) the
DIAT-uSAT CNN paper itself uses for its own headline comparisons.
"""

import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from radarsense import config as C
from radarsense.diat_usat_cnn import (
    BOTTLENECK_DIM, QUANTUM_REPS, QUANTUM_TEST_SIZE, QUANTUM_TRAIN_SIZES,
    eval_multiclass, list_files, make_arrays, split_files, stratified_subsample)

N_SEEDS = 5


def statevectors(fmap, X):
    return np.stack([Statevector(fmap.assign_parameters(x)).data for x in X])


def fidelity_kernel(S_a, S_b=None):
    S_b = S_a if S_b is None else S_b
    return np.abs(S_a.conj() @ S_b.T) ** 2


def main():
    print("loading saved CNN and rebuilding the identical file split...")
    cnn = tf.keras.models.load_model(C.MODELS_DIR / "diat_usat_cnn.keras")
    encoder = tf.keras.Model(cnn.input, cnn.get_layer("bottleneck").output)

    rows = list_files()
    train, _, test = split_files(rows)
    Xtr_img, ytr = make_arrays(train)
    Xte_img, yte = make_arrays(test)

    print("extracting frozen bottleneck embeddings...")
    Ztr = encoder.predict(Xtr_img, verbose=0)
    Zte = encoder.predict(Xte_img, verbose=0)

    fmap = ZZFeatureMap(feature_dimension=BOTTLENECK_DIM, reps=QUANTUM_REPS)

    all_results = []
    t0 = time.time()
    for n in QUANTUM_TRAIN_SIZES:
        for seed in range(N_SEEDS):
            rng = C.SEED + seed
            Ztr_sub, ytr_sub = stratified_subsample(Ztr, ytr, n, rng)
            Zte_sub, yte_sub = stratified_subsample(Zte, yte, QUANTUM_TEST_SIZE, rng)
            scaler = MinMaxScaler(feature_range=(0.0, 2.0)).fit(Ztr_sub)
            Ztr_s, Zte_s = scaler.transform(Ztr_sub), scaler.transform(Zte_sub)

            S_tr, S_te = statevectors(fmap, Ztr_s), statevectors(fmap, Zte_s)
            Ktr, Kte = fidelity_kernel(S_tr), fidelity_kernel(S_te, S_tr)

            qsvc = SVC(kernel="precomputed", C=1.0, probability=True, random_state=C.SEED)
            qsvc.fit(Ktr, ytr_sub)
            all_results.append(eval_multiclass(
                "quantum_zz", yte_sub, qsvc.predict(Kte), qsvc.predict_proba(Kte),
                list(qsvc.classes_), {"n_train": n, "seed": seed}))

            for kname in ("rbf", "linear"):
                csvc = SVC(kernel=kname, C=1.0, probability=True, random_state=C.SEED)
                csvc.fit(Ztr_s, ytr_sub)
                all_results.append(eval_multiclass(
                    kname, yte_sub, csvc.predict(Zte_s), csvc.predict_proba(Zte_s),
                    list(csvc.classes_), {"n_train": n, "seed": seed}))

        print(f"n={n}: done {N_SEEDS} seeds ({time.time()-t0:.0f}s elapsed)")

    import pandas as pd
    df = pd.DataFrame(all_results)
    summary = (df.groupby(["n_train", "kernel"])
                 .agg(accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
                      recall_drone_mean=("recall_drone", "mean"), recall_drone_std=("recall_drone", "std"),
                      pr_auc_drone_mean=("pr_auc_drone", "mean"), pr_auc_drone_std=("pr_auc_drone", "std"))
                 .reset_index())
    print(summary.to_string(index=False))

    paired_note = {}
    for n in QUANTUM_TRAIN_SIZES:
        q = df[(df.n_train == n) & (df.kernel == "quantum_zz")].sort_values("seed").recall_drone.values
        r = df[(df.n_train == n) & (df.kernel == "rbf")].sort_values("seed").recall_drone.values
        diff = q - r
        paired_note[str(n)] = {
            "quantum_wins_n_of_5_seeds": int((diff > 0).sum()),
            "mean_diff_quantum_minus_rbf": float(diff.mean()),
            "std_diff": float(diff.std()),
        }
    print("paired quantum-vs-rbf drone recall, per seed:", paired_note)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for kernel, g in summary.groupby("kernel"):
        axes[0].errorbar(g.n_train, g.accuracy_mean, yerr=g.accuracy_std, marker="o", capsize=3, label=kernel)
        axes[1].errorbar(g.n_train, g.recall_drone_mean, yerr=g.recall_drone_std, marker="o", capsize=3, label=kernel)
    axes[0].set_xlabel("train size"); axes[0].set_ylabel("accuracy (mean +/- std, 5 seeds)")
    axes[0].set_title("Quantum vs classical kernels: accuracy, 5-seed repeat")
    axes[1].set_xlabel("train size"); axes[1].set_ylabel("drone recall (mean +/- std, 5 seeds)")
    axes[1].set_title("Drone recall: 5-seed repeat")
    for ax in axes:
        ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(C.PLOTS_DIR / "diat_usat_hybrid_quantum_multiseed.png", dpi=150)
    plt.close(fig)

    out = {
        "n_seeds": N_SEEDS,
        "note": "repeats the diat_usat_cnn.py kernel ablation across N_SEEDS "
                "independent random subsamples per train size, on the SAME "
                "frozen CNN embeddings (model not retrained), to check whether "
                "the single-seed quantum-wins-drone-recall finding is a real, "
                "scale-consistent effect or one lucky draw.",
        "per_seed_results": all_results,
        "summary_mean_std": summary.to_dict(orient="records"),
        "paired_quantum_vs_rbf_drone_recall": paired_note,
        "total_seconds": round(time.time() - t0, 1),
    }
    (C.REPORTS_DIR / "diat_usat_cnn_quantum_multiseed_results.json").write_text(json.dumps(out, indent=2))
    print(f"results -> {C.REPORTS_DIR / 'diat_usat_cnn_quantum_multiseed_results.json'}")


if __name__ == "__main__":
    main()
