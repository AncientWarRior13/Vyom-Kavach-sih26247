"""RAD-DAR CNN on raw range-Doppler matrices + quantum-kernel-on-embeddings
ablation -- a REPLICATION ATTEMPT of the diat_usat_cnn.py finding (quantum
kernel beats classical on CNN-learned embeddings of real spectrogram
images) on a SECOND, independent real sensor with genuinely different
physics: RAD-DAR is 8.75GHz FMCW (vs. DIAT-uSAT's X-band CW), range-Doppler
MATRICES (11 range x 61 Doppler cells, vs. DIAT-uSAT's rendered spectrogram
images), and a BINARY task (drone vs. anomaly -- RAD-DAR has no bird/
aircraft classes, per raddar_experiments.py's own CLASS_MAP: Drones->drone,
Cars+People->anomaly).

Same split discipline as raddar_experiments.py: session-aware
GroupShuffleSplit (no session spans train/val/test), reused directly for
leakage-avoidance consistency.

Reported honestly regardless of outcome: if the quantum-kernel-wins-drone-
recall pattern replicates here, that's two independent real sensors
showing the same effect -- if it doesn't replicate, that's an equally
legitimate, and arguably more important, honest finding (the DIAT-uSAT
result would not generalize past one dataset).
"""

import glob
import json
import re
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
tf.config.run_functions_eagerly(True)
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight

from radarsense import config as C
from radarsense.diat_usat_cnn import se_block
from radarsense.raddar_experiments import RADDAR, CLASS_MAP

BOTTLENECK_DIM = 8
QUANTUM_REPS = 2
QUANTUM_TRAIN_SIZES = [200, 400, 800]
QUANTUM_TEST_SIZE = 300
N_SEEDS = 5
EPOCHS = 30
BATCH = 32


def list_matrix_files():
    rows = []
    for cls_dir, label in CLASS_MAP.items():
        for sess_path in sorted(glob.glob(str(RADDAR / cls_dir / "*"))):
            sess = f"{cls_dir}/{sess_path.split('/')[-1]}"
            files = sorted(glob.glob(f"{sess_path}/*.csv"),
                           key=lambda p: int(re.findall(r"(\d+)\.csv$", p)[0]))
            for fp in files:
                rows.append((fp, label, sess))
    return rows


def load_matrices(rows):
    X, y, sessions = [], [], []
    for fp, label, sess in rows:
        m = np.loadtxt(fp, delimiter=",")
        if m.shape != (11, 61):
            continue
        X.append(m)
        y.append(label)
        sessions.append(sess)
    X = np.stack(X).astype(np.float32)
    X = (X - X.mean()) / (X.std() + 1e-6)
    X = X[..., None]
    return X, np.array(y), np.array(sessions)


def build_model():
    inp = tf.keras.Input(shape=(11, 61, 1))
    x = tf.keras.layers.Conv2D(16, 3, padding="same")(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = se_block(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(1, 2))(x)

    x = tf.keras.layers.Conv2D(32, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = se_block(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(1, 2))(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    bottleneck = tf.keras.layers.Dense(BOTTLENECK_DIM, activation="relu", name="bottleneck")(x)
    x = tf.keras.layers.Dropout(0.3)(bottleneck)
    out = tf.keras.layers.Dense(2, activation="softmax")(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def stratified_subsample(X, y, n, rng):
    rng = np.random.RandomState(rng)
    classes, counts = np.unique(y, return_counts=True)
    target = np.maximum(1, np.round(counts / counts.sum() * n).astype(int))
    target[np.argmax(target)] += n - target.sum()
    idx = []
    for cls, cnt in zip(classes, target):
        cls_idx = np.where(y == cls)[0]
        idx.append(rng.choice(cls_idx, size=min(cnt, len(cls_idx)), replace=False))
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return X[idx], y[idx]


def statevectors(fmap, X):
    return np.stack([Statevector(fmap.assign_parameters(x)).data for x in X])


def fidelity_kernel(S_a, S_b=None):
    S_b = S_a if S_b is None else S_b
    return np.abs(S_a.conj() @ S_b.T) ** 2


def batched_predict(keras_model, X, batch=256):
    out = []
    for start in range(0, len(X), batch):
        out.append(keras_model(X[start:start + batch], training=False).numpy())
    return np.concatenate(out, axis=0)


def binary_metrics(y_true, y_pred, y_score):
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "recall_drone": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision_drone": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_drone": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "pr_auc_drone": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
    }


def main():
    print("loading RAD-DAR range-Doppler matrices + session-aware split...")
    rows = list_matrix_files()
    X, y_str, sessions = load_matrices(rows)
    y = (y_str == "drone").astype(int)
    print(f"loaded {len(X)} matrices | {dict(zip(*np.unique(y_str, return_counts=True)))}")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=C.SEED)
    tr_i, rest_i = next(gss.split(X, groups=sessions))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=C.SEED)
    va_i_rel, te_i_rel = next(gss2.split(rest_i, groups=sessions[rest_i]))
    va_i, te_i = rest_i[va_i_rel], rest_i[te_i_rel]

    Xtr, ytr = X[tr_i], y[tr_i]
    Xva, yva = X[va_i], y[va_i]
    Xte, yte = X[te_i], y[te_i]
    print(f"split: train={len(Xtr)} val={len(Xva)} test={len(Xte)} "
          f"(sessions: {len(set(sessions[tr_i]))}/{len(set(sessions[va_i]))}/{len(set(sessions[te_i]))})")

    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=ytr)
    class_weight = {0: float(cw[0]), 1: float(cw[1])}
    sample_weight_tr = np.array([class_weight[label] for label in ytr], dtype=np.float32)

    model = build_model()
    rng = np.random.default_rng(C.SEED)
    n = len(Xtr)
    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        losses, accs = [], []
        for start in range(0, n, BATCH):
            idx = order[start:start + BATCH]
            loss, acc = model.train_on_batch(Xtr[idx], ytr[idx], sample_weight=sample_weight_tr[idx])
            losses.append(loss); accs.append(acc)
        val_loss, val_acc = model.test_on_batch(Xva, yva)
        print(f"Epoch {epoch+1}/{EPOCHS} - loss: {np.mean(losses):.4f} "
              f"acc: {np.mean(accs):.4f} - val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}", flush=True)

    print("[checkpoint] training loop done, running test predict()...", flush=True)
    proba_test = batched_predict(model, Xte)[:, 1]
    print("[checkpoint] test predict() done", flush=True)
    pred_test = (proba_test >= 0.5).astype(int)
    cnn_metrics = binary_metrics(yte, pred_test, proba_test)
    print(f"\nCNN classifier test: acc={cnn_metrics['accuracy']:.3f} "
          f"drone_recall={cnn_metrics['recall_drone']:.3f} "
          f"pr_auc={cnn_metrics['pr_auc_drone']:.3f}")

    print("[checkpoint] saving model...", flush=True)
    model.save(C.MODELS_DIR / "raddar_cnn.keras")
    print("[checkpoint] model saved", flush=True)

    print("\nextracting frozen bottleneck embeddings...", flush=True)
    encoder = tf.keras.Model(model.input, model.get_layer("bottleneck").output)
    print("[checkpoint] encoder built, predicting Ztr...", flush=True)
    Ztr = batched_predict(encoder, Xtr)
    print("[checkpoint] Ztr done, predicting Zte...", flush=True)
    Zte = batched_predict(encoder, Xte)
    print("[checkpoint] Zte done", flush=True)

    fmap = ZZFeatureMap(feature_dimension=BOTTLENECK_DIM, reps=QUANTUM_REPS)
    circuit_meta = {"feature_map": "ZZFeatureMap", "qubits": BOTTLENECK_DIM,
                    "reps": QUANTUM_REPS, "circuit_depth": int(fmap.decompose().depth()),
                    "backend": "Qiskit exact statevector simulation (noiseless)"}
    print("circuit:", circuit_meta, flush=True)

    all_results = []
    t0 = time.time()
    for n in QUANTUM_TRAIN_SIZES:
        for seed in range(N_SEEDS):
            print(f"[checkpoint] starting n={n} seed={seed}", flush=True)
            rng = C.SEED + seed
            Ztr_sub, ytr_sub = stratified_subsample(Ztr, ytr, n, rng)
            Zte_sub, yte_sub = stratified_subsample(Zte, yte, QUANTUM_TEST_SIZE, rng)
            scaler = MinMaxScaler(feature_range=(0.0, 2.0)).fit(Ztr_sub)
            Ztr_s, Zte_s = scaler.transform(Ztr_sub), scaler.transform(Zte_sub)

            S_tr, S_te = statevectors(fmap, Ztr_s), statevectors(fmap, Zte_s)
            Ktr, Kte = fidelity_kernel(S_tr), fidelity_kernel(S_te, S_tr)
            qsvc = SVC(kernel="precomputed", C=1.0, probability=True, random_state=C.SEED)
            qsvc.fit(Ktr, ytr_sub)
            proba_q = qsvc.predict_proba(Kte)[:, list(qsvc.classes_).index(1)]
            m = binary_metrics(yte_sub, qsvc.predict(Kte), proba_q)
            m.update({"kernel": "quantum_zz", "n_train": n, "seed": seed})
            all_results.append(m)

            for kname in ("rbf", "linear"):
                csvc = SVC(kernel=kname, C=1.0, probability=True, random_state=C.SEED)
                csvc.fit(Ztr_s, ytr_sub)
                proba_c = csvc.predict_proba(Zte_s)[:, list(csvc.classes_).index(1)]
                m = binary_metrics(yte_sub, csvc.predict(Zte_s), proba_c)
                m.update({"kernel": kname, "n_train": n, "seed": seed})
                all_results.append(m)
        print(f"n={n}: done {N_SEEDS} seeds ({time.time()-t0:.0f}s elapsed)")

    df = pd.DataFrame(all_results)
    summary = (df.groupby(["n_train", "kernel"])
                 .agg(accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
                      recall_drone_mean=("recall_drone", "mean"), recall_drone_std=("recall_drone", "std"),
                      pr_auc_mean=("pr_auc_drone", "mean"), pr_auc_std=("pr_auc_drone", "std"))
                 .reset_index())
    print("\n" + summary.to_string(index=False))

    paired = {}
    for n in QUANTUM_TRAIN_SIZES:
        q = df[(df.n_train == n) & (df.kernel == "quantum_zz")].sort_values("seed").recall_drone.values
        r = df[(df.n_train == n) & (df.kernel == "rbf")].sort_values("seed").recall_drone.values
        diff = q - r
        paired[str(n)] = {"quantum_wins_n_of_5_seeds": int((diff > 0).sum()),
                          "mean_diff_quantum_minus_rbf": float(diff.mean()),
                          "std_diff": float(diff.std())}
    print("\npaired quantum-vs-rbf drone recall:", paired)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for kernel, g in summary.groupby("kernel"):
        axes[0].errorbar(g.n_train, g.accuracy_mean, yerr=g.accuracy_std, marker="o", capsize=3, label=kernel)
        axes[1].errorbar(g.n_train, g.recall_drone_mean, yerr=g.recall_drone_std, marker="o", capsize=3, label=kernel)
    axes[0].set_xlabel("train size"); axes[0].set_ylabel("accuracy (mean +/- std, 5 seeds)")
    axes[0].set_title("RAD-DAR: quantum vs classical kernels on CNN embeddings")
    axes[1].set_xlabel("train size"); axes[1].set_ylabel("drone recall (mean +/- std, 5 seeds)")
    axes[1].set_title("Drone recall: 5-seed repeat")
    for ax in axes:
        ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(C.PLOTS_DIR / "raddar_cnn_quantum_multiseed.png", dpi=150)
    plt.close(fig)

    out = {
        "replication_of": "diat_usat_cnn.py quantum-kernel-on-CNN-embeddings finding, "
                          "second independent real sensor (RAD-DAR, 8.75GHz FMCW range-Doppler "
                          "matrices vs. DIAT-uSAT's X-band CW spectrogram images)",
        "n_seeds": N_SEEDS,
        "split_sizes": {"train": len(Xtr), "val": len(Xva), "test": len(Xte)},
        "cnn_test_metrics": cnn_metrics,
        "circuit": circuit_meta,
        "per_seed_results": all_results,
        "summary_mean_std": summary.to_dict(orient="records"),
        "paired_quantum_vs_rbf_drone_recall": paired,
        "total_seconds": round(time.time() - t0, 1),
    }
    (C.REPORTS_DIR / "raddar_cnn_quantum_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nresults -> {C.REPORTS_DIR / 'raddar_cnn_quantum_results.json'}")


if __name__ == "__main__":
    main()
