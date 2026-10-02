"""Physics-informed CNN on RAD-DAR range-Doppler matrices -- an auxiliary
regression head, trained jointly with the classification head, forced to
predict physically-meaningful quantities computed DIRECTLY from each raw
matrix (doppler_velocity, micro_doppler_bandwidth, micro_doppler_jitter --
via raddar_experiments.matrix_features(), imported unmodified, the SAME
deterministic transform already used elsewhere in this project to bridge
RAD-DAR into the tabular schema).

Why this counts as "physics-informed" and not just "more features": the
auxiliary targets are NOT given to the network as inputs (that would just
be feature engineering) -- they're a second loss term the network must
predict FROM THE SAME RAW MATRIX INPUT AS THE CLASSIFICATION TASK,
regularizing the learned convolutional filters toward representations that
also encode real radar physics (Doppler centroid/spread/jitter), not just
whatever pattern happens to separate the two classes. No extra labels are
needed -- the auxiliary targets are self-supervised, computed
deterministically from the same matrix the network already sees.

Architecture: IDENTICAL backbone to raddar_cnn.py's build_model() (same
Conv2D/SE/pool stack, same 8-dim bottleneck) so any difference in result
reflects the added physics loss, not an architecture change. Two heads off
the shared bottleneck: classification (2-way softmax, same as before) and
auxiliary regression (3 physically-meaningful targets, z-scored using
TRAIN statistics so they contribute comparably to the combined loss
despite very different natural scales -- m/s vs. Hz vs. a [0,1] ratio).

Compared against raddar_cnn.py's own already-reported, real baseline
numbers (raddar_cnn_quantum_results.json's cnn_test_metrics) -- same
architecture/data/split/seed, not a fresh from-scratch matched run in this
session, disclosed as a caveat (TensorFlow training isn't perfectly
bit-reproducible run-to-run even with a fixed seed).
"""

import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
tf.config.run_functions_eagerly(True)
from qiskit.circuit.library import ZZFeatureMap
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight

from radarsense import config as C
from radarsense.diat_usat_cnn import se_block
from radarsense.raddar_cnn import batched_predict, fidelity_kernel, list_matrix_files, statevectors, stratified_subsample
from radarsense.raddar_experiments import matrix_features

BOTTLENECK_DIM = 8
QUANTUM_REPS = 2
QUANTUM_TRAIN_SIZES = [200, 400, 800]
QUANTUM_TEST_SIZE = 300
N_SEEDS = 5
EPOCHS = 30
BATCH = 32
AUX_TARGETS = ["doppler_velocity", "micro_doppler_bandwidth", "micro_doppler_jitter"]
LAMBDA_AUX = 1.0


def load_matrices_with_physics(rows):
    """Same loading/standardization as raddar_cnn.py's load_matrices, PLUS
    per-sample physics auxiliary targets computed from the RAW (pre-
    standardization) matrix via raddar_experiments.matrix_features."""
    X_raw, y, sessions, aux = [], [], [], []
    for fp, label, sess in rows:
        m = np.loadtxt(fp, delimiter=",")
        if m.shape != (11, 61):
            continue
        f = matrix_features(m)
        aux.append([f[t] for t in AUX_TARGETS])
        X_raw.append(m)
        y.append(label)
        sessions.append(sess)
    X_raw = np.stack(X_raw).astype(np.float32)
    X = (X_raw - X_raw.mean()) / (X_raw.std() + 1e-6)
    X = X[..., None]
    return X, np.array(y), np.array(sessions), np.array(aux, dtype=np.float32)


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

    class_out = tf.keras.layers.Dense(2, activation="softmax", name="class_out")(x)
    aux_out = tf.keras.layers.Dense(len(AUX_TARGETS), activation="linear", name="aux_out")(bottleneck)

    model = tf.keras.Model(inp, [class_out, aux_out])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss={"class_out": "sparse_categorical_crossentropy", "aux_out": "mse"},
        loss_weights={"class_out": 1.0, "aux_out": LAMBDA_AUX},
        metrics={"class_out": "accuracy"},
    )
    return model


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
    print("loading RAD-DAR range-Doppler matrices + physics auxiliary targets...")
    rows = list_matrix_files()
    X, y_str, sessions, aux_raw = load_matrices_with_physics(rows)
    y = (y_str == "drone").astype(int)
    print(f"loaded {len(X)} matrices | {dict(zip(*np.unique(y_str, return_counts=True)))}")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=C.SEED)
    tr_i, rest_i = next(gss.split(X, groups=sessions))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=C.SEED)
    va_i_rel, te_i_rel = next(gss2.split(rest_i, groups=sessions[rest_i]))
    va_i, te_i = rest_i[va_i_rel], rest_i[te_i_rel]

    Xtr, ytr, auxtr = X[tr_i], y[tr_i], aux_raw[tr_i]
    Xva, yva, auxva = X[va_i], y[va_i], aux_raw[va_i]
    Xte, yte, auxte = X[te_i], y[te_i], aux_raw[te_i]
    print(f"split: train={len(Xtr)} val={len(Xva)} test={len(Xte)}")

    aux_mean, aux_std = auxtr.mean(axis=0), auxtr.std(axis=0) + 1e-6
    auxtr_z = (auxtr - aux_mean) / aux_std
    auxva_z = (auxva - aux_mean) / aux_std
    print(f"auxiliary target stats (train): mean={aux_mean.round(2)} std={aux_std.round(2)}")

    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=ytr)
    class_weight = {0: float(cw[0]), 1: float(cw[1])}
    sample_weight_tr = np.array([class_weight[label] for label in ytr], dtype=np.float32)

    model = build_model()
    rng = np.random.default_rng(C.SEED)
    n = len(Xtr)
    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        losses, class_accs = [], []
        for start in range(0, n, BATCH):
            idx = order[start:start + BATCH]
            loss = model.train_on_batch(
                Xtr[idx], {"class_out": ytr[idx], "aux_out": auxtr_z[idx]},
                sample_weight=sample_weight_tr[idx], return_dict=True)
            losses.append(loss["loss"]); class_accs.append(loss["class_out_accuracy"])
        val_loss = model.test_on_batch(Xva, {"class_out": yva, "aux_out": auxva_z}, return_dict=True)
        print(f"Epoch {epoch+1}/{EPOCHS} - loss: {np.mean(losses):.4f} "
              f"class_acc: {np.mean(class_accs):.4f} - val_loss: {val_loss['loss']:.4f} "
              f"val_class_acc: {val_loss['class_out_accuracy']:.4f}", flush=True)

    print("[checkpoint] training loop done, running test predict()...", flush=True)
    proba_test, aux_pred_test = batched_predict_multi(model, Xte)
    proba_test = proba_test[:, 1]
    print("[checkpoint] test predict() done", flush=True)
    pred_test = (proba_test >= 0.5).astype(int)
    cnn_metrics = binary_metrics(yte, pred_test, proba_test)
    aux_mse = float(np.mean((aux_pred_test - (auxte - aux_mean) / aux_std) ** 2))
    print(f"\nphysics-informed CNN test: acc={cnn_metrics['accuracy']:.3f} "
          f"drone_recall={cnn_metrics['recall_drone']:.3f} "
          f"pr_auc={cnn_metrics['pr_auc_drone']:.3f} aux_mse(z)={aux_mse:.3f}")

    print("[checkpoint] saving model...", flush=True)
    model.save(C.MODELS_DIR / "raddar_physics_informed_cnn.keras")

    print("\nextracting frozen bottleneck embeddings...", flush=True)
    encoder = tf.keras.Model(model.input, model.get_layer("bottleneck").output)
    Ztr = batched_predict(encoder, Xtr)
    Zte = batched_predict(encoder, Xte)
    print("[checkpoint] embeddings done", flush=True)

    fmap = ZZFeatureMap(feature_dimension=BOTTLENECK_DIM, reps=QUANTUM_REPS)
    circuit_meta = {"feature_map": "ZZFeatureMap", "qubits": BOTTLENECK_DIM,
                    "reps": QUANTUM_REPS, "circuit_depth": int(fmap.decompose().depth()),
                    "backend": "Qiskit exact statevector simulation (noiseless)"}

    all_results = []
    t0 = time.time()
    for n_train in QUANTUM_TRAIN_SIZES:
        for seed in range(N_SEEDS):
            rng_q = C.SEED + seed
            Ztr_sub, ytr_sub = stratified_subsample(Ztr, ytr, n_train, rng_q)
            Zte_sub, yte_sub = stratified_subsample(Zte, yte, QUANTUM_TEST_SIZE, rng_q)
            scaler = MinMaxScaler(feature_range=(0.0, 2.0)).fit(Ztr_sub)
            Ztr_s, Zte_s = scaler.transform(Ztr_sub), scaler.transform(Zte_sub)

            S_tr, S_te = statevectors(fmap, Ztr_s), statevectors(fmap, Zte_s)
            Ktr, Kte = fidelity_kernel(S_tr), fidelity_kernel(S_te, S_tr)
            qsvc = SVC(kernel="precomputed", C=1.0, probability=True, random_state=C.SEED)
            qsvc.fit(Ktr, ytr_sub)
            proba_q = qsvc.predict_proba(Kte)[:, list(qsvc.classes_).index(1)]
            m = binary_metrics(yte_sub, qsvc.predict(Kte), proba_q)
            m.update({"kernel": "quantum_zz", "n_train": n_train, "seed": seed})
            all_results.append(m)

            csvc = SVC(kernel="rbf", C=1.0, probability=True, random_state=C.SEED)
            csvc.fit(Ztr_s, ytr_sub)
            proba_c = csvc.predict_proba(Zte_s)[:, list(csvc.classes_).index(1)]
            m = binary_metrics(yte_sub, csvc.predict(Zte_s), proba_c)
            m.update({"kernel": "rbf", "n_train": n_train, "seed": seed})
            all_results.append(m)
        print(f"n={n_train}: done {N_SEEDS} seeds ({time.time()-t0:.0f}s elapsed)")

    df = pd.DataFrame(all_results)
    summary = (df.groupby(["n_train", "kernel"])
                 .agg(accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
                      recall_drone_mean=("recall_drone", "mean"), recall_drone_std=("recall_drone", "std"))
                 .reset_index())
    print("\n" + summary.to_string(index=False))

    baseline = json.loads((C.REPORTS_DIR / "raddar_cnn_quantum_results.json").read_text())
    base_cnn = baseline["cnn_test_metrics"]
    print(f"\n--- head-to-head vs. pure-classification baseline (same architecture/data/split) ---")
    print(f"  baseline (no physics loss): acc={base_cnn['accuracy']:.4f} drone_recall={base_cnn['recall_drone']:.4f}")
    print(f"  physics-informed:           acc={cnn_metrics['accuracy']:.4f} drone_recall={cnn_metrics['recall_drone']:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for kernel, g in summary.groupby("kernel"):
        axes[0].errorbar(g.n_train, g.accuracy_mean, yerr=g.accuracy_std, marker="o", capsize=3, label=kernel)
        axes[1].errorbar(g.n_train, g.recall_drone_mean, yerr=g.recall_drone_std, marker="o", capsize=3, label=kernel)
    axes[0].axhline(cnn_metrics["accuracy"], color="gray", ls="--", label="physics-informed CNN (direct)")
    axes[0].set_xlabel("train size"); axes[0].set_ylabel("accuracy (mean +/- std, 5 seeds)")
    axes[0].set_title("Physics-informed RAD-DAR: quantum vs classical kernels on CNN embeddings")
    axes[1].set_xlabel("train size"); axes[1].set_ylabel("drone recall (mean +/- std, 5 seeds)")
    axes[1].set_title("Drone recall: 5-seed repeat")
    for ax in axes:
        ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(C.PLOTS_DIR / "raddar_physics_informed_cnn_quantum_multiseed.png", dpi=150)
    plt.close(fig)

    out = {
        "auxiliary_targets": AUX_TARGETS,
        "lambda_aux": LAMBDA_AUX,
        "split_sizes": {"train": len(Xtr), "val": len(Xva), "test": len(Xte)},
        "cnn_test_metrics": cnn_metrics,
        "auxiliary_test_mse_zscored": aux_mse,
        "circuit": circuit_meta,
        "summary_mean_std": summary.to_dict(orient="records"),
        "comparison_vs_pure_classification_baseline": {
            "baseline_no_physics_loss": {"accuracy": base_cnn["accuracy"], "recall_drone": base_cnn["recall_drone"]},
            "physics_informed": {"accuracy": cnn_metrics["accuracy"], "recall_drone": cnn_metrics["recall_drone"]},
        },
        "total_seconds": round(time.time() - t0, 1),
    }
    (C.REPORTS_DIR / "raddar_physics_informed_cnn_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nresults -> {C.REPORTS_DIR / 'raddar_physics_informed_cnn_results.json'}")


def batched_predict_multi(model, X, batch=256):
    class_out, aux_out = [], []
    for start in range(0, len(X), batch):
        c, a = model(X[start:start + batch], training=False)
        class_out.append(c.numpy()); aux_out.append(a.numpy())
    return np.concatenate(class_out, axis=0), np.concatenate(aux_out, axis=0)


if __name__ == "__main__":
    main()
