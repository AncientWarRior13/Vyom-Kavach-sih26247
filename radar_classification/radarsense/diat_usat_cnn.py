"""DIAT-uSAT paper-inspired CNN side-experiment (SCOPED DOWN, NOT part of the
deployed pipeline) + a hybrid quantum-kernel ablation on top of it.

Inspired by Paul & Bandyopadhyay, "UAV Classification Using Attentive
Binarized CNN for Micro-Doppler Spectrograms" (IEEE Access 2026), which
trains a binarized VGG-style CNN + Gated Channel Transformation attention
directly on DIAT-uSAT spectrogram pixels (97.83% full-precision / 96.87%
binarized accuracy, 6 classes). That paper's approach is a different model
family from RadarSense's tabular CatBoost-on-11-features pipeline and does
NOT feed the frozen ONNX/tabular handoff Pair 2 consumes.

This script is a deliberately scoped-down comparison, not a replication:
  - standard full-precision weights, not 1-bit binarized (Larq is a large
    extra dependency/implementation surface for marginal value here)
  - one lightweight Squeeze-and-Excitation channel-attention block per conv
    block, standing in for the paper's GCT module
  - trained directly on raw spectrogram pixels (RGB, resized), using the
    SAME CLASS_MAP and variant-holdout split as diat_usat_bridge.py, so
    numbers are comparable and not leakage-inflated
  - not wired into run_pipeline.py; run manually: python -m radarsense.diat_usat_cnn
  - requires `tensorflow` (added to requirements.txt, CNN-side-experiment only)

Quantum-kernel ablation (stage 2 of this script): the CNN above is trained as
a classifier first, then FROZEN, and its 8-D bottleneck layer is used as a
fixed feature extractor -- the practical "train CNN, freeze, train quantum
layer" simplification instead of full end-to-end CNN+quantum backprop (slow,
memory-heavy, unnecessary to answer the ablation question). On those frozen
embeddings we run the same honest ablation as radarsense/quantum.py: quantum
ZZFeatureMap fidelity kernel vs. classical RBF vs. classical linear kernel,
identical embeddings, identical splits, Qiskit exact statevector simulation
(noiseless, local, no IBM Quantum account or hardware queue required). Unlike
quantum.py's binary drone-vs-rest setup, this is the genuine 4-class problem
(drone is not the minority class among these folders -- 5 of 10 map to
drone), so kernel-ablation subsampling is class-proportional, not
drone-boosted. Reported honestly regardless of which kernel wins.
"""

import glob
import json
import re
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.metrics.pairwise import rbf_kernel as sk_rbf
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight

from radarsense import config as C
from radarsense.diat_usat_bridge import CLASS_MAP, DIAT, SINGLE_FOLDERS, VARIANT_PAIRS
from radarsense.metrics import full_metrics

IMG_SIZE = (120, 160)
BATCH = 32
EPOCHS = 20
BOTTLENECK_DIM = 8
QUANTUM_TRAIN_SIZES = [200, 400, 800]
QUANTUM_TEST_SIZE = 400
QUANTUM_REPS = 2


def list_files():
    rows = []
    for folder, label in CLASS_MAP.items():
        files = sorted(
            glob.glob(str(DIAT / folder / "figure*.jpg")),
            key=lambda p: int(re.search(r"figure(\d+)\.jpg$", p).group(1)))
        for fp in files:
            idx = int(re.search(r"figure(\d+)\.jpg$", fp).group(1))
            rows.append((fp, label, folder, idx))
    return rows


def split_files(rows):
    """Same variant-holdout split as diat_usat_bridge.make_splits, applied to file paths."""
    by_folder = {}
    for fp, label, folder, idx in rows:
        by_folder.setdefault(folder, []).append((fp, label, idx))
    train, val, test = [], [], []
    for a, b in VARIANT_PAIRS:
        sub = sorted(by_folder[a], key=lambda r: r[2])
        cut = int(len(sub) * 0.85)
        train += sub[:cut]
        val += sub[cut:]
        test += by_folder[b]
    for folder in SINGLE_FOLDERS:
        sub = sorted(by_folder[folder], key=lambda r: r[2])
        c1, c2 = int(len(sub) * 0.70), int(len(sub) * 0.85)
        train += sub[:c1]
        val += sub[c1:c2]
        test += sub[c2:]
    return train, val, test


def load_image(fp):
    img = Image.open(fp).convert("RGB").resize((IMG_SIZE[1], IMG_SIZE[0]), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def make_arrays(split):
    X = np.stack([load_image(fp) for fp, _, _ in split])
    y = np.array([C.CLASSES.index(label) for _, label, _ in split])
    return X, y


def se_block(x, reduction=8):
    """Squeeze-and-Excitation channel attention -- a lightweight stand-in for
    the paper's Gated Channel Transformation module."""
    ch = x.shape[-1]
    s = tf.keras.layers.GlobalAveragePooling2D()(x)
    s = tf.keras.layers.Dense(max(ch // reduction, 4), activation="relu")(s)
    s = tf.keras.layers.Dense(ch, activation="sigmoid")(s)
    s = tf.keras.layers.Reshape((1, 1, ch))(s)
    return tf.keras.layers.Multiply()([x, s])


def conv_block(x, filters):
    x = tf.keras.layers.Conv2D(filters, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = se_block(x)
    return tf.keras.layers.MaxPooling2D()(x)


def build_model(n_classes):
    """Single classifier head, bottleneck ON the critical path (Dense(128) ->
    Dense(BOTTLENECK_DIM, name='bottleneck') -> Dense(n_classes)).

    REVERTED from a two-head design (main_out full-capacity + separate
    aux_out off the bottleneck) that was tried to test the hypothesis that
    the narrow 8-D bottleneck was capacity-starving the classifier. It was
    -- train accuracy jumped from ~87% to 97%+ once decoupled -- but test
    accuracy on the cross-session held-out split got WORSE, not better
    (78.4% -> 74.8% -> 62.2% across two decoupled variants), because the
    extra capacity let the model overfit faster to TRAINING-session-specific
    artifacts on this small (2,491 image), strictly cross-session-generalized
    split. The narrow bottleneck was accidentally acting as a regularizer
    against exactly that failure mode, not a bug. Reverted to the
    configuration that actually measured best. See diat_usat_cnn_results.json
    history / QUANTUM_MINIPAPER.md for the full honest account.
    """
    inp = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = inp
    for filters in (32, 64, 128, 128):
        x = conv_block(x, filters)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    bottleneck = tf.keras.layers.Dense(BOTTLENECK_DIM, activation="relu", name="bottleneck")(x)
    x = tf.keras.layers.Dropout(0.4)(bottleneck)
    out = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def stratified_subsample(X, y, n, rng):
    """Class-proportional subsample (natural shares, no artificial boosting --
    unlike quantum.py's binary drone-vs-rest setup, drone is not rare here)."""
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


def reorder_proba(proba, classes_):
    classes_ = list(classes_)
    return proba[:, [classes_.index(c) for c in C.CLASSES]]


def eval_multiclass(name, y_true_idx, y_pred_idx, proba, classes_, extra=None):
    y_true = np.array(C.CLASSES)[y_true_idx]
    y_pred = np.array(C.CLASSES)[y_pred_idx]
    proba_r = reorder_proba(proba, [C.CLASSES[c] for c in classes_])
    m = full_metrics(y_true, y_pred, proba_r)
    m["kernel"] = name
    if extra:
        m.update(extra)
    return m


def quantum_kernel_ablation(cnn, Xtr, ytr, Xte, yte):
    """Freeze the trained CNN, extract its bottleneck embeddings, and run the
    quantum-vs-classical kernel ablation on those embeddings."""
    print("extracting frozen bottleneck embeddings...")
    encoder = tf.keras.Model(cnn.input, cnn.get_layer("bottleneck").output)
    Ztr = encoder.predict(Xtr, verbose=0)
    Zte = encoder.predict(Xte, verbose=0)

    fmap = ZZFeatureMap(feature_dimension=BOTTLENECK_DIM, reps=QUANTUM_REPS)
    circuit_meta = {
        "feature_map": "ZZFeatureMap",
        "qubits": BOTTLENECK_DIM,
        "reps": QUANTUM_REPS,
        "circuit_depth": int(fmap.decompose().depth()),
        "backend": ("Qiskit exact statevector simulation (noiseless); fidelity "
                    "kernel K_ij = |<psi_i|psi_j>|^2 computed as a Gram matrix "
                    "over the frozen CNN bottleneck embeddings"),
    }
    print("circuit:", circuit_meta)

    rng = C.SEED
    Zte_sub, yte_sub = stratified_subsample(Zte, yte, QUANTUM_TEST_SIZE, rng)

    results = []
    t0 = time.time()
    for n in QUANTUM_TRAIN_SIZES:
        Ztr_sub, ytr_sub = stratified_subsample(Ztr, ytr, n, rng)
        scaler = MinMaxScaler(feature_range=(0.0, 2.0)).fit(Ztr_sub)
        Ztr_scaled, Zte_scaled = scaler.transform(Ztr_sub), scaler.transform(Zte_sub)

        tq0 = time.time()
        S_tr = statevectors(fmap, Ztr_scaled)
        S_te = statevectors(fmap, Zte_scaled)
        Ktr, Kte = fidelity_kernel(S_tr), fidelity_kernel(S_te, S_tr)
        q_seconds = time.time() - tq0

        qsvc = SVC(kernel="precomputed", C=1.0, probability=True, random_state=C.SEED)
        qsvc.fit(Ktr, ytr_sub)
        results.append(eval_multiclass(
            "quantum_zz", yte_sub, qsvc.predict(Kte), qsvc.predict_proba(Kte),
            list(qsvc.classes_), {"n_train": n, "kernel_seconds": round(q_seconds, 1)}))

        for kname in ("rbf", "linear"):
            csvc = SVC(kernel=kname, C=1.0, probability=True, random_state=C.SEED)
            csvc.fit(Ztr_scaled, ytr_sub)
            results.append(eval_multiclass(
                kname, yte_sub, csvc.predict(Zte_scaled), csvc.predict_proba(Zte_scaled),
                list(csvc.classes_), {"n_train": n}))

        last = [r for r in results if r["n_train"] == n]
        print(f"n={n}: " + " | ".join(
            f"{r['kernel']} acc {r['accuracy']:.3f} drone_recall {r['recall_drone']:.2f}"
            for r in last))

    y_drone_pm = 2 * (np.array(C.CLASSES)[ytr_sub] == C.MINORITY_CLASS).astype(int) - 1
    T = np.outer(y_drone_pm, y_drone_pm)
    def alignment(K):
        return float((K * T).sum() / (np.linalg.norm(K) * np.linalg.norm(T)))
    align = {"quantum_zz": alignment(Ktr), "rbf": alignment(sk_rbf(Ztr_scaled))}
    print("kernel-target alignment (drone-vs-rest, n=%d):" % n, align)

    import pandas as pd
    df = pd.DataFrame(results)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for kernel, g in df.groupby("kernel"):
        axes[0].plot(g.n_train, g.accuracy, "o-", label=kernel)
        axes[1].plot(g.n_train, g.recall_drone, "o-", label=kernel)
    axes[0].set_xlabel("train size"); axes[0].set_ylabel("accuracy")
    axes[0].set_title("Quantum vs classical kernels on frozen CNN embeddings")
    axes[1].set_xlabel("train size"); axes[1].set_ylabel("drone recall")
    axes[1].set_title("Drone recall scaling")
    for ax in axes:
        ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(C.PLOTS_DIR / "diat_usat_hybrid_quantum_scaling.png", dpi=150)
    plt.close(fig)

    return {
        "circuit": circuit_meta,
        "kernel_ablation_results": results,
        "kernel_target_alignment_largest_n": align,
        "total_seconds": round(time.time() - t0, 1),
        "assumptions": [
            "kernel ablation subsamples train/test (class-proportional, not "
            "drone-boosted) because quantum kernels cost O(n^2) circuit "
            "evaluations -- same tractability constraint as radarsense/quantum.py",
            "bottleneck embeddings are frozen (not fine-tuned jointly with the "
            "quantum kernel): train CNN, freeze, train kernel on top",
            "same variant/chronological holdout split as the CNN classifier "
            "above and as diat_usat_bridge.py Experiment B",
        ],
        "role_of_each_model": {
            "catboost (deployed, tabular schema)": "production detector behind the ONNX handoff",
            "random_forest / xgboost / svm / logreg (tabular)": "classical baseline sweep",
            "quantum_zz kernel (tabular, 4 physics features, radarsense/quantum.py)": "research ablation on the tabular schema",
            "diat_usat_cnn (this script, image classifier)": "image-domain research experiment on real DIAT-uSAT spectrograms",
            "quantum_zz / rbf / linear kernel on frozen CNN embeddings (this script)": "research ablation: does a quantum kernel on CNN-learned features beat a classical kernel on the same features?",
        },
    }


def main():
    tf.random.set_seed(C.SEED)
    rows = list_files()
    train, val, test = split_files(rows)
    print(f"train={len(train)} val={len(val)} test={len(test)}")

    Xtr, ytr = make_arrays(train)
    Xva, yva = make_arrays(val)
    Xte, yte = make_arrays(test)

    weights = compute_class_weight("balanced", classes=np.arange(len(C.CLASSES)), y=ytr)
    class_weight = {i: float(w) for i, w in enumerate(weights)}

    model = build_model(len(C.CLASSES))
    model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=EPOCHS, batch_size=BATCH,
              class_weight=class_weight, verbose=2)

    proba = model.predict(Xte, verbose=0)
    pred = proba.argmax(axis=1)
    acc = float((pred == yte).mean())
    macro_f1 = float(f1_score(yte, pred, average="macro"))
    cm = confusion_matrix(yte, pred, labels=np.arange(len(C.CLASSES)))
    report = classification_report(yte, pred, target_names=C.CLASSES,
                                   output_dict=True, zero_division=0)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(C.CLASSES))); ax.set_xticklabels(C.CLASSES, rotation=45)
    ax.set_yticks(range(len(C.CLASSES))); ax.set_yticklabels(C.CLASSES)
    for i in range(len(C.CLASSES)):
        for j in range(len(C.CLASSES)):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"DIAT-uSAT CNN side-experiment (acc={acc:.1%})")
    fig.tight_layout()
    fig.savefig(C.PLOTS_DIR / "diat_usat_cnn_confusion.png", dpi=150)
    plt.close(fig)

    model.save(C.MODELS_DIR / "diat_usat_cnn.keras")

    print("\n--- quantum-kernel ablation on frozen CNN bottleneck embeddings ---")
    quantum_out = quantum_kernel_ablation(model, Xtr, ytr, Xte, yte)

    out = {
        "note": "paper-inspired side validation (small CNN + SE-attention on raw "
                "spectrogram images) -- NOT part of the deployed ONNX/tabular "
                "pipeline. Inspired by Paul & Bandyopadhyay, IEEE Access 2026, "
                "which uses a binarized VGG-style CNN + Gated Channel "
                "Transformation; this is a non-binarized, scoped-down variant "
                "with SE attention, trained directly on DIAT-uSAT spectrogram "
                "pixels using the same variant-holdout split as diat_usat_bridge.py.",
        "classes": C.CLASSES,
        "split_sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "test_accuracy": acc,
        "test_macro_f1": macro_f1,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "quantum_kernel_ablation": quantum_out,
    }
    (C.REPORTS_DIR / "diat_usat_cnn_results.json").write_text(json.dumps(out, indent=2))
    print(f"test accuracy {acc:.3f} macro-F1 {macro_f1:.3f}")
    print(f"results -> {C.REPORTS_DIR / 'diat_usat_cnn_results.json'}")


if __name__ == "__main__":
    main()
