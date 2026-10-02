"""Stability check for diat_usat_cnn.py's classifier test accuracy.

Earlier same-architecture reruns (all with the identical bottleneck-on-
critical-path build_model(), before this fix) produced 81.1%, 78.4%, and
64.5% test accuracy on separate runs -- a 14+ point swing attributed at the
time to an architecture change, which a same-code rerun then disproved. The
root cause: EarlyStopping on a small (483-image), same-session, highly
volatile validation signal was picking wildly different checkpoints run to
run. diat_usat_cnn.py now trains a FIXED epoch count with no early stopping,
removing that lottery. This script verifies that actually worked: trains the
classifier 3 times with different seeds (fixed epochs, no early stopping,
quantum-kernel ablation skipped here for speed -- that stage was already
validated separately in diat_usat_cnn_quantum_multiseed_results.json) and
reports mean +/- std test accuracy, so the reported number has an honest
uncertainty estimate instead of being a single lucky/unlucky run.
"""

import json
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

from radarsense import config as C
from radarsense.diat_usat_cnn import (BATCH, EPOCHS, build_model, list_files,
                                       make_arrays, split_files)

N_REPEATS = 3


def main():
    rows = list_files()
    train, val, test = split_files(rows)
    Xtr, ytr = make_arrays(train)
    Xva, yva = make_arrays(val)
    Xte, yte = make_arrays(test)
    print(f"train={len(train)} val={len(val)} test={len(test)}")

    weights = compute_class_weight("balanced", classes=np.arange(len(C.CLASSES)), y=ytr)
    class_weight = {i: float(w) for i, w in enumerate(weights)}

    runs = []
    t0 = time.time()
    for seed in range(N_REPEATS):
        tf.random.set_seed(C.SEED + seed)
        model = build_model(len(C.CLASSES))
        tr0 = time.time()
        model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=EPOCHS, batch_size=BATCH,
                  class_weight=class_weight, verbose=0)
        proba = model.predict(Xte, verbose=0)
        pred = proba.argmax(axis=1)
        acc = float((pred == yte).mean())
        macro_f1 = float(f1_score(yte, pred, average="macro"))
        runs.append({"seed": seed, "test_accuracy": acc, "test_macro_f1": macro_f1,
                      "train_seconds": round(time.time() - tr0, 1)})
        print(f"seed {seed}: test_accuracy={acc:.3f} macro_f1={macro_f1:.3f} "
              f"({time.time()-tr0:.0f}s)")

    accs = [r["test_accuracy"] for r in runs]
    f1s = [r["test_macro_f1"] for r in runs]
    summary = {
        "n_repeats": N_REPEATS,
        "runs": runs,
        "test_accuracy_mean": float(np.mean(accs)),
        "test_accuracy_std": float(np.std(accs)),
        "test_accuracy_min": float(np.min(accs)),
        "test_accuracy_max": float(np.max(accs)),
        "test_macro_f1_mean": float(np.mean(f1s)),
        "test_macro_f1_std": float(np.std(f1s)),
        "total_seconds": round(time.time() - t0, 1),
        "note": ("Fixed-epoch training (no early stopping) across 3 seeds. "
                 "Compare test_accuracy_std here against the ~14-point spread "
                 "seen across early-stopped single runs to confirm removing "
                 "the validation-based checkpoint lottery actually reduced "
                 "run-to-run variance."),
    }
    print(f"\ntest_accuracy: {summary['test_accuracy_mean']:.3f} "
          f"+/- {summary['test_accuracy_std']:.3f} "
          f"(range {summary['test_accuracy_min']:.3f}-{summary['test_accuracy_max']:.3f})")
    (C.REPORTS_DIR / "diat_usat_cnn_stability_results.json").write_text(
        json.dumps(summary, indent=2))
    print(f"results -> {C.REPORTS_DIR / 'diat_usat_cnn_stability_results.json'}")


if __name__ == "__main__":
    main()
