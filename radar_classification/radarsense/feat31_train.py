"""Trains a classifier on the standalone 31-feature schema and evaluates it
-- overall and specifically at the -25 dB slice, since that's the whole
point of extending the SNR range beyond the frozen schema's -20 dB floor.

Same model family, imbalance handling, AND drone-threshold tuning as the
incumbent 11-feature model (CatBoost + class-weighting + PR-curve/F-beta2
threshold tuning on validation, reusing radarsense.optimize's generic
tune_threshold/threshold_predict -- they operate on proba arrays only, not
the feature schema, so they're directly reusable) so any accuracy
difference reflects the FEATURE SCHEMA, not a missing optimization step.
The first version of this script used raw argmax with no threshold tuning
at all, which is a likely reason it under-performed on drone recall
specifically -- fixed here, not hidden.
"""

import json

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.utils.class_weight import compute_sample_weight

from radarsense import config as C
from radarsense.feat31_features import FEAT31_COLUMNS
from radarsense.metrics import confusion, full_metrics
from radarsense.optimize import threshold_predict, tune_threshold


def main():
    train = pd.read_parquet(C.DATA_DIR / "feat31_train_features.parquet")
    val = pd.read_parquet(C.DATA_DIR / "feat31_val_features.parquet")
    test = pd.read_parquet(C.DATA_DIR / "feat31_test_features.parquet")

    sw = compute_sample_weight("balanced", train[C.LABEL_COLUMN])
    clf = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.1,
                             random_seed=C.SEED, verbose=0, allow_writing_files=False)
    clf.fit(train[FEAT31_COLUMNS], train[C.LABEL_COLUMN], sample_weight=sw,
            eval_set=(val[FEAT31_COLUMNS], val[C.LABEL_COLUMN]))

    def reorder_proba(df):
        proba = clf.predict_proba(df[FEAT31_COLUMNS])
        return proba[:, [list(clf.classes_).index(c) for c in C.CLASSES]]

    proba_val = reorder_proba(val)
    t_fbeta, t_cost, _ = tune_threshold(proba_val, val[C.LABEL_COLUMN])
    print(f"tuned drone threshold: fbeta2-optimal={t_fbeta:.3f} cost-optimal={t_cost:.3f}")

    def evaluate(df, label, t):
        proba = reorder_proba(df)
        pred = threshold_predict(proba, t)
        m = full_metrics(df[C.LABEL_COLUMN], pred, proba)
        m["confusion_matrix"] = confusion(df[C.LABEL_COLUMN], pred)
        m["n"] = len(df)
        m["threshold_used"] = t
        print(f"{label}: n={len(df)} accuracy={m['accuracy']:.4f} "
              f"macro_f1={m['macro_f1']:.4f} drone_recall={m['recall_drone']:.4f}")
        return m

    print("\n--- with tuned threshold (fbeta2-optimal) ---")
    overall = evaluate(test, "overall test", t_fbeta)

    snr_slices = {}
    for lo, hi in [(-25, -24), (-24, -22), (-22, -20), (-20, -18), (-18, -16),
                   (-16, -14), (-14, -12), (-12, -10)]:
        sub = test[(test.snr_db >= lo) & (test.snr_db < hi)]
        if len(sub) < 20:
            continue
        snr_slices[f"{lo}_to_{hi}"] = evaluate(sub, f"SNR [{lo},{hi})", t_fbeta)

    deep_low = test[test.snr_db <= -23]
    m_25db = evaluate(deep_low, "SNR <= -23dB (deep low-SNR slice)", t_fbeta)

    print("\n--- for comparison: raw argmax (no threshold tuning, the original approach) ---")
    proba_deep = reorder_proba(deep_low)
    pred_argmax = np.array(C.CLASSES)[proba_deep.argmax(axis=1)]
    m_argmax = full_metrics(deep_low[C.LABEL_COLUMN], pred_argmax, proba_deep)
    print(f"argmax (no tuning) at SNR<=-23dB: accuracy={m_argmax['accuracy']:.4f} "
          f"drone_recall={m_argmax['recall_drone']:.4f}")

    joblib.dump(clf, C.MODELS_DIR / "feat31_model.joblib")
    out = {
        "feature_columns": FEAT31_COLUMNS,
        "snr_range_db": [-25.0, -10.0],
        "threshold_fbeta2_optimal": t_fbeta,
        "threshold_cost_optimal": t_cost,
        "overall_test": overall,
        "snr_stratified": snr_slices,
        "deep_low_snr_leq_minus23db": m_25db,
        "deep_low_snr_leq_minus23db_argmax_no_tuning": m_argmax,
    }
    (C.REPORTS_DIR / "feat31_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nmodel -> {C.MODELS_DIR / 'feat31_model.joblib'}")
    print(f"results -> {C.REPORTS_DIR / 'feat31_results.json'}")


if __name__ == "__main__":
    main()
