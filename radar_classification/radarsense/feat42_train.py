"""Trains the combined 42-feature (11 physics + 31 waveform-statistic, 41
unique columns -- see feat42_simulator.py's docstring) schema and compares
it against BOTH individual models at -25dB, on THIS model's own test set
sliced the same way.

Same model family, imbalance handling, and threshold-tuning discipline as
both individual models (CatBoost + class-weighting + PR-curve/F-beta2
threshold tuning via radarsense.optimize, reused unmodified) -- so any
difference reflects the feature schema, not an inconsistent optimization
step.

Rationale for attempting this at all: the physics features (11) are the
established stronger performer for drone recall specifically (this
project's own frozen model), while the waveform-statistic features (31)
showed a genuine accuracy edge under extreme noise (feat31_compare.py,
after the noise-calibration bug fix). A combined model is a legitimate
"best of both worlds" attempt -- not guaranteed to beat both individually
(more features is not automatically better; CatBoost has to find useful
splits across 41 columns instead of 11 or 31, and the physics/waveform
features are partially redundant by construction, both derived from the
same v0/rcs/md_bw/md_jit per instance), so this is reported honestly
either way.
"""

import json

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.utils.class_weight import compute_sample_weight

from radarsense import config as C
from radarsense.feat42_simulator import FEAT42_COLUMNS
from radarsense.metrics import confusion, full_metrics
from radarsense.optimize import threshold_predict, tune_threshold


def main():
    train = pd.read_parquet(C.DATA_DIR / "feat42_train.parquet")
    val = pd.read_parquet(C.DATA_DIR / "feat42_val.parquet")
    test = pd.read_parquet(C.DATA_DIR / "feat42_test.parquet")

    sw = compute_sample_weight("balanced", train[C.LABEL_COLUMN])
    clf = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.1,
                             random_seed=C.SEED, verbose=0, allow_writing_files=False)
    clf.fit(train[FEAT42_COLUMNS], train[C.LABEL_COLUMN], sample_weight=sw,
            eval_set=(val[FEAT42_COLUMNS], val[C.LABEL_COLUMN]))

    def reorder_proba(df):
        proba = clf.predict_proba(df[FEAT42_COLUMNS])
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

    joblib.dump(clf, C.MODELS_DIR / "feat42_model.joblib")

    feat31_results = json.loads((C.REPORTS_DIR / "feat31_results.json").read_text())
    m31_25db = feat31_results["deep_low_snr_leq_minus23db"]

    print("\n--- three-way comparison at SNR<=-23dB (each model on ITS OWN "
          "matching test set, same slice definition) ---")
    print(f"  11-feature (frozen, out-of-distribution here): "
          f"see feat31_vs_11feature_comparison.json for this leg")
    print(f"  31-feature: accuracy={m31_25db['accuracy']:.4f} "
          f"drone_recall={m31_25db['recall_drone']:.4f} (n={m31_25db['n']})")
    print(f"  42-feature (combined): accuracy={m_25db['accuracy']:.4f} "
          f"drone_recall={m_25db['recall_drone']:.4f} (n={m_25db['n']})")

    try:
        feat31_vs_11 = json.loads((C.REPORTS_DIR / "feat31_vs_11feature_comparison.json").read_text())
        m11_25db = feat31_vs_11["model_11feature_frozen_ood"]
    except FileNotFoundError:
        m11_25db = None

    winner = max(
        [("11-feature", m11_25db["accuracy"] if m11_25db else -1),
         ("31-feature", m31_25db["accuracy"]),
         ("42-feature-combined", m_25db["accuracy"])],
        key=lambda x: x[1],
    )[0]
    winner_recall = max(
        [("11-feature", m11_25db["recall_drone"] if m11_25db else -1),
         ("31-feature", m31_25db["recall_drone"]),
         ("42-feature-combined", m_25db["recall_drone"])],
        key=lambda x: x[1],
    )[0]

    out = {
        "feature_columns": FEAT42_COLUMNS,
        "n_unique_features": len(FEAT42_COLUMNS),
        "snr_range_db": [-25.0, -10.0],
        "threshold_fbeta2_optimal": t_fbeta,
        "threshold_cost_optimal": t_cost,
        "overall_test": overall,
        "snr_stratified": snr_slices,
        "deep_low_snr_leq_minus23db": m_25db,
        "three_way_comparison_at_minus23db": {
            "11_feature_frozen_ood": m11_25db,
            "31_feature": {"accuracy": m31_25db["accuracy"], "recall_drone": m31_25db["recall_drone"],
                           "n": m31_25db["n"]},
            "42_feature_combined": {"accuracy": m_25db["accuracy"], "recall_drone": m_25db["recall_drone"],
                                    "n": m_25db["n"]},
            "higher_accuracy": winner,
            "higher_drone_recall": winner_recall,
        },
    }
    (C.REPORTS_DIR / "feat42_results.json").write_text(json.dumps(out, indent=2))

    (C.MODELS_DIR / "feat42_model_meta.json").write_text(json.dumps({
        "model_family": "catboost",
        "variant": "class_weight + fbeta2-tuned threshold",
        "drone_threshold_cost_optimal": t_cost,
        "drone_threshold_fbeta2": t_fbeta,
        "drone_threshold_deployed": t_fbeta,
        "classes": C.CLASSES,
        "feature_order": FEAT42_COLUMNS,
        "snr_range_db": [-25.0, -10.0],
    }, indent=2))

    print(f"\nhigher accuracy at SNR<=-23dB: {winner}")
    print(f"higher drone recall at SNR<=-23dB: {winner_recall}")
    print(f"model -> {C.MODELS_DIR / 'feat42_model.joblib'}")
    print(f"meta -> {C.MODELS_DIR / 'feat42_model_meta.json'}")
    print(f"results -> {C.REPORTS_DIR / 'feat42_results.json'}")


if __name__ == "__main__":
    main()
