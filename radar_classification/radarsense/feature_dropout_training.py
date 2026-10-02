"""Feature-dropout retraining: does teaching the 11-feature model that
low micro_doppler_bandwidth/jitter doesn't always mean "not drone" reduce
its over-reliance on those two features, the mechanism behind the
micro-Doppler-masking vulnerability in THREAT_MODEL.md?

CatBoost has no built-in dropout the way a neural net does; the standard
practical equivalent for tree ensembles is data augmentation: add
duplicate copies of drone training rows with micro_doppler_bandwidth/
jitter attenuated across the SAME range the masking attack itself uses
(0.3-0.9x), correctly labeled "drone" -- these are still real drones,
just with a masked/degraded micro-Doppler signature, forcing the model to
find signal in the other 9 features for that specific value regime instead
of relying on micro-Doppler as a near-sufficient statistic.

Re-tested against the EXACT SAME masking stress test as robustness.py
(x0.7/x0.5/x0.3 attenuation, same 200-sample-scale evaluation), so the
comparison is directly apples-to-apples with the reported 91.8%->12.3%
collapse.
"""

import json

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.utils.class_weight import compute_sample_weight

from radarsense import config as C
from radarsense.metrics import full_metrics
from radarsense.optimize import threshold_predict, tune_threshold

DROPOUT_FRACTION = 0.5
ATTENUATION_RANGE = (0.3, 0.9)


def main():
    model = joblib.load(C.MODELS_DIR / "final_model.joblib")
    meta = json.loads((C.MODELS_DIR / "final_model_meta.json").read_text())
    orig_threshold = meta["drone_threshold_deployed"]

    train = pd.read_parquet(C.DATA_DIR / "radar_train.parquet")
    val = pd.read_parquet(C.DATA_DIR / "radar_val.parquet")
    test = pd.read_parquet(C.DATA_DIR / "radar_test.parquet")

    print("--- augmenting training data with masked-micro-Doppler drone duplicates ---")
    rng = np.random.default_rng(C.SEED)
    train_drones = train[train[C.LABEL_COLUMN] == "drone"]
    dup = train_drones.sample(frac=DROPOUT_FRACTION, random_state=C.SEED).copy()
    att = rng.uniform(*ATTENUATION_RANGE, len(dup))
    dup["micro_doppler_bandwidth"] = dup["micro_doppler_bandwidth"] * att
    dup["micro_doppler_jitter"] = np.clip(dup["micro_doppler_jitter"] * att, 0, 1)
    train_augmented = pd.concat([train, dup], ignore_index=True)
    print(f"added {len(dup)} masked-duplicate drone rows ({DROPOUT_FRACTION:.0%} of "
          f"{len(train_drones)} drone training rows), attenuation range {ATTENUATION_RANGE}")
    print(f"train set: {len(train)} -> {len(train_augmented)} rows")

    print("\n--- retraining (same CatBoost config as the deployed model) ---")
    sw = compute_sample_weight("balanced", train_augmented[C.LABEL_COLUMN])
    clf = CatBoostClassifier(iterations=400, depth=6, learning_rate=0.1,
                             random_seed=C.SEED, verbose=0, allow_writing_files=False)
    clf.fit(train_augmented[C.FEATURE_COLUMNS], train_augmented[C.LABEL_COLUMN], sample_weight=sw)

    def reorder_proba(m, X):
        p = m.predict_proba(X)
        return p[:, [list(m.classes_).index(c) for c in C.CLASSES]]

    proba_val = reorder_proba(clf, val[C.FEATURE_COLUMNS])
    t_fbeta, t_cost, _ = tune_threshold(proba_val, val[C.LABEL_COLUMN])
    print(f"tuned drone threshold on retrained model: fbeta2={t_fbeta:.3f}")

    print("\n--- sanity check: normal (unperturbed) test performance ---")
    proba_orig = reorder_proba(model, test[C.FEATURE_COLUMNS])
    pred_orig = threshold_predict(proba_orig, orig_threshold)
    m_orig = full_metrics(test[C.LABEL_COLUMN], pred_orig, proba_orig)

    proba_new = reorder_proba(clf, test[C.FEATURE_COLUMNS])
    pred_new = threshold_predict(proba_new, t_fbeta)
    m_new = full_metrics(test[C.LABEL_COLUMN], pred_new, proba_new)
    print(f"  original  : accuracy={m_orig['accuracy']:.4f} drone_recall={m_orig['recall_drone']:.4f}")
    print(f"  retrained : accuracy={m_new['accuracy']:.4f} drone_recall={m_new['recall_drone']:.4f}")

    print("\n--- re-running the SAME masking stress test (robustness.py's exact levels) ---")
    drones = test[test[C.LABEL_COLUMN] == "drone"].reset_index(drop=True)

    def drone_recall(m, t, X):
        proba = reorder_proba(m, X[C.FEATURE_COLUMNS])
        pred = threshold_predict(proba, t)
        return float((pred == "drone").mean())

    results = {"unperturbed": drone_recall(clf, t_fbeta, drones)}
    for att_level in (0.7, 0.5, 0.3):
        z = drones.copy()
        z["micro_doppler_bandwidth"] *= att_level
        z["micro_doppler_jitter"] *= att_level
        results[f"x{att_level}"] = drone_recall(clf, t_fbeta, z)
        print(f"  x{att_level}: drone_recall={results[f'x{att_level}']:.4f}")

    print("\n--- head-to-head vs. the original (undefended) model ---")
    reported_original = {"unperturbed": 0.918, "x0.7": 0.764, "x0.5": 0.518, "x0.3": 0.123}
    for level, key in [("unperturbed", "unperturbed"), ("x0.7", "x0.7"), ("x0.5", "x0.5"), ("x0.3", "x0.3")]:
        print(f"  {level:14s} original={reported_original[key]:.3f}  "
              f"feature-dropout-trained={results[key]:.3f}")

    joblib.dump(clf, C.MODELS_DIR / "final_model_feature_dropout_trained.joblib")
    out = {
        "dropout_fraction": DROPOUT_FRACTION,
        "attenuation_range": list(ATTENUATION_RANGE),
        "n_augmented_rows": len(dup),
        "retrained_threshold_fbeta2": t_fbeta,
        "sanity_check_normal_test_performance": {
            "original_model": {"accuracy": m_orig["accuracy"], "recall_drone": m_orig["recall_drone"]},
            "retrained_model": {"accuracy": m_new["accuracy"], "recall_drone": m_new["recall_drone"]},
        },
        "masking_stress_test": results,
        "comparison_vs_original": reported_original,
    }
    (C.REPORTS_DIR / "feature_dropout_training_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nmodel -> {C.MODELS_DIR / 'final_model_feature_dropout_trained.joblib'}")
    print(f"results -> {C.REPORTS_DIR / 'feature_dropout_training_results.json'}")


if __name__ == "__main__":
    main()
