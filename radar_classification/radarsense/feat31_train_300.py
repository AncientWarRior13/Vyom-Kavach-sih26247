"""Trains on the 300-step ("longer waveform") variant of the 31-feature
schema and compares directly against the 100-step model -- same CatBoost +
class-weighting + threshold-tuning discipline as feat31_train.py (so any
difference reflects waveform length, not a methodology change), evaluated
on BOTH the overall test set and the SNR<=-23dB deep-low-SNR slice that is
this schema's whole reason for existing.

Also directly checks the mechanism this experiment is testing, not just
the end-to-end accuracy number: the specific features SHAP flagged as
highest-importance and (by construction) most degraded by short, noisy
waveforms -- amplitude_spectral_bandwidth, amplitude_dominant_freq,
energy_spectral_bandwidth -- should show LOWER noise-induced variance at
300 steps than at 100, if the "more samples -> better-conditioned
spectral statistics" reasoning is actually correct and not just a story
that happens to also come with more accuracy for unrelated reasons.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.utils.class_weight import compute_sample_weight


def make_clf(seed, use_gpu: bool = False) -> CatBoostClassifier:
    """Roadmap training-time-cutoff step 1: this script used to train a
    fixed 500 iterations despite passing an eval_set, so early stopping
    never had a chance to end the run sooner. A higher learning rate with
    fewer split candidates (border_count) and real early stopping cuts
    training time several-fold on this schema's dataset size; re-verify
    accuracy against the previous fixed-iteration model after switching.
    """
    return CatBoostClassifier(
        iterations=2000,
        learning_rate=0.15,
        depth=6,
        border_count=64,
        thread_count=os.cpu_count(),
        task_type="GPU" if use_gpu else "CPU",
        early_stopping_rounds=40,
        use_best_model=True,
        random_seed=seed,
        verbose=0,
        allow_writing_files=False,
    )

from radarsense import config as C
from radarsense.feat31_features_300 import FEAT31_COLUMNS
from radarsense.metrics import confusion, full_metrics
from radarsense.optimize import threshold_predict, tune_threshold

SPECTRAL_FEATURES = ["amplitude_spectral_bandwidth", "amplitude_dominant_freq",
                     "energy_spectral_bandwidth"]


def main():
    train = pd.read_parquet(C.DATA_DIR / "feat31_300_train_features.parquet")
    val = pd.read_parquet(C.DATA_DIR / "feat31_300_val_features.parquet")
    test = pd.read_parquet(C.DATA_DIR / "feat31_300_test_features.parquet")

    sw = compute_sample_weight("balanced", train[C.LABEL_COLUMN])
    clf = make_clf(C.SEED)
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

    print("\n--- 300-step model, with tuned threshold ---")
    overall = evaluate(test, "overall test", t_fbeta)
    deep_low = test[test.snr_db <= -23]
    m_25db = evaluate(deep_low, "SNR <= -23dB", t_fbeta)

    print("\n--- mechanism check: spectral-feature noise-induced std, "
          "100-step vs 300-step (same class, same SNR slice) ---")
    train_100 = pd.read_parquet(C.DATA_DIR / "feat31_train_features.parquet")
    mech = {}
    for feat in SPECTRAL_FEATURES:
        row = {}
        for label, df_ in [("100_step", train_100), ("300_step", train)]:
            deep = df_[(df_.snr_db <= -23) & (df_[C.LABEL_COLUMN] == "drone")]
            good = df_[(df_.snr_db >= -12) & (df_[C.LABEL_COLUMN] == "drone")]
            row[f"{label}_std_at_deep_low_snr"] = float(deep[feat].std())
            row[f"{label}_std_at_good_snr"] = float(good[feat].std())
            row[f"{label}_noise_inflation_ratio"] = float(deep[feat].std() / max(good[feat].std(), 1e-9))
        mech[feat] = row
        print(f"  {feat}:")
        print(f"    100-step: good-SNR std={row['100_step_std_at_good_snr']:.3f}  "
              f"deep-low-SNR std={row['100_step_std_at_deep_low_snr']:.3f}  "
              f"inflation={row['100_step_noise_inflation_ratio']:.2f}x")
        print(f"    300-step: good-SNR std={row['300_step_std_at_good_snr']:.3f}  "
              f"deep-low-SNR std={row['300_step_std_at_deep_low_snr']:.3f}  "
              f"inflation={row['300_step_noise_inflation_ratio']:.2f}x")

    feat31_results = json.loads((C.REPORTS_DIR / "feat31_results.json").read_text())
    m_100_25db = feat31_results["deep_low_snr_leq_minus23db"]

    print(f"\n--- head-to-head at SNR<=-23dB ---")
    print(f"  100-step: accuracy={m_100_25db['accuracy']:.4f} drone_recall={m_100_25db['recall_drone']:.4f}")
    print(f"  300-step: accuracy={m_25db['accuracy']:.4f} drone_recall={m_25db['recall_drone']:.4f}")

    joblib.dump(clf, C.MODELS_DIR / "feat31_300_model.joblib")
    out = {
        "feature_columns": FEAT31_COLUMNS,
        "waveform_length_steps": 300,
        "window_seconds": 3.0,
        "threshold_fbeta2_optimal": t_fbeta,
        "threshold_cost_optimal": t_cost,
        "overall_test": overall,
        "deep_low_snr_leq_minus23db": m_25db,
        "mechanism_check_spectral_feature_noise_inflation": mech,
        "comparison_vs_100step_at_minus23db": {
            "100_step": {"accuracy": m_100_25db["accuracy"], "recall_drone": m_100_25db["recall_drone"],
                        "n": m_100_25db["n"]},
            "300_step": {"accuracy": m_25db["accuracy"], "recall_drone": m_25db["recall_drone"],
                        "n": m_25db["n"]},
        },
    }
    (C.REPORTS_DIR / "feat31_300_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nmodel -> {C.MODELS_DIR / 'feat31_300_model.joblib'}")
    print(f"results -> {C.REPORTS_DIR / 'feat31_300_results.json'}")


if __name__ == "__main__":
    main()
