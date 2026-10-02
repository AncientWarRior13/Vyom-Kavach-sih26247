"""Kinematic fallback classifier -- the OTHER mitigation direction
THREAT_MODEL.md named for micro-Doppler masking (distinct mechanism from
feature-dropout training: detection + routing, not retraining the primary
model at all): "combining kinematic features ... with micro-Doppler when
it's available, and falling back to kinematics-weighted decisions when
micro-Doppler bandwidth reads anomalously low (a signature of masking,
not just a quiet rotor)."

Two pieces:
  1. A SECOND classifier trained on the 9 non-micro-Doppler features only
     (never sees micro_doppler_bandwidth/jitter at all, so it structurally
     cannot be fooled by masking them).
  2. A masking DETECTOR: flags a sample as "suspicious" if
     micro_doppler_bandwidth falls below a low percentile of its
     TRAINING distribution -- masking pushes it down; a real quiet-rotor
     drone's bandwidth is still drawn from the same distribution the
     detector is calibrated on, so both get compared fairly.

Routing: blend the primary and fallback model's probabilities, weighted
toward the fallback as the suspicion score increases. Tested for both
(a) whether it recovers recall under the SAME masking stress test, and
(b) whether it costs anything on UNPERTURBED data, including on real
naturally-quiet-rotor drones specifically -- a fallback that only works
under attack but breaks quiet-but-honest drones would be a real problem,
not a fix.
"""

import json

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.utils.class_weight import compute_sample_weight

from radarsense import config as C
from radarsense.optimize import threshold_predict, tune_threshold

NON_MD_FEATURES = [f for f in C.FEATURE_COLUMNS
                   if f not in ("micro_doppler_bandwidth", "micro_doppler_jitter")]
SUSPICION_PERCENTILE = 20


def reorder_proba(m, X, classes):
    p = m.predict_proba(X)
    return p[:, [list(m.classes_).index(c) for c in classes]]


def main():
    primary = joblib.load(C.MODELS_DIR / "final_model.joblib")
    meta = json.loads((C.MODELS_DIR / "final_model_meta.json").read_text())
    threshold = meta["drone_threshold_deployed"]

    train = pd.read_parquet(C.DATA_DIR / "radar_train.parquet")
    val = pd.read_parquet(C.DATA_DIR / "radar_val.parquet")
    test = pd.read_parquet(C.DATA_DIR / "radar_test.parquet")

    print("--- training the kinematic-only fallback classifier (never sees micro-Doppler) ---")
    sw = compute_sample_weight("balanced", train[C.LABEL_COLUMN])
    fallback = CatBoostClassifier(iterations=400, depth=6, learning_rate=0.1,
                                  random_seed=C.SEED, verbose=0, allow_writing_files=False)
    fallback.fit(train[NON_MD_FEATURES], train[C.LABEL_COLUMN], sample_weight=sw)

    fallback_val_proba = reorder_proba(fallback, val[NON_MD_FEATURES], C.CLASSES)
    t_fallback, _, _ = tune_threshold(fallback_val_proba, val[C.LABEL_COLUMN])
    print(f"fallback-alone tuned threshold: {t_fallback:.3f}")

    train_drone_md_bw = train.loc[train[C.LABEL_COLUMN] == "drone", "micro_doppler_bandwidth"]
    md_bw_suspicion_cutoff = float(np.percentile(train_drone_md_bw, SUSPICION_PERCENTILE))
    print(f"suspicion cutoff (training {SUSPICION_PERCENTILE}th percentile of DRONE micro_doppler_bandwidth): "
          f"{md_bw_suspicion_cutoff:.2f}")

    def blended_predict(df, blend_power=2.0):
        """Suspicion score in [0,1]: 0 = md_bw at/above cutoff (fully trust
        primary), 1 = md_bw at 0 (fully trust fallback). blend_power>1
        makes the transition sharper near the cutoff."""
        p_primary = reorder_proba(primary, df[C.FEATURE_COLUMNS], C.CLASSES)
        p_fallback = reorder_proba(fallback, df[NON_MD_FEATURES], C.CLASSES)
        suspicion = np.clip(1 - df["micro_doppler_bandwidth"].values / md_bw_suspicion_cutoff, 0, 1) ** blend_power
        blended = (1 - suspicion[:, None]) * p_primary + suspicion[:, None] * p_fallback
        return blended, suspicion

    def drone_recall_blended(df):
        proba, _ = blended_predict(df)
        pred = threshold_predict(proba, threshold)
        return float((pred == "drone").mean())

    def drone_recall_primary_only(df):
        proba = reorder_proba(primary, df[C.FEATURE_COLUMNS], C.CLASSES)
        pred = threshold_predict(proba, threshold)
        return float((pred == "drone").mean())

    print("\n--- sanity check: unperturbed test performance, primary-only vs. blended ---")
    proba_all_primary = reorder_proba(primary, test[C.FEATURE_COLUMNS], C.CLASSES)
    pred_all_primary = threshold_predict(proba_all_primary, threshold)
    acc_primary = float((pred_all_primary == test[C.LABEL_COLUMN]).mean())

    proba_all_blend, susp_all = blended_predict(test)
    pred_all_blend = threshold_predict(proba_all_blend, threshold)
    acc_blend = float((pred_all_blend == test[C.LABEL_COLUMN]).mean())
    print(f"  primary-only overall accuracy: {acc_primary:.4f}")
    print(f"  blended overall accuracy:      {acc_blend:.4f}")
    print(f"  fraction of ALL test samples that trigger any fallback blending (suspicion>0): "
          f"{(susp_all > 0).mean():.1%}")

    print("\n--- checking specifically on real, naturally-quiet-rotor drones (bottom 20% md_bw, unperturbed) ---")
    drones_test = test[test[C.LABEL_COLUMN] == "drone"]
    quiet_drones = drones_test[drones_test["micro_doppler_bandwidth"] <= drones_test["micro_doppler_bandwidth"].quantile(0.2)]
    r_primary_quiet = drone_recall_primary_only(quiet_drones)
    r_blend_quiet = drone_recall_blended(quiet_drones)
    print(f"  primary-only recall on naturally-quiet drones: {r_primary_quiet:.4f}")
    print(f"  blended recall on naturally-quiet drones:      {r_blend_quiet:.4f}")

    print("\n--- re-running the SAME masking stress test ---")
    drones = test[test[C.LABEL_COLUMN] == "drone"].reset_index(drop=True)
    results_primary, results_blend = {}, {}
    results_primary["unperturbed"] = drone_recall_primary_only(drones)
    results_blend["unperturbed"] = drone_recall_blended(drones)
    for att in (0.7, 0.5, 0.3):
        z = drones.copy()
        z["micro_doppler_bandwidth"] *= att
        z["micro_doppler_jitter"] *= att
        results_primary[f"x{att}"] = drone_recall_primary_only(z)
        results_blend[f"x{att}"] = drone_recall_blended(z)
        print(f"  x{att}: primary-only={results_primary[f'x{att}']:.4f}  "
              f"blended-with-fallback={results_blend[f'x{att}']:.4f}")

    out = {
        "non_md_features": NON_MD_FEATURES,
        "suspicion_percentile_cutoff": SUSPICION_PERCENTILE,
        "md_bw_suspicion_cutoff_value": md_bw_suspicion_cutoff,
        "overall_unperturbed_accuracy": {"primary_only": acc_primary, "blended": acc_blend},
        "fraction_test_samples_triggering_any_blend": float((susp_all > 0).mean()),
        "naturally_quiet_drones_recall": {"primary_only": r_primary_quiet, "blended": r_blend_quiet},
        "masking_stress_test": {"primary_only": results_primary, "blended_with_fallback": results_blend},
    }
    joblib.dump(fallback, C.MODELS_DIR / "kinematic_fallback_model.joblib")
    (C.REPORTS_DIR / "kinematic_fallback_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nmodel -> {C.MODELS_DIR / 'kinematic_fallback_model.joblib'}")
    print(f"results -> {C.REPORTS_DIR / 'kinematic_fallback_results.json'}")


if __name__ == "__main__":
    main()
