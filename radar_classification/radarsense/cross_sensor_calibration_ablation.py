""""Unified calibration protocol" -- what's actually achievable and honest.

Checked first, before building anything: all 4 real-sensor bridges
(real_data_bridge.py/SAAB, raddar_experiments.py/RAD-DAR,
tsms_experiments.py/TSMS, diat_usat_bridge.py/DIAT-uSAT) already use the
IDENTICAL snr_db formula -- scale = 5.0/(quantile(0.95)-median), snr_db =
-15.0 + (raw-median)*scale, clip[-20,-10]. The inconsistency isn't the
formula, it's what it's anchored to: each sensor's own empirical median
self-defines as -15dB, not a shared physical reference. A genuinely
absolute cross-sensor SNR calibration would need each sensor's real
system parameters (transmit power, antenna gain, noise figure) -- not
available for these third-party public datasets, so faking one would be
unfounded, not a fix.

RCS calibration status, checked precisely: SAAB and TSMS both have REAL
radiometric calibration against a known corner-reflector reference
(+20 dBsm). RAD-DAR and DIAT-uSAT use an uncalibrated z-scored power
proxy (their own docstrings already say so -- "no known-RCS reference
target in this dataset"). So rcs_estimate is consistently anchored for
half the sensors and not the other half.

Instead of fabricating a fake unified calibration, this tests the direct,
falsifiable question: does dropping the features known to be
inconsistently anchored (snr_db always self-referential; rcs_estimate
inconsistently calibrated across sensors) actually improve cross-sensor
transfer? Reuses cross_sensor_transfer.py's exact methodology (imported
unmodified) with 3 feature-set variants.
"""

import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from radarsense import config as C
from radarsense.cross_sensor_transfer import SENSORS, load_binary
from radarsense.raddar_cnn import binary_metrics

FEATURE_VARIANTS = {
    "all_11_features": C.FEATURE_COLUMNS,
    "drop_snr_db": [f for f in C.FEATURE_COLUMNS if f != "snr_db"],
    "drop_snr_and_rcs": [f for f in C.FEATURE_COLUMNS if f not in ("snr_db", "rcs_estimate")],
}


def train_one(train_df, features):
    sw = compute_sample_weight("balanced", train_df.y)
    clf = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1,
                             random_seed=C.SEED, verbose=0, allow_writing_files=False)
    clf.fit(train_df[features], train_df.y, sample_weight=sw)
    return clf


def evaluate(clf, df, features):
    proba = clf.predict_proba(df[features])[:, 1]
    pred = (proba >= 0.5).astype(int)
    m = binary_metrics(df.y.values, pred, proba)
    m["n"] = len(df)
    return m


def main():
    data = {name: load_binary(path) for name, path in SENSORS.items()}

    summary_rows = []
    for variant_name, features in FEATURE_VARIANTS.items():
        print(f"\n=== variant: {variant_name} ({len(features)} features) ===")
        models, in_distribution = {}, {}
        for name, df in data.items():
            tr, te = train_test_split(df, test_size=0.2, stratify=df.y, random_state=C.SEED)
            clf = train_one(tr, features)
            models[name] = clf
            in_distribution[name] = evaluate(clf, te, features)

        cross_accs, cross_pr_aucs = [], []
        for src, clf in models.items():
            for tgt, df_tgt in data.items():
                if tgt == src:
                    continue
                m = evaluate(clf, df_tgt, features)
                cross_accs.append(m["accuracy"])
                cross_pr_aucs.append(m["pr_auc_drone"])

        mean_indist = float(np.mean([m["accuracy"] for m in in_distribution.values()]))
        mean_cross = float(np.mean(cross_accs))
        mean_pr_auc = float(np.mean(cross_pr_aucs))
        print(f"  mean in-distribution accuracy: {mean_indist:.4f}")
        print(f"  mean cross-sensor accuracy (all 12 pairs): {mean_cross:.4f}")
        print(f"  mean cross-sensor PR-AUC: {mean_pr_auc:.4f}")
        summary_rows.append({"variant": variant_name, "n_features": len(features),
                             "mean_in_distribution_accuracy": mean_indist,
                             "mean_cross_sensor_accuracy": mean_cross,
                             "mean_cross_sensor_pr_auc": mean_pr_auc})

    print("\n--- does dropping inconsistently-calibrated features help? ---")
    for row in summary_rows:
        print(f"  {row['variant']:20s} ({row['n_features']} feat): "
              f"in-dist={row['mean_in_distribution_accuracy']:.4f}  "
              f"cross-sensor={row['mean_cross_sensor_accuracy']:.4f}  "
              f"pr_auc={row['mean_cross_sensor_pr_auc']:.4f}")

    (C.REPORTS_DIR / "cross_sensor_calibration_ablation_results.json").write_text(
        json.dumps({"variants": summary_rows}, indent=2))
    print(f"\nresults -> {C.REPORTS_DIR / 'cross_sensor_calibration_ablation_results.json'}")


if __name__ == "__main__":
    main()
