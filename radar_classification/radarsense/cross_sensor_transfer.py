"""Cross-real-sensor ZERO-SHOT transfer: train a drone-vs-anomaly classifier
on ONE real sensor's actual measured data, then evaluate it -- with ZERO
retraining, ZERO fine-tuning -- on a DIFFERENT real sensor's actual data.

Every real-sensor result elsewhere in this project is either WITHIN-sensor
(diat_usat_bridge.py's own variant-holdout retrain, raddar_experiments.py's
own retrain-and-test -- trained and tested on the SAME sensor) or
SIM-to-real zero-shot (the frozen SIMULATED-trained model tested on real
data, e.g. diat_usat_bridge.py's Experiment A). This is neither of those --
it's the one genuinely untested question in the whole project: does a model
that has only ever seen ONE real sensor's actual noise/calibration/physics
characteristics generalize to a DIFFERENT real sensor's, with nothing in
between.

Common ground: drone-vs-anomaly binary -- the only task all four sensors'
existing bridges share.
    SAAB 77GHz FMCW      (real_data_bridge.py)   drone/bird/anomaly
    RAD-DAR 8.75GHz FMCW (raddar_experiments.py) drone/anomaly
    TSMS-Drone 2.5GHz CW (tsms_experiments.py)   drone/anomaly
    DIAT-uSAT X-band CW  (diat_usat_bridge.py)   drone/bird/aircraft/anomaly
RAD-DAR and TSMS have neither bird nor aircraft, so a 4-class comparison
isn't possible across all pairs -- drone-vs-anomaly is the largest common
task, already the project's own headline minority-class metric everywhere
else.

For each sensor: an 80/20 stratified split. The SAME model, trained ONCE
on that sensor's 80% train split, is evaluated on (a) its own 20% held-out
test (in-distribution reference number) and (b) every OTHER sensor's FULL
real data (cross-sensor zero-shot number). Using the identical trained
model for both is what makes the in-distribution-vs-cross-sensor GAP a
fair, apples-to-apples comparison -- that gap, not the cross-sensor number
alone, is the actual finding.

Caveat carried over from every individual bridge (disclosed there, repeated
here rather than re-litigated): each sensor's snr_db is an independently
affine-mapped, "indicative only" calibration into the schema's [-20,-10]dB
band -- not a physically identical quantity across sensors. Cross-sensor
transfer is tested on the schema's OTHER 10 features too, not snr_db alone,
so this doesn't invalidate the test, but it's a real limitation on how
literally "generalizes across sensors" should be read.
"""

import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from radarsense import config as C
from radarsense.raddar_cnn import binary_metrics

SENSORS = {
    "SAAB_77GHz_FMCW": "real_saab77.parquet",
    "RADDAR_8.75GHz_FMCW": "real_raddar.parquet",
    "TSMS_2.5GHz_CW": "real_tsms_cw.parquet",
    "DIAT_uSAT_Xband_CW": "real_diat_usat.parquet",
}


def load_binary(path):
    df = pd.read_parquet(C.DATA_DIR / path)
    df = df[df[C.LABEL_COLUMN].isin(["drone", "anomaly"])].copy()
    df["y"] = (df[C.LABEL_COLUMN] == "drone").astype(int)
    return df.reset_index(drop=True)


def train_one(train_df):
    sw = compute_sample_weight("balanced", train_df.y)
    clf = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1,
                             random_seed=C.SEED, verbose=0, allow_writing_files=False)
    clf.fit(train_df[C.FEATURE_COLUMNS], train_df.y, sample_weight=sw)
    return clf


def evaluate(clf, df):
    proba = clf.predict_proba(df[C.FEATURE_COLUMNS])[:, 1]
    pred = (proba >= 0.5).astype(int)
    m = binary_metrics(df.y.values, pred, proba)
    m["n"] = len(df)

    thresholds = np.linspace(0.02, 0.98, 97)
    accs = [(t, ((proba >= t).astype(int) == df.y.values).mean()) for t in thresholds]
    best_t, best_acc = max(accs, key=lambda x: x[1])
    m["oracle_threshold"] = float(best_t)
    m["oracle_threshold_accuracy"] = float(best_acc)
    return m


def main():
    data = {name: load_binary(path) for name, path in SENSORS.items()}
    for name, df in data.items():
        print(f"{name}: n={len(df)} drone={int(df.y.sum())} anomaly={int((1-df.y).sum())}")

    models, test_splits, in_distribution = {}, {}, {}
    print("\n--- training one model per sensor (80% split), in-distribution reference ---")
    for name, df in data.items():
        tr, te = train_test_split(df, test_size=0.2, stratify=df.y, random_state=C.SEED)
        clf = train_one(tr)
        models[name] = clf
        test_splits[name] = te
        m = evaluate(clf, te)
        in_distribution[name] = m
        print(f"  {name}: acc={m['accuracy']:.4f} drone_recall={m['recall_drone']:.4f} "
              f"pr_auc={m['pr_auc_drone']:.4f} (n_test={m['n']})")

    print("\n--- cross-sensor zero-shot transfer matrix (train on row, test on column, "
          "ZERO retraining) ---")
    matrix = {}
    for src, clf in models.items():
        matrix[src] = {}
        for tgt, df_tgt in data.items():
            if tgt == src:
                continue
            m = evaluate(clf, df_tgt)
            matrix[src][tgt] = m
            gap = in_distribution[src]["accuracy"] - m["accuracy"]
            oracle_gap = in_distribution[src]["accuracy"] - m["oracle_threshold_accuracy"]
            print(f"  {src:24s} -> {tgt:24s}: acc={m['accuracy']:.4f} (t=0.5)  "
                  f"oracle_acc={m['oracle_threshold_accuracy']:.4f} (t={m['oracle_threshold']:.2f})  "
                  f"pr_auc={m['pr_auc_drone']:.4f}  gap@0.5={gap:+.4f}  oracle_gap={oracle_gap:+.4f}")

    print("\n--- average degradation: naive 0.5 threshold vs. oracle-recalibrated threshold ---")
    degradation = {}
    for src in SENSORS:
        cross_accs = [matrix[src][tgt]["accuracy"] for tgt in SENSORS if tgt != src]
        cross_oracle = [matrix[src][tgt]["oracle_threshold_accuracy"] for tgt in SENSORS if tgt != src]
        cross_pr_auc = [matrix[src][tgt]["pr_auc_drone"] for tgt in SENSORS if tgt != src]
        cross_recalls = [matrix[src][tgt]["recall_drone"] for tgt in SENSORS if tgt != src]
        deg = {
            "in_distribution_accuracy": in_distribution[src]["accuracy"],
            "mean_cross_sensor_accuracy_at_0.5": float(np.mean(cross_accs)),
            "mean_cross_sensor_accuracy_oracle_threshold": float(np.mean(cross_oracle)),
            "mean_cross_sensor_pr_auc": float(np.mean(cross_pr_auc)),
            "accuracy_gap_at_0.5": in_distribution[src]["accuracy"] - float(np.mean(cross_accs)),
            "accuracy_gap_oracle_threshold": in_distribution[src]["accuracy"] - float(np.mean(cross_oracle)),
            "in_distribution_drone_recall": in_distribution[src]["recall_drone"],
            "mean_cross_sensor_drone_recall": float(np.mean(cross_recalls)),
        }
        degradation[src] = deg
        print(f"  {src:24s}: in-dist acc={deg['in_distribution_accuracy']:.4f}  "
              f"cross-sensor@0.5={deg['mean_cross_sensor_accuracy_at_0.5']:.4f}  "
              f"cross-sensor@oracle={deg['mean_cross_sensor_accuracy_oracle_threshold']:.4f}  "
              f"mean_pr_auc={deg['mean_cross_sensor_pr_auc']:.4f}")

    out = {
        "task": "drone vs. anomaly binary classification, 11-feature schema",
        "sensors": {name: {"n": len(df), "n_drone": int(df.y.sum()), "n_anomaly": int((1 - df.y).sum())}
                   for name, df in data.items()},
        "in_distribution_reference": in_distribution,
        "cross_sensor_transfer_matrix": matrix,
        "degradation_summary": degradation,
    }
    (C.REPORTS_DIR / "cross_sensor_transfer_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nresults -> {C.REPORTS_DIR / 'cross_sensor_transfer_results.json'}")


if __name__ == "__main__":
    main()
