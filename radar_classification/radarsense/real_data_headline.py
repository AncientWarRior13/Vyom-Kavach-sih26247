"""Real-data headline model -- CORRECTED for measurement-level leakage.

Earlier this script did a random train_test_split on real_saab77.parquet.
That leaks: the SAAB set is 75,868 segments from 130 continuous recordings
("measurements", median 169 segments each), so a random split puts segments
from the SAME recording of the SAME physical target in train AND test. The
resulting PR-AUC ~0.9999 +/- 0.00002 was a leakage signature, not
generalization -- a mistake, corrected here.

Verified while fixing: even the dataset's OWN 'split' column leaks -- all 130
measurements span more than one split value -- so it cannot be used for an
honest generalization number either. The only leak-free protocol is
GroupKFold on 'measurement': every test fold contains recordings never seen
in training. That is what this now does. The number is lower and true.

Also reported: split-conformal uncertainty, with calibration on
group-disjoint recordings (the honest grouped analogue; exact conformal
exchangeability is weakened by grouping -- disclosed).
"""

import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_sample_weight

from radarsense import config as C
from radarsense.cross_sensor_transfer import load_binary
from radarsense.raddar_cnn import binary_metrics

ALPHA = 0.10
FEATURES = [f for f in C.FEATURE_COLUMNS if f != "doppler_freq_shift"]


def fit(Xtr, ytr, seed=C.SEED):
    clf = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1,
                             random_seed=seed, verbose=0, allow_writing_files=False)
    clf.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))
    return clf


def conformal_sets(proba_cal, y_cal, proba_test, alpha):
    n = len(y_cal)
    scores = 1.0 - proba_cal[np.arange(n), y_cal]
    qhat = np.quantile(scores, min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0), method="higher")
    return proba_test >= (1.0 - qhat), float(qhat)


def main():
    df = pd.read_parquet(C.DATA_DIR / "real_saab77.parquet")
    df = df[df["label"].isin(["drone", "anomaly", "bird"])].copy()
    df["y"] = (df["label"] == "drone").astype(int)
    groups = df["measurement"].values
    print(f"--- REAL SAAB 77GHz: {len(df)} segments, {df['measurement'].nunique()} recordings, "
          f"drone frac {df.y.mean():.3f} ---")
    print("--- measurement-GROUPED 5-fold CV (whole recordings held out; leak-free) ---")

    gkf = GroupKFold(n_splits=5)
    accs, recs, praucs, covs, sizes = [], [], [], [], []
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(df, df.y, groups)):
        tr, te = df.iloc[tr_idx], df.iloc[te_idx]
        tr_groups = tr["measurement"].unique()
        cal_groups = set(tr_groups[::5])
        cal = tr[tr["measurement"].isin(cal_groups)]
        tr = tr[~tr["measurement"].isin(cal_groups)]

        clf = fit(tr[FEATURES].values, tr.y.values, seed=C.SEED + fold)
        p_te = clf.predict_proba(te[FEATURES].values)
        m = binary_metrics(te.y.values, (p_te[:, 1] >= 0.5).astype(int), p_te[:, 1])
        accs.append(m["accuracy"]); recs.append(m["recall_drone"]); praucs.append(m["pr_auc_drone"])

        p_cal = clf.predict_proba(cal[FEATURES].values)
        sets, _ = conformal_sets(p_cal, cal.y.values, p_te, ALPHA)
        covs.append(float(sets[np.arange(len(te)), te.y.values].mean()))
        sizes.append(float(sets.sum(axis=1).mean()))
        print(f"  fold {fold}: acc={m['accuracy']:.4f} recall={m['recall_drone']:.4f} pr_auc={m['pr_auc_drone']:.4f}")

    def ms(v):
        return {"mean": float(np.mean(v)), "std": float(np.std(v))}

    print(f"\n=== leak-free grouped CV (mean +/- std over 5 folds) ===")
    print(f"  accuracy    : {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(f"  drone recall: {np.mean(recs):.4f} +/- {np.std(recs):.4f}")
    print(f"  drone PR-AUC: {np.mean(praucs):.4f} +/- {np.std(praucs):.4f}")
    print(f"  (contrast: the earlier LEAKY random split reported 0.9999 +/- 0.00002 -- that was memorization)")
    print(f"\n=== split-conformal uncertainty (target {1-ALPHA:.0%}) ===")
    print(f"  empirical coverage: {np.mean(covs):.4f} +/- {np.std(covs):.4f}   mean set size: {np.mean(sizes):.3f}")

    out = {
        "data": "SAAB 77GHz real, drone vs rest",
        "protocol": "measurement-GROUPED 5-fold CV (leak-free); doppler_freq_shift dropped (collinear)",
        "n_segments": len(df), "n_recordings": int(df["measurement"].nunique()), "n_features_used": len(FEATURES),
        "accuracy": ms(accs), "recall_drone": ms(recs), "pr_auc_drone": ms(praucs),
        "conformal": {"alpha": ALPHA, "empirical_coverage": ms(covs), "mean_set_size": ms(sizes)},
        "correction_note": ("supersedes an earlier random-split result (PR-AUC 0.9999 +/- 0.00002) that "
                            "leaked whole-recording segments across train/test; the dataset's own split "
                            "column also leaks (all 130 measurements span splits), so grouped CV is the "
                            "only honest protocol."),
    }
    (C.REPORTS_DIR / "real_data_headline_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nresults -> {C.REPORTS_DIR / 'real_data_headline_results.json'}")


if __name__ == "__main__":
    main()
