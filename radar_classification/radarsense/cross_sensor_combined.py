"""WS4 follow-up: CROSS_SENSOR_DOMAIN_ADAPT.md measured four recovery
mechanisms SEPARATELY (CORAL zero-label, leave-one-out pooling, few-shot,
recalibration). Each helped; none was combined. This asks whether stacking
them beats the best single one (few-shot's 82% at 500 shots).

Combined pipeline, per target sensor:
  1. POOL the other three sensors' data (multi-source, from the leave-one-out
     win).
  2. CORAL-align the pooled source to the target's UNLABELED features.
  3. Mix in a few LABELED target shots and retrain.
  4. RECALIBRATE the decision threshold on the shots (the leave-one-out
     diagnostic showed some sensors have near-perfect ranking but a wrong
     0.5 threshold -- SAAB PR-AUC 0.946).

Compared head-to-head against the best single mechanism at matched shot
budgets. Reuses cross_sensor_domain_adapt.py's CORAL + data loading.
"""

import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from radarsense import config as C
from radarsense.cross_sensor_transfer import SENSORS, load_binary
from radarsense.cross_sensor_domain_adapt import coral_align
from radarsense.raddar_cnn import binary_metrics

FEATURES = C.FEATURE_COLUMNS
SHOT_BUDGETS = [50, 100, 200, 500]


def fit(X, y):
    sw = compute_sample_weight("balanced", y)
    clf = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1,
                             random_seed=C.SEED, verbose=0, allow_writing_files=False)
    clf.fit(X, y, sample_weight=sw)
    return clf


def best_threshold(proba, y):
    ts = np.linspace(0.05, 0.95, 46)
    return float(max(ts, key=lambda t: ((proba >= t).astype(int) == y).mean()))


def main():
    data = {name: load_binary(path) for name, path in SENSORS.items()}
    splits = {}
    for name, df in data.items():
        tr, te = train_test_split(df, test_size=0.3, stratify=df.y, random_state=C.SEED)
        splits[name] = (tr.reset_index(drop=True), te.reset_index(drop=True))
    names = list(SENSORS)

    fewshot_only = {b: [] for b in SHOT_BUDGETS}
    combined = {b: [] for b in SHOT_BUDGETS}
    for tgt in names:
        tgt_tr, tgt_te = splits[tgt]
        Xe, ye = tgt_te[FEATURES].values, tgt_te.y.values
        pool = pd.concat([splits[s][0] for s in names if s != tgt], ignore_index=True)
        Xpool, ypool = pool[FEATURES].values, pool.y.values
        Xpool_coral = coral_align(Xpool, tgt_tr[FEATURES].values)
        for b in SHOT_BUDGETS:
            npc = max(1, b // 2)
            shots = pd.concat([
                tgt_tr[tgt_tr.y == 0].sample(min(npc, int((tgt_tr.y == 0).sum())), random_state=C.SEED),
                tgt_tr[tgt_tr.y == 1].sample(min(npc, int((tgt_tr.y == 1).sum())), random_state=C.SEED),
            ])
            Xsh, ysh = shots[FEATURES].values, shots.y.values

            m_fs = fit(np.vstack([Xpool, Xsh]), np.concatenate([ypool, ysh]))
            p_fs = m_fs.predict_proba(Xe)[:, 1]
            fewshot_only[b].append(((p_fs >= 0.5).astype(int) == ye).mean())

            m_cb = fit(np.vstack([Xpool_coral, Xsh]), np.concatenate([ypool, ysh]))
            p_shots = m_cb.predict_proba(Xsh)[:, 1]
            t = best_threshold(p_shots, ysh)
            p_cb = m_cb.predict_proba(Xe)[:, 1]
            combined[b].append(((p_cb >= t).astype(int) == ye).mean())

    fs_mean = {b: float(np.mean(v)) for b, v in fewshot_only.items()}
    cb_mean = {b: float(np.mean(v)) for b, v in combined.items()}
    print(f"  {'shots':>6} {'pooled+fewshot':>15} {'COMBINED(all 4)':>16}")
    for b in SHOT_BUDGETS:
        print(f"  {b:>6} {fs_mean[b]:>15.4f} {cb_mean[b]:>16.4f}")

    out = {"pooled_fewshot_mean_acc": fs_mean, "combined_all_mechanisms_mean_acc": cb_mean,
           "single_best_prior_fewshot_500_shot_baseline": 0.8198}
    (C.REPORTS_DIR / "cross_sensor_combined_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nresults -> {C.REPORTS_DIR / 'cross_sensor_combined_results.json'}")


if __name__ == "__main__":
    main()
