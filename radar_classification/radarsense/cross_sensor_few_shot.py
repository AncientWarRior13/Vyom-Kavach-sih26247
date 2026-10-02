"""Few-shot cross-sensor adaptation: CROSS_SENSOR_TRANSFER.md found pure
zero-shot transfer degrades substantially (25-63pt accuracy). This asks
the more realistic deployment question: how many labeled examples from
the TARGET sensor does it take to recover most of that gap? Zero
labeled examples (true zero-shot) is a harder bar than most real
deployments would actually face -- a new sensor install can usually
collect at least a small labeled calibration set.

Reuses cross_sensor_transfer.py's exact data loading, model training, and
metrics (imported unmodified) for direct comparability with the zero-shot
numbers already reported there.

Each target sensor's data is split ONCE into an "adaptation pool" (from
which the few-shot examples are drawn) and a "final-eval pool" (never
touched by adaptation, used for every shot-budget level so the comparison
is apples-to-apples as the budget grows) -- the same discipline used
throughout this project to avoid leakage.

Two few-shot strategies tested at each budget:
  1. Threshold-only recalibration: pick the best decision threshold using
     ONLY the small few-shot sample (a realistic version of
     cross_sensor_transfer.py's oracle-threshold diagnostic, which used
     the full target set -- this uses a genuinely small, deployable one).
  2. Full retraining: mix the few-shot target examples into the source
     training data and retrain from scratch.
"""

import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from radarsense import config as C
from radarsense.cross_sensor_transfer import SENSORS, load_binary, train_one
from radarsense.raddar_cnn import binary_metrics

SHOT_BUDGETS = [0, 10, 20, 50]
ADAPTATION_POOL_SIZE = 200


def threshold_only_recalibration(clf, X_shots, y_shots, X_eval, y_eval):
    proba_shots = clf.predict_proba(X_shots)[:, 1]
    thresholds = np.linspace(0.02, 0.98, 49)
    best_t, best_acc = 0.5, -1
    for t in thresholds:
        acc = ((proba_shots >= t).astype(int) == y_shots).mean()
        if acc > best_acc:
            best_acc, best_t = acc, t
    proba_eval = clf.predict_proba(X_eval)[:, 1]
    pred_eval = (proba_eval >= best_t).astype(int)
    m = binary_metrics(y_eval, pred_eval, proba_eval)
    m["threshold_used"] = float(best_t)
    return m


def full_retrain(source_train_df, X_shots, y_shots, shots_df, X_eval, y_eval):
    mixed_X = pd.concat([source_train_df[C.FEATURE_COLUMNS], shots_df[C.FEATURE_COLUMNS]], ignore_index=True)
    mixed_y = np.concatenate([source_train_df.y.values, y_shots])
    sw = compute_sample_weight("balanced", mixed_y)
    clf = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1,
                             random_seed=C.SEED, verbose=0, allow_writing_files=False)
    clf.fit(mixed_X, mixed_y, sample_weight=sw)
    proba_eval = clf.predict_proba(X_eval)[:, 1]
    pred_eval = (proba_eval >= 0.5).astype(int)
    return binary_metrics(y_eval, pred_eval, proba_eval)


def main():
    data = {name: load_binary(path) for name, path in SENSORS.items()}

    print("--- splitting each sensor's data into an adaptation pool + final-eval pool ---")
    adapt_pools, eval_pools = {}, {}
    for name, df in data.items():
        n_adapt = min(ADAPTATION_POOL_SIZE, len(df) // 3)
        adapt, ev = train_test_split(df, train_size=n_adapt, stratify=df.y, random_state=C.SEED)
        adapt_pools[name] = adapt.reset_index(drop=True)
        eval_pools[name] = ev.reset_index(drop=True)
        print(f"  {name}: adapt_pool={len(adapt)} eval_pool={len(ev)}")

    print("\n--- training one full-data source model per sensor (matches cross_sensor_transfer.py) ---")
    models, source_trains = {}, {}
    for name, df in data.items():
        models[name] = train_one(df)
        source_trains[name] = df

    print("\n--- few-shot adaptation, all 12 pairs x shot budgets", SHOT_BUDGETS, "---")
    results = []
    for src in SENSORS:
        for tgt in SENSORS:
            if tgt == src:
                continue
            X_eval, y_eval = eval_pools[tgt][C.FEATURE_COLUMNS], eval_pools[tgt].y.values
            for n_shots in SHOT_BUDGETS:
                if n_shots == 0:
                    proba = models[src].predict_proba(X_eval)[:, 1]
                    pred = (proba >= 0.5).astype(int)
                    m_thresh = binary_metrics(y_eval, pred, proba)
                    m_retrain = m_thresh
                else:
                    n_per_class = max(1, n_shots // 2)
                    pool = adapt_pools[tgt]
                    shots = pd.concat([
                        pool[pool.y == 0].sample(min(n_per_class, (pool.y == 0).sum()), random_state=C.SEED),
                        pool[pool.y == 1].sample(min(n_per_class, (pool.y == 1).sum()), random_state=C.SEED),
                    ]).reset_index(drop=True)
                    X_shots, y_shots = shots[C.FEATURE_COLUMNS], shots.y.values

                    m_thresh = threshold_only_recalibration(models[src], X_shots, y_shots, X_eval, y_eval)
                    m_retrain = full_retrain(source_trains[src], X_shots, y_shots, shots, X_eval, y_eval)

                results.append({"source": src, "target": tgt, "n_shots": n_shots,
                                "strategy": "threshold_only", **{k: v for k, v in m_thresh.items()}})
                results.append({"source": src, "target": tgt, "n_shots": n_shots,
                                "strategy": "full_retrain", **{k: v for k, v in m_retrain.items()}})
        print(f"  {src}: done")

    df_res = pd.DataFrame(results)
    summary = (df_res.groupby(["n_shots", "strategy"])
                 .agg(accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
                      recall_drone_mean=("recall_drone", "mean"))
                 .reset_index())
    print("\n--- averaged across all 12 source->target pairs ---")
    print(summary.to_string(index=False))

    out = {
        "shot_budgets": SHOT_BUDGETS,
        "adaptation_pool_size_target": ADAPTATION_POOL_SIZE,
        "summary_mean_std": summary.to_dict(orient="records"),
        "per_pair_results": results,
    }
    (C.REPORTS_DIR / "cross_sensor_few_shot_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nresults -> {C.REPORTS_DIR / 'cross_sensor_few_shot_results.json'}")


if __name__ == "__main__":
    main()
