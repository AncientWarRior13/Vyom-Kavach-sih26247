"""Decision-level multi-sensor fusion across the four real sensors -- the
'sensor fusion' gap the review flagged. Honest framing: the four public
datasets do NOT co-observe the same targets, so true track-level
co-observation fusion isn't possible here. What IS: train one specialist
model per sensor, then fuse the four specialists' decisions on a POOLED
cross-sensor test stream (each sample scored by all four specialists), and
show that decision-level fusion is more robust across sensors than any single
specialist -- the deployment case where a target could be seen by any of
several fielded sensor types.

Fusion rules: mean probability, and a stacked meta-learner (logistic
regression on the four specialists' drone-probabilities, fit on a held-out
fusion-calibration split). Compared against the best single specialist and a
naively pooled-trained model.
"""

import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from radarsense import config as C
from radarsense.cross_sensor_transfer import SENSORS, load_binary
from radarsense.raddar_cnn import binary_metrics


def fit(Xtr, ytr, seed=C.SEED):
    clf = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1,
                             random_seed=seed, verbose=0, allow_writing_files=False)
    clf.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))
    return clf


def main():
    data = {name: load_binary(path) for name, path in SENSORS.items()}
    names = list(SENSORS)

    specialists, cal_parts, test_parts, pooled_train = {}, [], [], []
    for name, df in data.items():
        tr, rest = train_test_split(df, test_size=0.4, stratify=df.y, random_state=C.SEED)
        cal, te = train_test_split(rest, test_size=0.5, stratify=rest.y, random_state=C.SEED)
        specialists[name] = fit(tr[C.FEATURE_COLUMNS].values, tr.y.values)
        cal_parts.append(cal); test_parts.append(te); pooled_train.append(tr)

    cal = pd.concat(cal_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    pooled_model = fit(pd.concat(pooled_train, ignore_index=True)[C.FEATURE_COLUMNS].values,
                       pd.concat(pooled_train, ignore_index=True).y.values)

    def stack_matrix(df):
        return np.column_stack([specialists[n].predict_proba(df[C.FEATURE_COLUMNS].values)[:, 1]
                                for n in names])

    Pcal, Ptest = stack_matrix(cal), stack_matrix(test)
    yte = test.y.values

    results = {}
    for i, n in enumerate(names):
        m = binary_metrics(yte, (Ptest[:, i] >= 0.5).astype(int), Ptest[:, i])
        results[f"single_{n}"] = {"accuracy": m["accuracy"], "pr_auc": m["pr_auc_drone"]}
    best_single = max((f"single_{n}" for n in names), key=lambda k: results[k]["accuracy"])

    mean_p = Ptest.mean(axis=1)
    m = binary_metrics(yte, (mean_p >= 0.5).astype(int), mean_p)
    results["fusion_mean"] = {"accuracy": m["accuracy"], "pr_auc": m["pr_auc_drone"]}

    meta = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Pcal, cal.y.values)
    stk = meta.predict_proba(Ptest)[:, 1]
    m = binary_metrics(yte, (stk >= 0.5).astype(int), stk)
    results["fusion_stacked"] = {"accuracy": m["accuracy"], "pr_auc": m["pr_auc_drone"]}

    pp = pooled_model.predict_proba(test[C.FEATURE_COLUMNS].values)[:, 1]
    m = binary_metrics(yte, (pp >= 0.5).astype(int), pp)
    results["pooled_trained"] = {"accuracy": m["accuracy"], "pr_auc": m["pr_auc_drone"]}

    print("=== multi-sensor fusion on pooled cross-sensor test ===")
    for k, v in sorted(results.items(), key=lambda kv: kv[1]["accuracy"]):
        print(f"  {k:28s} acc={v['accuracy']:.4f}  pr_auc={v['pr_auc']:.4f}")
    gain = results["fusion_stacked"]["accuracy"] - results[best_single]["accuracy"]
    print(f"\n  stacked fusion vs best single specialist ({best_single}): {gain:+.4f} accuracy")

    out = {"task": "drone-vs-rest, pooled cross-sensor test, decision-level fusion",
           "best_single_specialist": best_single, "results": results,
           "fusion_gain_over_best_single": float(gain)}
    (C.REPORTS_DIR / "multi_sensor_fusion_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nresults -> {C.REPORTS_DIR / 'multi_sensor_fusion_results.json'}")


if __name__ == "__main__":
    main()
