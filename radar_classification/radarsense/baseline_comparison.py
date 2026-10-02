"""Baseline comparison -- the 'no baselines' gap the review flagged. Two kinds,
both honest:

  1. COMPUTED baselines: standard classifiers (majority-class, logistic
     regression, linear SVM, kNN, random forest) run on the SAME real SAAB
     77GHz data as our deployed CatBoost, multi-seed. Shows where our model
     actually sits vs off-the-shelf methods -- a real, reproducible comparison.

  2. REFERENCED external result (clearly provenanced, not re-run here): the
     DIAT-uSAT micro-Doppler CNN (Paul & Bandyopadhyay, IEEE Access 2026)
     reports 97.83% full-precision / 96.87% binarized on the DIAT spectrograms.
     Different model family (image CNN) and a different task split from our
     tabular pipeline -- cited for context, NOT claimed as a head-to-head.
"""

import json

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from catboost import CatBoostClassifier

from radarsense import config as C
from radarsense.cross_sensor_transfer import load_binary
from radarsense.raddar_cnn import binary_metrics

SEEDS = list(range(5))


def models():
    return {
        "majority_class": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
        "linear_svm": make_pipeline(StandardScaler(), SVC(kernel="linear", probability=True, class_weight="balanced")),
        "knn_k15": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15)),
        "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=C.SEED),
        "catboost_OURS": None,
    }


def main():
    df = load_binary("real_saab77.parquet")
    if len(df) > 12000:
        df = df.groupby("y", group_keys=False).apply(
            lambda g: g.sample(int(12000 * len(g) / len(df)), random_state=C.SEED))
    print(f"--- REAL SAAB 77GHz drone-vs-rest, {len(df)} segments (stratified subsample) ---")
    X, y = df[C.FEATURE_COLUMNS].values, df.y.values

    table = {}
    for name in models():
        accs, praucs = [], []
        for seed in SEEDS:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
            if name == "catboost_OURS":
                clf = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1,
                                         random_seed=seed, verbose=0, allow_writing_files=False)
                clf.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))
            else:
                clf = models()[name]
                try:
                    clf.set_params(**{k: seed for k in clf.get_params() if k.endswith("random_state")})
                except Exception:
                    pass
                clf.fit(Xtr, ytr)
            proba = clf.predict_proba(Xte)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(Xte)
            m = binary_metrics(yte, (proba >= 0.5).astype(int), proba)
            accs.append(m["accuracy"]); praucs.append(m["pr_auc_drone"])
        table[name] = {"accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
                       "pr_auc_mean": float(np.mean(praucs))}

    print(f"\n{'model':22s} {'accuracy':>16s} {'PR-AUC':>8s}")
    for name, v in sorted(table.items(), key=lambda kv: kv[1]["accuracy_mean"]):
        star = "  <-- deployed" if name == "catboost_OURS" else ""
        print(f"  {name:22s} {v['accuracy_mean']:.4f} +/- {v['accuracy_std']:.4f}  {v['pr_auc_mean']:.4f}{star}")

    out = {
        "dataset": "SAAB 77GHz real, drone-vs-rest", "n_seeds": len(SEEDS),
        "computed_baselines": table,
        "referenced_external": {
            "DIAT_uSAT_microDoppler_CNN": {
                "source": "Paul & Bandyopadhyay, IEEE Access 2026",
                "full_precision_accuracy": 0.9783, "binarized_accuracy": 0.9687,
                "note": "image-CNN on DIAT spectrograms; different model family and split -- context only, not a head-to-head with our tabular pipeline",
            }
        },
    }
    (C.REPORTS_DIR / "baseline_comparison_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nresults -> {C.REPORTS_DIR / 'baseline_comparison_results.json'}")


if __name__ == "__main__":
    main()
