"""Stage 3: classical model bake-off (vanilla, pre-optimization).

- 6 models, stratified 5-fold CV on the train+val pool, mean +/- std per metric.
- Model selection by drone PR-AUC (primary metric for extreme imbalance).
- Every model also gets a test-set confusion matrix (required to defend claims),
  but the test set plays no role in selection.
- The best vanilla model's test numbers ARE the reproduced "vanilla baseline"
  that the optimization stage (section 4) must beat.

SVM note: RBF SVC is O(n^2); it trains on a stratified 8,000-sample subsample
of each fold's training split. Documented, and applies to SVM only.
"""

import json
import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from radarsense import config as C
from radarsense.metrics import confusion, full_metrics

SVM_SUBSAMPLE = 8000


def build_models():
    return {
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=C.SEED)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=300, n_jobs=-1, random_state=C.SEED),
        "balanced_rf": BalancedRandomForestClassifier(
            n_estimators=300, n_jobs=-1, random_state=C.SEED,
            sampling_strategy="all", replacement=True, bootstrap=False),
        "xgboost": XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.1,
            tree_method="hist", n_jobs=-1, random_state=C.SEED,
            eval_metric="mlogloss"),
        "catboost": CatBoostClassifier(
            iterations=400, depth=6, learning_rate=0.1,
            random_seed=C.SEED, verbose=0, allow_writing_files=False),
        "svm_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=C.SEED)),
        ]),
    }


def _fit(model, name, X, y):
    if name == "svm_rbf" and len(X) > SVM_SUBSAMPLE:
        Xs, _, ys, _ = train_test_split(
            X, y, train_size=SVM_SUBSAMPLE, stratify=y, random_state=C.SEED)
        model.fit(Xs, ys)
    else:
        model.fit(X, y)
    return model


def _encode_labels(y):
    """xgboost needs integer labels; keep everything else on strings."""
    return np.searchsorted(C.CLASSES, y)


def cross_validate(name, model_factory, X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=C.SEED)
    fold_metrics = []
    for tr_idx, va_idx in skf.split(X, y):
        model = model_factory()
        Xtr, ytr = X.iloc[tr_idx], y.iloc[tr_idx]
        Xva, yva = X.iloc[va_idx], y.iloc[va_idx]
        if name == "xgboost":
            _fit(model, name, Xtr, _encode_labels(ytr))
            proba = model.predict_proba(Xva)
            pred = np.array(C.CLASSES)[np.argmax(proba, axis=1)]
        else:
            _fit(model, name, Xtr, ytr)
            proba = model.predict_proba(Xva)
            pred = model.predict(Xva)
        fold_metrics.append(full_metrics(yva, pred, proba))
    keys = fold_metrics[0].keys()
    return {k: {"mean": float(np.mean([m[k] for m in fold_metrics])),
                "std": float(np.std([m[k] for m in fold_metrics]))}
            for k in keys}


def main():
    train = pd.read_parquet(C.DATA_DIR / "radar_train.parquet")
    val = pd.read_parquet(C.DATA_DIR / "radar_val.parquet")
    test = pd.read_parquet(C.DATA_DIR / "radar_test.parquet")
    pool = pd.concat([train, val], ignore_index=True)
    Xp, yp = pool[C.FEATURE_COLUMNS], pool[C.LABEL_COLUMN]
    Xte, yte = test[C.FEATURE_COLUMNS], test[C.LABEL_COLUMN]

    results = {}
    for name in build_models():
        t0 = time.time()
        factory = lambda: build_models()[name]
        cv = cross_validate(name, factory, Xp, yp)

        model = build_models()[name]
        if name == "xgboost":
            _fit(model, name, Xp, _encode_labels(yp))
            proba = model.predict_proba(Xte)
            pred = np.array(C.CLASSES)[np.argmax(proba, axis=1)]
        else:
            _fit(model, name, Xp, yp)
            proba = model.predict_proba(Xte)
            pred = model.predict(Xte)
        test_m = full_metrics(yte, pred, proba)
        results[name] = {
            "cv": cv,
            "test": test_m,
            "test_confusion_matrix": confusion(yte, pred),
            "fit_seconds": round(time.time() - t0, 1),
        }
        joblib.dump(model, C.MODELS_DIR / f"baseline_{name}.joblib")
        print(f"{name:14s} cv drone PR-AUC {cv['pr_auc_drone']['mean']:.3f}"
              f"±{cv['pr_auc_drone']['std']:.3f} | test acc {test_m['accuracy']:.3f}"
              f" | test drone recall {test_m['recall_drone']:.3f}"
              f" precision {test_m['precision_drone']:.3f}"
              f" | {results[name]['fit_seconds']}s")

    best = max(results, key=lambda k: results[k]["cv"]["pr_auc_drone"]["mean"])
    results["_selection"] = {
        "criterion": "cv mean drone PR-AUC",
        "best_model": best,
    }

    imp_model_name = best if best in ("random_forest", "balanced_rf", "xgboost", "catboost") else "xgboost"
    imp_model = joblib.load(C.MODELS_DIR / f"baseline_{imp_model_name}.joblib")
    if imp_model_name == "catboost":
        imp = imp_model.get_feature_importance()
    else:
        imp = imp_model.feature_importances_
    order = np.argsort(imp)[::-1]
    results["_feature_importance"] = {
        "model": imp_model_name,
        "ranking": [{"feature": C.FEATURE_COLUMNS[i], "importance": float(imp[i])}
                    for i in order],
    }

    out = C.REPORTS_DIR / "baseline_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nbest vanilla model: {best}")
    print("top features:", [r["feature"] for r in results["_feature_importance"]["ranking"][:5]])
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
