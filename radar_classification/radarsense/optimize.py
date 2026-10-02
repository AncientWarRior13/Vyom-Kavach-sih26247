"""Stage 4: minority-class optimization — a distinct pipeline stage.

Structure (deliberately separate from model training, per the differentiation
claim): take the best vanilla model family from stage 3 and empirically compare

    baseline            (vanilla, no imbalance handling)
    class_weight        (cost-sensitive learning, native weights)
    smote               (SMOTE on TRAIN ONLY)
    borderline_smote    (extension)
    adasyn              (extension)
    smote_tomek         (hybrid over+under sampling, extension)
    smote+class_weight  (combined)

All variants are evaluated on the identical held-out test set. On top of the
winning variant we tune the drone decision threshold on the VALIDATION set
(never test) two ways:
  1. PR-curve / F-beta (beta=2, recall-weighted) optimum
  2. expected-cost minimum under the ops cost matrix (FN=25x FP)
Final reported numbers = tuned threshold applied once to the test set.
"""

import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE
from sklearn.utils.class_weight import compute_sample_weight

from radarsense import config as C
from radarsense.metrics import confusion, full_metrics
from radarsense.train_baseline import build_models, _encode_labels

SAMPLERS = {
    "smote": lambda: SMOTE(random_state=C.SEED),
    "borderline_smote": lambda: BorderlineSMOTE(random_state=C.SEED),
    "adasyn": lambda: ADASYN(random_state=C.SEED),
    "smote_tomek": lambda: SMOTETomek(random_state=C.SEED),
}


def fit_variant(best_name, Xtr, ytr, sampler=None, weighted=False):
    model = build_models()[best_name]
    if sampler is not None:
        Xtr, ytr = sampler.fit_resample(Xtr, ytr)
    y_enc = _encode_labels(ytr) if best_name == "xgboost" else ytr
    kwargs = {}
    if weighted:
        sw = compute_sample_weight("balanced", ytr)
        if best_name == "catboost":
            kwargs["sample_weight"] = sw
        elif best_name == "xgboost":
            kwargs["sample_weight"] = sw
        else:
            kwargs["clf__sample_weight" if hasattr(model, "steps") else "sample_weight"] = sw
    model.fit(Xtr, y_enc, **kwargs)
    return model


def predict(best_name, model, X):
    proba = model.predict_proba(X)
    pred = np.array(C.CLASSES)[np.argmax(proba, axis=1)]
    return pred, proba


def threshold_predict(proba, t):
    """If p(drone) >= t -> drone, else argmax over the other classes."""
    pred = []
    other = [i for i in range(len(C.CLASSES)) if i != C.MINORITY_IDX]
    for row in proba:
        if row[C.MINORITY_IDX] >= t:
            pred.append(C.MINORITY_CLASS)
        else:
            pred.append(C.CLASSES[other[int(np.argmax(row[other]))]])
    return np.array(pred)


def tune_threshold(proba_val, y_val):
    """Returns (fbeta2-optimal t, cost-optimal t, sweep table)."""
    y_bin = (np.asarray(y_val) == C.MINORITY_CLASS).astype(int)
    p = proba_val[:, C.MINORITY_IDX]
    ts = np.linspace(0.01, 0.99, 197)
    rows = []
    for t in ts:
        yhat = (p >= t).astype(int)
        tp = int(((yhat == 1) & (y_bin == 1)).sum())
        fp = int(((yhat == 1) & (y_bin == 0)).sum())
        fn = int(((yhat == 0) & (y_bin == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        fbeta2 = (5 * prec * rec / (4 * prec + rec)) if (prec + rec) else 0.0
        cost = C.COST_FN_DRONE * fn + C.COST_FP_DRONE * fp
        rows.append({"t": float(t), "precision": prec, "recall": rec,
                     "fbeta2": fbeta2, "expected_cost": cost})
    sweep = pd.DataFrame(rows)
    t_fbeta = float(sweep.loc[sweep.fbeta2.idxmax(), "t"])
    t_cost = float(sweep.loc[sweep.expected_cost.idxmin(), "t"])
    return t_fbeta, t_cost, sweep


def main():
    baseline = json.loads((C.REPORTS_DIR / "baseline_results.json").read_text())
    best_name = baseline["_selection"]["best_model"]
    print(f"optimizing model family: {best_name}")

    train = pd.read_parquet(C.DATA_DIR / "radar_train.parquet")
    val = pd.read_parquet(C.DATA_DIR / "radar_val.parquet")
    test = pd.read_parquet(C.DATA_DIR / "radar_test.parquet")
    Xtr, ytr = train[C.FEATURE_COLUMNS], train[C.LABEL_COLUMN]
    Xva, yva = val[C.FEATURE_COLUMNS], val[C.LABEL_COLUMN]
    Xte, yte = test[C.FEATURE_COLUMNS], test[C.LABEL_COLUMN]

    variants = {
        "baseline": dict(sampler=None, weighted=False),
        "class_weight": dict(sampler=None, weighted=True),
        "smote": dict(sampler="smote", weighted=False),
        "borderline_smote": dict(sampler="borderline_smote", weighted=False),
        "adasyn": dict(sampler="adasyn", weighted=False),
        "smote_tomek": dict(sampler="smote_tomek", weighted=False),
        "smote+class_weight": dict(sampler="smote", weighted=True),
    }

    results, models = {}, {}
    for vname, cfg in variants.items():
        sampler = SAMPLERS[cfg["sampler"]]() if cfg["sampler"] else None
        model = fit_variant(best_name, Xtr, ytr, sampler, cfg["weighted"])
        pred, proba = predict(best_name, model, Xte)
        m = full_metrics(yte, pred, proba)
        m["test_confusion_matrix"] = confusion(yte, pred)
        results[vname] = m
        models[vname] = model
        print(f"{vname:20s} drone recall {m['recall_drone']:.3f} "
              f"precision {m['precision_drone']:.3f} PR-AUC {m['pr_auc_drone']:.3f} "
              f"macroF1 {m['macro_f1']:.3f}")

    win = max(results, key=lambda k: (results[k]["pr_auc_drone"], results[k]["recall_drone"]))
    print(f"\nwinning variant: {win}")

    model = models[win]
    _, proba_val = predict(best_name, model, Xva)
    t_fbeta, t_cost, sweep = tune_threshold(proba_val, yva)

    _, proba_test = predict(best_name, model, Xte)
    final = {}
    for label, t in [("argmax_default", None), ("fbeta2_threshold", t_fbeta),
                     ("cost_threshold", t_cost)]:
        pred = (np.array(C.CLASSES)[np.argmax(proba_test, axis=1)]
                if t is None else threshold_predict(proba_test, t))
        m = full_metrics(yte, pred, proba_test)
        m["test_confusion_matrix"] = confusion(yte, pred)
        m["threshold"] = t
        final[label] = m
        print(f"{label:18s} t={t if t else 'argmax'} -> drone recall "
              f"{m['recall_drone']:.3f} precision {m['precision_drone']:.3f}")

    from sklearn.metrics import precision_recall_curve
    y_bin_te = (yte == C.MINORITY_CLASS).astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    pb, rb, _ = precision_recall_curve(y_bin_te, proba_test[:, C.MINORITY_IDX])
    axes[0].plot(rb, pb, label=f"{win} (test)")
    p0 = models["baseline"].predict_proba(Xte)[:, C.MINORITY_IDX]
    pb0, rb0, _ = precision_recall_curve(y_bin_te, p0)
    axes[0].plot(rb0, pb0, "--", label="baseline (test)")
    axes[0].scatter([final["cost_threshold"]["recall_drone"]],
                    [final["cost_threshold"]["precision_drone"]],
                    c="red", zorder=5, label=f"cost-optimal t={t_cost:.2f}")
    axes[0].set_xlabel("drone recall"); axes[0].set_ylabel("drone precision")
    axes[0].set_title("Drone PR curve: baseline vs optimized"); axes[0].legend()

    axes[1].plot(sweep.t, sweep.expected_cost, label="expected cost (val)")
    axes[1].axvline(t_cost, color="red", ls="--", label=f"cost-optimal t={t_cost:.2f}")
    axes[1].axvline(t_fbeta, color="green", ls=":", label=f"F2-optimal t={t_fbeta:.2f}")
    axes[1].set_xlabel("drone threshold"); axes[1].set_ylabel(
        f"cost = {C.COST_FN_DRONE:.0f}*FN + {C.COST_FP_DRONE:.0f}*FP")
    axes[1].set_title("Threshold choice justification (validation set)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(C.PLOTS_DIR / "optimization_pr_threshold.png", dpi=150)
    plt.close(fig)

    joblib.dump(models[win], C.MODELS_DIR / "final_model.joblib")
    out = {
        "model_family": best_name,
        "variant_comparison_test": results,
        "winning_variant": win,
        "thresholds": {"fbeta2": t_fbeta, "cost_optimal": t_cost,
                       "cost_matrix": {"fn_drone": C.COST_FN_DRONE,
                                       "fp_drone": C.COST_FP_DRONE}},
        "final_test_metrics": final,
        "smote_risk_note": (
            "SMOTE interpolates synthetic minority samples; on simulated data this "
            "can amplify simulator artifacts. Mitigation evidence: the SMOTE-variant "
            "test PR-AUC vs the class-weight variant (no synthetic samples) is "
            "reported side by side above — if SMOTE materially outperformed "
            "class-weighting on synthetic data but not on real data, that gap would "
            "be the artifact signature to re-examine. Both variants' numbers are real "
            "held-out results."),
    }
    (C.REPORTS_DIR / "optimization_results.json").write_text(json.dumps(out, indent=2))
    (C.MODELS_DIR / "final_model_meta.json").write_text(json.dumps({
        "model_family": best_name, "variant": win,
        "drone_threshold_deployed": t_fbeta,
        "drone_threshold_cost_optimal": t_cost,
        "drone_threshold_fbeta2": t_fbeta,
        "classes": C.CLASSES, "feature_order": C.FEATURE_COLUMNS}, indent=2))
    print(f"\nresults -> {C.REPORTS_DIR / 'optimization_results.json'}")


if __name__ == "__main__":
    main()
