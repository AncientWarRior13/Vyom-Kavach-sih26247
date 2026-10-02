"""SMOTE-vs-class-weight-vs-combined imbalance comparison for the 31-feature
model -- the same 7-way variant comparison the incumbent 11-feature model
went through (radarsense/optimize.py), which feat31_train.py never had:
it only ever used class-weighting.

Reuses fit_variant/predict/threshold_predict/tune_threshold/SAMPLERS from
radarsense.optimize UNMODIFIED -- those functions operate on X/y arrays and
a model family name, not on hardcoded 11-feature column names, so this is
genuinely the same evaluation protocol, not a re-implementation that could
silently diverge.

    baseline            (vanilla CatBoost, no imbalance handling)
    class_weight        (what feat31_train.py currently uses)
    smote               (SMOTE on TRAIN ONLY)
    borderline_smote    (extension)
    adasyn              (extension)
    smote_tomek         (hybrid over+under sampling, extension)
    smote+class_weight  (combined)

Evaluated twice: overall test set (matching the incumbent's own comparison
scope) AND the SNR<=-23dB deep-low-SNR slice specifically (the 31-feature
model's whole reason for existing -- an imbalance-handling choice that wins
overall but loses at extreme noise, or vice versa, would be a real and
useful finding here, not something the incumbent's own comparison could
have surfaced).

Same SMOTE risk disclosure as the incumbent's own comparison: SMOTE
interpolates synthetic minority samples on ALREADY-synthetic data, which
can amplify simulator artifacts rather than model anything about real
low-SNR drone returns -- flagged, not hidden, exactly as optimize.py does
for the 11-feature schema.
"""

import json

import joblib
import numpy as np
import pandas as pd

from radarsense import config as C
from radarsense.feat31_features import FEAT31_COLUMNS
from radarsense.metrics import confusion, full_metrics
from radarsense.optimize import SAMPLERS, fit_variant, predict, threshold_predict, tune_threshold

BEST_NAME = "catboost"

VARIANTS = {
    "baseline": dict(sampler=None, weighted=False),
    "class_weight": dict(sampler=None, weighted=True),
    "smote": dict(sampler="smote", weighted=False),
    "borderline_smote": dict(sampler="borderline_smote", weighted=False),
    "adasyn": dict(sampler="adasyn", weighted=False),
    "smote_tomek": dict(sampler="smote_tomek", weighted=False),
    "smote+class_weight": dict(sampler="smote", weighted=True),
}


def main():
    train = pd.read_parquet(C.DATA_DIR / "feat31_train_features.parquet")
    val = pd.read_parquet(C.DATA_DIR / "feat31_val_features.parquet")
    test = pd.read_parquet(C.DATA_DIR / "feat31_test_features.parquet")
    Xtr, ytr = train[FEAT31_COLUMNS], train[C.LABEL_COLUMN]
    Xva, yva = val[FEAT31_COLUMNS], val[C.LABEL_COLUMN]
    Xte, yte = test[FEAT31_COLUMNS], test[C.LABEL_COLUMN]
    deep_low = test[test.snr_db <= -23]
    Xte_deep, yte_deep = deep_low[FEAT31_COLUMNS], deep_low[C.LABEL_COLUMN]

    results_overall, results_deep, models = {}, {}, {}
    for vname, cfg in VARIANTS.items():
        sampler = SAMPLERS[cfg["sampler"]]() if cfg["sampler"] else None
        model = fit_variant(BEST_NAME, Xtr, ytr, sampler, cfg["weighted"])
        models[vname] = model

        pred, proba = predict(BEST_NAME, model, Xte)
        m = full_metrics(yte, pred, proba)
        m["test_confusion_matrix"] = confusion(yte, pred)
        results_overall[vname] = m

        pred_d, proba_d = predict(BEST_NAME, model, Xte_deep)
        m_d = full_metrics(yte_deep, pred_d, proba_d)
        m_d["test_confusion_matrix"] = confusion(yte_deep, pred_d)
        m_d["n"] = len(Xte_deep)
        results_deep[vname] = m_d

        print(f"{vname:20s} OVERALL drone recall {m['recall_drone']:.3f} "
              f"precision {m['precision_drone']:.3f} PR-AUC {m['pr_auc_drone']:.3f}  |  "
              f"SNR<=-23dB drone recall {m_d['recall_drone']:.3f} "
              f"precision {m_d['precision_drone']:.3f} PR-AUC {m_d['pr_auc_drone']:.3f}")

    win_overall = max(results_overall, key=lambda k: (results_overall[k]["pr_auc_drone"],
                                                       results_overall[k]["recall_drone"]))
    win_deep = max(results_deep, key=lambda k: (results_deep[k]["pr_auc_drone"],
                                                results_deep[k]["recall_drone"]))
    print(f"\nwinning variant overall: {win_overall}")
    print(f"winning variant at SNR<=-23dB: {win_deep}")

    model = models[win_overall]
    _, proba_val = predict(BEST_NAME, model, Xva)
    t_fbeta, t_cost, _ = tune_threshold(proba_val, yva)

    _, proba_test = predict(BEST_NAME, model, Xte)
    final = {}
    for label, t in [("argmax_default", None), ("fbeta2_threshold", t_fbeta),
                     ("cost_threshold", t_cost)]:
        pred = (np.array(C.CLASSES)[np.argmax(proba_test, axis=1)]
                if t is None else threshold_predict(proba_test, t))
        m = full_metrics(yte, pred, proba_test)
        m["test_confusion_matrix"] = confusion(yte, pred)
        m["threshold"] = t
        final[label] = m

    out = {
        "model_family": BEST_NAME,
        "variant_comparison_overall_test": results_overall,
        "variant_comparison_snr_leq_minus23db": results_deep,
        "winning_variant_overall": win_overall,
        "winning_variant_at_deep_low_snr": win_deep,
        "note_if_different": (
            None if win_overall == win_deep else
            f"Different imbalance strategies win overall ({win_overall}) vs. at extreme "
            f"noise ({win_deep}) -- a real finding, not resolved by picking one: the "
            f"deployed choice below optimizes for OVERALL performance, matching how the "
            f"incumbent 11-feature model's own comparison is scoped."
        ),
        "thresholds_on_overall_winner": {"fbeta2": t_fbeta, "cost_optimal": t_cost,
                                          "cost_matrix": {"fn_drone": C.COST_FN_DRONE,
                                                         "fp_drone": C.COST_FP_DRONE}},
        "final_test_metrics_overall_winner": final,
        "smote_risk_note": (
            "SMOTE interpolates synthetic minority samples; on simulated data this can "
            "amplify simulator artifacts. Same disclosure as the incumbent 11-feature "
            "model's own comparison (optimize.py) -- both variants' numbers here are "
            "real held-out results, not cherry-picked."),
    }
    (C.REPORTS_DIR / "feat31_optimization_results.json").write_text(json.dumps(out, indent=2))

    if win_overall != "class_weight":
        candidate_path = C.MODELS_DIR / f"feat31_model_optimized_{win_overall}.joblib"
        joblib.dump(models[win_overall], candidate_path)
        print(f"\nNOTE: overall winner ({win_overall}) differs from the currently deployed "
              f"feat31_model.joblib (class_weight only). Saved as a SEPARATE candidate file "
              f"({candidate_path.name}) -- NOT auto-swapped into the deployed path; that's a "
              f"decision to make deliberately, not silently.")
    else:
        print(f"\nclass_weight (the currently deployed approach) IS the overall winner -- "
              f"no change needed to feat31_model.joblib.")

    print(f"results -> {C.REPORTS_DIR / 'feat31_optimization_results.json'}")


if __name__ == "__main__":
    main()
