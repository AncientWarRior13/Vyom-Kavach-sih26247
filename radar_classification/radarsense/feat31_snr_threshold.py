"""SNR-aware (piecewise) drone-decision threshold for the 31-feature model,
vs. the single global threshold feat31_train.py currently uses.

Rationale: a single threshold tuned on the FULL validation set is a
compromise across the entire [-25,-10] dB range. If the model's drone
probability calibration genuinely shifts with SNR (plausible -- lower SNR
means noisier features means less separated probability distributions),
a threshold tuned separately per SNR band, applied per-sample based on
that sample's own SNR bin, should do at least as well and plausibly better
than one global compromise value.

Same 8 SNR bins feat31_train.py already reports per-bin metrics for
(reused, not reinvented). tune_threshold is reused UNMODIFIED from
radarsense.optimize -- same F-beta2 criterion as every other threshold
choice in this project. Bins with too few validation samples to tune
reliably (<20, matching feat31_train.py's own skip threshold) fall back to
the global threshold rather than overfitting a threshold to noise.

Reported honestly regardless of outcome -- piecewise tuning adds real
complexity (8 thresholds instead of 1, a live SNR-bin lookup at inference
time) that's only worth it if it measurably helps; if it's within noise of
the single global threshold, that's reported as the finding, not hidden to
justify the added complexity.
"""

import json

import joblib
import numpy as np
import pandas as pd

from radarsense import config as C
from radarsense.feat31_features import FEAT31_COLUMNS
from radarsense.metrics import confusion, full_metrics
from radarsense.optimize import threshold_predict, tune_threshold

SNR_BINS = [(-25, -24), (-24, -22), (-22, -20), (-20, -18), (-18, -16),
           (-16, -14), (-14, -12), (-12, -10)]
MIN_BIN_N = 20


def assign_bin(snr):
    for lo, hi in SNR_BINS:
        if lo <= snr < hi or (hi == -10 and snr == -10):
            return (lo, hi)
    return None


def main():
    model = joblib.load(C.MODELS_DIR / "feat31_model.joblib")
    val = pd.read_parquet(C.DATA_DIR / "feat31_val_features.parquet")
    test = pd.read_parquet(C.DATA_DIR / "feat31_test_features.parquet")
    classes_ = list(model.classes_)

    def reorder_proba(df):
        proba = model.predict_proba(df[FEAT31_COLUMNS])
        return proba[:, [classes_.index(c) for c in C.CLASSES]]

    proba_val_all = reorder_proba(val)
    t_global, _, _ = tune_threshold(proba_val_all, val[C.LABEL_COLUMN])
    print(f"global threshold (current approach): {t_global:.3f}")

    bin_thresholds = {}
    print("\nper-SNR-bin thresholds (tuned on validation):")
    for lo, hi in SNR_BINS:
        sub = val[(val.snr_db >= lo) & (val.snr_db < hi)]
        if len(sub) < MIN_BIN_N:
            bin_thresholds[(lo, hi)] = t_global
            print(f"  [{lo},{hi}): n={len(sub)} < {MIN_BIN_N}, falling back to global {t_global:.3f}")
            continue
        proba_sub = reorder_proba(sub)
        t_bin, _, _ = tune_threshold(proba_sub, sub[C.LABEL_COLUMN])
        bin_thresholds[(lo, hi)] = t_bin
        print(f"  [{lo},{hi}): n={len(sub)}, threshold={t_bin:.3f}")

    test = test.copy()
    test["_bin"] = test.snr_db.apply(assign_bin)
    proba_test = reorder_proba(test)

    pred_global = threshold_predict(proba_test, t_global)

    pred_piecewise = np.empty(len(test), dtype=object)
    for b, t in bin_thresholds.items():
        mask = (test["_bin"] == b).values
        if mask.any():
            pred_piecewise[mask] = threshold_predict(proba_test[mask], t)
    out_of_bin = test["_bin"].isna().values
    if out_of_bin.any():
        pred_piecewise[out_of_bin] = threshold_predict(proba_test[out_of_bin], t_global)

    def score(y_true, pred, proba, label):
        m = full_metrics(y_true, pred, proba)
        m["confusion_matrix"] = confusion(y_true, pred)
        m["n"] = len(y_true)
        print(f"{label}: n={len(y_true)} accuracy={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f} "
              f"drone_recall={m['recall_drone']:.4f} drone_precision={m['precision_drone']:.4f}")
        return m

    y_true_all = test[C.LABEL_COLUMN].values
    print("\n--- overall test set ---")
    m_global_overall = score(y_true_all, pred_global, proba_test, "global threshold")
    m_piecewise_overall = score(y_true_all, pred_piecewise, proba_test, "piecewise (per-SNR-bin) threshold")

    deep_mask = (test.snr_db <= -23).values
    print("\n--- SNR <= -23dB slice ---")
    m_global_deep = score(y_true_all[deep_mask], pred_global[deep_mask], proba_test[deep_mask], "global threshold")
    m_piecewise_deep = score(y_true_all[deep_mask], pred_piecewise[deep_mask], proba_test[deep_mask],
                              "piecewise (per-SNR-bin) threshold")

    delta_overall = m_piecewise_overall["recall_drone"] - m_global_overall["recall_drone"]
    delta_deep = m_piecewise_deep["recall_drone"] - m_global_deep["recall_drone"]
    print(f"\ndrone recall delta (piecewise - global): overall={delta_overall:+.4f} "
          f"deep_low_snr={delta_deep:+.4f}")

    out = {
        "global_threshold": t_global,
        "per_bin_thresholds": {f"{lo}_to_{hi}": t for (lo, hi), t in bin_thresholds.items()},
        "min_bin_n_for_own_threshold": MIN_BIN_N,
        "overall_test": {"global_threshold": m_global_overall, "piecewise_threshold": m_piecewise_overall},
        "snr_leq_minus23db": {"global_threshold": m_global_deep, "piecewise_threshold": m_piecewise_deep},
        "drone_recall_delta_piecewise_minus_global": {"overall": delta_overall, "deep_low_snr": delta_deep},
    }
    (C.REPORTS_DIR / "feat31_snr_threshold_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nresults -> {C.REPORTS_DIR / 'feat31_snr_threshold_results.json'}")


if __name__ == "__main__":
    main()
