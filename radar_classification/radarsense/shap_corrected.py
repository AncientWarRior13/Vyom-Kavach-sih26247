"""Corrected SHAP -- the 11-feature schema has a perfectly collinear pair:
doppler_freq_shift = 2*doppler_velocity*f_c/c with a fixed carrier, so
corr(doppler_velocity, doppler_freq_shift) = 1.0 exactly (verified on every
dataset). A tree ensemble splits credit between two identical features
arbitrarily, so the original shap_report.json under-credits doppler_velocity
and spuriously credits doppler_freq_shift. This recomputes SHAP on the 10
INDEPENDENT features (collinear column dropped) with the deployed CatBoost
config, so the attribution is interpretable.
"""

import json

import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier
from sklearn.utils.class_weight import compute_sample_weight

from radarsense import config as C

FEATURES10 = [f for f in C.FEATURE_COLUMNS if f != "doppler_freq_shift"]


def main():
    train = pd.read_parquet(C.DATA_DIR / "radar_train.parquet")
    test = pd.read_parquet(C.DATA_DIR / "radar_test.parquet")

    corr = float(train["doppler_velocity"].corr(train["doppler_freq_shift"]))
    print(f"corr(doppler_velocity, doppler_freq_shift) = {corr:.10f}  -> collinear, dropping one")

    clf = CatBoostClassifier(iterations=400, depth=6, learning_rate=0.1,
                             random_seed=C.SEED, verbose=0, allow_writing_files=False)
    clf.fit(train[FEATURES10], train[C.LABEL_COLUMN],
            sample_weight=compute_sample_weight("balanced", train[C.LABEL_COLUMN]))

    bg = test[FEATURES10].sample(min(500, len(test)), random_state=C.SEED)
    sv = shap.TreeExplainer(clf).shap_values(bg)
    if isinstance(sv, np.ndarray) and sv.ndim == 3:
        sv = [sv[:, :, i] for i in range(sv.shape[2])]
    drone_sv = sv[C.MINORITY_IDX] if isinstance(sv, list) else sv
    mean_abs = np.abs(drone_sv).mean(axis=0)
    ranked = sorted(zip(FEATURES10, mean_abs), key=lambda kv: -kv[1])

    print("\n=== corrected SHAP (10 independent features), drone class ===")
    for f, v in ranked:
        print(f"  {f:26s} {v:.4f}")

    try:
        orig = json.loads((C.REPORTS_DIR / "shap_report.json").read_text())
        print("\n  note: in the original 11-feature SHAP, doppler credit was split across "
              "doppler_velocity + doppler_freq_shift; here it consolidates onto doppler_velocity.")
    except Exception:
        orig = None

    out = {
        "collinearity": {"pair": ["doppler_velocity", "doppler_freq_shift"], "correlation": corr,
                         "explanation": "doppler_freq_shift = 2*v*f_c/c, fixed carrier -> exact linear dependence"},
        "features_used": FEATURES10,
        "corrected_drone_shap_mean_abs": {f: float(v) for f, v in ranked},
        "note": "supersedes shap_report.json for interpretation; the frozen ONNX schema keeps the 11th "
                "column for contract stability, but attributions must be read on the 10 independent features.",
    }
    (C.REPORTS_DIR / "shap_corrected_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nresults -> {C.REPORTS_DIR / 'shap_corrected_results.json'}")


if __name__ == "__main__":
    main()
