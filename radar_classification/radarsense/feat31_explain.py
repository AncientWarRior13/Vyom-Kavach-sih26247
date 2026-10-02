"""SHAP explainability for the 31-feature model. Same structure as
radarsense/explain.py: global summary, per-detection waterfall, and a
physics-plausibility check -- adapted because the physically-meaningful
drone signal here is spread across several raw-statistic features
(velocity jerk/jitter -> velocity_std/velocity_diff_var; rotor-blade
modulation -> amplitude_dominant_freq/amplitude_std) rather than one named
feature like the 11-feature schema's micro_doppler_bandwidth. The check
reports the actual top-3 and reasons about physical plausibility rather
than hard-failing on one specific expected name, since which of several
physically-plausible features ranks highest is not something either
schema's design promises in advance.
"""

import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from radarsense import config as C
from radarsense.feat31_features import FEAT31_COLUMNS
from radarsense.optimize import threshold_predict

PLAUSIBLE_DRONE_SIGNALS = {
    "velocity_std", "velocity_diff_var", "velocity_range", "velocity_energy",
    "amplitude_std", "amplitude_dominant_freq", "amplitude_diff_var",
    "amplitude_spectral_bandwidth", "energy_std", "energy_diff_var",
    "energy_dominant_freq", "energy_spectral_bandwidth",
}


def main():
    model = joblib.load(C.MODELS_DIR / "feat31_model.joblib")
    results = json.loads((C.REPORTS_DIR / "feat31_results.json").read_text())
    t = results["threshold_fbeta2_optimal"]
    test = pd.read_parquet(C.DATA_DIR / "feat31_test_features.parquet")
    X = test[FEAT31_COLUMNS]
    classes_ = list(model.classes_)
    drone_idx = classes_.index("drone")

    bg = X.sample(min(2000, len(X)), random_state=C.SEED)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(bg)
    if isinstance(sv, np.ndarray) and sv.ndim == 3:
        sv = [sv[:, :, i] for i in range(sv.shape[2])]

    plt.figure()
    shap.summary_plot(sv[drone_idx], bg, feature_names=FEAT31_COLUMNS, show=False, max_display=31)
    plt.title("SHAP summary — drone class (31-feature model)")
    plt.tight_layout()
    plt.savefig(C.PLOTS_DIR / "feat31_shap_summary_drone.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    mean_abs = np.mean([np.abs(s).mean(axis=0) for s in sv], axis=0)
    order = np.argsort(mean_abs)
    plt.figure(figsize=(8, 9))
    plt.barh([FEAT31_COLUMNS[i] for i in order], mean_abs[order], color="#4878CF")
    plt.xlabel("mean |SHAP| (all classes)")
    plt.title("Global feature importance (31-feature model)")
    plt.tight_layout()
    plt.savefig(C.PLOTS_DIR / "feat31_shap_summary_global.png", dpi=150)
    plt.close("all")

    drone_rank = np.argsort(np.abs(sv[drone_idx]).mean(axis=0))[::-1]
    top5 = [FEAT31_COLUMNS[i] for i in drone_rank[:5]]
    plausible = [f for f in top5 if f in PLAUSIBLE_DRONE_SIGNALS]
    print(f"drone-class top-5 SHAP features: {top5}")
    print(f"physically-plausible-for-drone among top-5: {plausible}")

    proba = model.predict_proba(X)
    proba_reordered = proba[:, [classes_.index(c) for c in C.CLASSES]]
    pred = threshold_predict(proba_reordered, t)
    hit_idx = np.where((test[C.LABEL_COLUMN] == "drone").values & (pred == "drone"))[0]
    if len(hit_idx):
        i = int(hit_idx[0])
        row = X.iloc[[i]]
        sv_row = explainer.shap_values(row)
        if isinstance(sv_row, np.ndarray) and sv_row.ndim == 3:
            sv_row = [sv_row[:, :, k] for k in range(sv_row.shape[2])]
        ev = explainer.expected_value
        ev_drone = ev[drone_idx] if np.ndim(ev) else ev
        expl = shap.Explanation(values=sv_row[drone_idx][0], base_values=ev_drone,
                                data=row.values[0], feature_names=FEAT31_COLUMNS)
        plt.figure()
        shap.plots.waterfall(expl, show=False, max_display=15)
        plt.title(f"Why this detection is a drone (31-feature model, row {i})")
        plt.tight_layout()
        plt.savefig(C.PLOTS_DIR / "feat31_shap_waterfall_drone.png", dpi=150, bbox_inches="tight")
        plt.close("all")
        waterfall_info = {"test_row": i, "p_drone": float(proba[i, drone_idx])}
    else:
        waterfall_info = None
        print("no correctly-detected drone found for waterfall plot")

    (C.REPORTS_DIR / "feat31_shap_report.json").write_text(json.dumps({
        "drone_class_top5_features": top5,
        "physically_plausible_matches": plausible,
        "note": ("unlike the 11-feature schema's single named "
                 "micro_doppler_bandwidth, drone-discriminative signal here is "
                 "spread across several raw-statistic features -- reporting "
                 "which ones rank highest rather than requiring one specific name"),
        "waterfall_instance": waterfall_info,
    }, indent=2))
    print("SHAP artifacts -> feat31_shap_summary_drone.png, feat31_shap_summary_global.png, "
          "feat31_shap_waterfall_drone.png")


if __name__ == "__main__":
    main()
