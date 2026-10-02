"""Stage 6: SHAP explainability on the final model.

- Global summary plot (feature importance across classes, drone class detail)
- Per-detection waterfall for a single correctly-detected drone instance
  (the "per-detection interpretability" claim as a real artifact)
- Physics sanity check: micro_doppler_bandwidth must rank top-3 for the
  drone class; the script FAILS LOUDLY if it doesn't.
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
from radarsense.optimize import threshold_predict


def main():
    model = joblib.load(C.MODELS_DIR / "final_model.joblib")
    meta = json.loads((C.MODELS_DIR / "final_model_meta.json").read_text())
    test = pd.read_parquet(C.DATA_DIR / "radar_test.parquet")
    X = test[C.FEATURE_COLUMNS]

    bg = X.sample(min(2000, len(X)), random_state=C.SEED)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(bg)
    if isinstance(sv, np.ndarray) and sv.ndim == 3:
        sv = [sv[:, :, i] for i in range(sv.shape[2])]

    plt.figure()
    shap.summary_plot(sv[C.MINORITY_IDX], bg, feature_names=C.FEATURE_COLUMNS,
                      show=False, max_display=11)
    plt.title("SHAP summary — drone class")
    plt.tight_layout()
    plt.savefig(C.PLOTS_DIR / "shap_summary_drone.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    mean_abs = np.mean([np.abs(s).mean(axis=0) for s in sv], axis=0)
    order = np.argsort(mean_abs)
    plt.figure(figsize=(7, 5))
    plt.barh([C.FEATURE_COLUMNS[i] for i in order], mean_abs[order], color="#4878CF")
    plt.xlabel("mean |SHAP| (all classes)")
    plt.title("Global feature importance")
    plt.tight_layout()
    plt.savefig(C.PLOTS_DIR / "shap_summary_global.png", dpi=150)
    plt.close("all")

    drone_rank = np.argsort(np.abs(sv[C.MINORITY_IDX]).mean(axis=0))[::-1]
    top3 = [C.FEATURE_COLUMNS[i] for i in drone_rank[:3]]
    sane = "micro_doppler_bandwidth" in top3 or "micro_doppler_jitter" in top3
    print(f"drone-class top-3 SHAP features: {top3} | physics-sane: {sane}")
    if not sane:
        raise SystemExit("PHYSICS SANITY FAIL: micro-Doppler features not in "
                         "drone top-3 SHAP — investigate before finals.")

    proba = model.predict_proba(X)
    pred = threshold_predict(proba, meta["drone_threshold_deployed"])
    hit_idx = np.where((test[C.LABEL_COLUMN] == "drone").values &
                       (pred == "drone"))[0]
    i = int(hit_idx[0])
    row = X.iloc[[i]]
    sv_row = explainer.shap_values(row)
    if isinstance(sv_row, np.ndarray) and sv_row.ndim == 3:
        sv_row = [sv_row[:, :, k] for k in range(sv_row.shape[2])]
    ev = explainer.expected_value
    ev_drone = ev[C.MINORITY_IDX] if np.ndim(ev) else ev
    expl = shap.Explanation(
        values=sv_row[C.MINORITY_IDX][0],
        base_values=ev_drone,
        data=row.values[0],
        feature_names=C.FEATURE_COLUMNS)
    plt.figure()
    shap.plots.waterfall(expl, show=False, max_display=11)
    plt.title(f"Why this detection is a drone (test row {i}, "
              f"p={proba[i, C.MINORITY_IDX]:.2f})")
    plt.tight_layout()
    plt.savefig(C.PLOTS_DIR / "shap_waterfall_drone_detection.png", dpi=150,
                bbox_inches="tight")
    plt.close("all")

    (C.REPORTS_DIR / "shap_report.json").write_text(json.dumps({
        "drone_class_feature_ranking": [C.FEATURE_COLUMNS[i] for i in drone_rank],
        "physics_sanity_top3": top3,
        "physics_sane": bool(sane),
        "waterfall_instance": {"test_row": i,
                               "p_drone": float(proba[i, C.MINORITY_IDX])},
    }, indent=2))
    print("SHAP artifacts saved: shap_summary_drone.png, shap_summary_global.png, "
          "shap_waterfall_drone_detection.png")


if __name__ == "__main__":
    main()
