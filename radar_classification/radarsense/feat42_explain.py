"""SHAP + LIME explainability for the 42-feature combined model. Same
structure as feat31_explain.py / explain.py for the SHAP half: global
summary, per-detection waterfall, and a physics-plausibility check.

The actual question this exists to answer (left open in
FEAT42_COMBINED_EXPERIMENT.md): does the combined model genuinely draw on
BOTH feature sets, or does the accuracy jump come from one schema doing
all the work while the other rides along unused? Reports the physics-vs-
waveform-stat SHAP-importance split explicitly, not just a top-N list.

LIME is added as a SECOND, independent explanation method -- a genuinely
different algorithm (locally-weighted linear surrogate fit on perturbed
samples around one instance, vs. SHAP's exact game-theoretic tree
attribution). Two different methods pointing at the same top features is
real corroborating evidence the physics-vs-waveform-stat finding isn't a
SHAP-specific artifact; genuine disagreement would be worth knowing too,
not something to paper over. Run on (a) the same single detection as the
SHAP waterfall, for a direct one-to-one comparison, and (b) an aggregate
sample of drone detections, for a LIME-based top-feature ranking to set
against SHAP's global ranking.
"""

import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

from radarsense import config as C
from radarsense.feat42_simulator import FEAT42_COLUMNS
from radarsense.optimize import threshold_predict

LIME_AGGREGATE_N = 20
LIME_NUM_SAMPLES = 1000

PHYSICS_FEATURES = set(C.FEATURE_COLUMNS) - {"snr_db"}
WAVEFORM_STAT_FEATURES = set(FEAT42_COLUMNS) - set(C.FEATURE_COLUMNS)

PLAUSIBLE_DRONE_SIGNALS = {
    "micro_doppler_bandwidth", "micro_doppler_jitter", "velocity_variance", "acceleration",
    "velocity_std", "velocity_diff_var", "velocity_range", "velocity_energy",
    "amplitude_std", "amplitude_dominant_freq", "amplitude_diff_var",
    "amplitude_spectral_bandwidth", "energy_std", "energy_diff_var",
    "energy_dominant_freq", "energy_spectral_bandwidth",
}


def main():
    model = joblib.load(C.MODELS_DIR / "feat42_model.joblib")
    meta = json.loads((C.MODELS_DIR / "feat42_model_meta.json").read_text())
    t = meta["drone_threshold_deployed"]
    test = pd.read_parquet(C.DATA_DIR / "feat42_test.parquet")
    X = test[FEAT42_COLUMNS]
    classes_ = list(model.classes_)
    drone_idx = classes_.index("drone")

    bg = X.sample(min(2000, len(X)), random_state=C.SEED)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(bg)
    if isinstance(sv, np.ndarray) and sv.ndim == 3:
        sv = [sv[:, :, i] for i in range(sv.shape[2])]

    plt.figure()
    shap.summary_plot(sv[drone_idx], bg, feature_names=FEAT42_COLUMNS, show=False, max_display=41)
    plt.title("SHAP summary -- drone class (42-feature combined model)")
    plt.tight_layout()
    plt.savefig(C.PLOTS_DIR / "feat42_shap_summary_drone.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    mean_abs = np.mean([np.abs(s).mean(axis=0) for s in sv], axis=0)
    order = np.argsort(mean_abs)
    plt.figure(figsize=(8, 11))
    colors = ["#D65F5F" if FEAT42_COLUMNS[i] in PHYSICS_FEATURES else
             ("#4878CF" if FEAT42_COLUMNS[i] in WAVEFORM_STAT_FEATURES else "#6ACC64")
             for i in order]
    plt.barh([FEAT42_COLUMNS[i] for i in order], mean_abs[order], color=colors)
    plt.xlabel("mean |SHAP| (all classes)")
    plt.title("Global feature importance (42-feature combined model)\n"
             "red=physics (11-feature schema)  blue=waveform-stat (31-feature schema)  green=snr_db (shared)")
    plt.tight_layout()
    plt.savefig(C.PLOTS_DIR / "feat42_shap_summary_global.png", dpi=150)
    plt.close("all")

    physics_importance = float(sum(mean_abs[i] for i, f in enumerate(FEAT42_COLUMNS) if f in PHYSICS_FEATURES))
    waveform_importance = float(sum(mean_abs[i] for i, f in enumerate(FEAT42_COLUMNS) if f in WAVEFORM_STAT_FEATURES))
    snr_importance = float(mean_abs[FEAT42_COLUMNS.index("snr_db")])
    total = physics_importance + waveform_importance + snr_importance
    split = {
        "physics_11feature_share": physics_importance / total,
        "waveform_stat_31feature_share": waveform_importance / total,
        "snr_db_shared_share": snr_importance / total,
        "physics_feature_count": len(PHYSICS_FEATURES),
        "waveform_stat_feature_count": len(WAVEFORM_STAT_FEATURES),
    }
    print(f"feature-set SHAP-importance split: physics={split['physics_11feature_share']:.1%} "
          f"({split['physics_feature_count']} features)  "
          f"waveform-stat={split['waveform_stat_31feature_share']:.1%} "
          f"({split['waveform_stat_feature_count']} features)  "
          f"snr_db={split['snr_db_shared_share']:.1%}")

    drone_rank = np.argsort(np.abs(sv[drone_idx]).mean(axis=0))[::-1]
    top10 = [FEAT42_COLUMNS[i] for i in drone_rank[:10]]
    top10_source = [("physics" if f in PHYSICS_FEATURES else
                     ("waveform_stat" if f in WAVEFORM_STAT_FEATURES else "shared")) for f in top10]
    plausible = [f for f in top10 if f in PLAUSIBLE_DRONE_SIGNALS]
    n_physics_in_top10 = sum(1 for s in top10_source if s == "physics")
    n_waveform_in_top10 = sum(1 for s in top10_source if s == "waveform_stat")
    print(f"drone-class top-10 SHAP features: {list(zip(top10, top10_source))}")
    print(f"top-10 split: {n_physics_in_top10} physics / {n_waveform_in_top10} waveform-stat "
          f"/ {10 - n_physics_in_top10 - n_waveform_in_top10} shared")
    print(f"physically-plausible-for-drone among top-10: {plausible}")

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
                                data=row.values[0], feature_names=FEAT42_COLUMNS)
        plt.figure()
        shap.plots.waterfall(expl, show=False, max_display=15)
        plt.title(f"Why this detection is a drone (42-feature model, row {i})")
        plt.tight_layout()
        plt.savefig(C.PLOTS_DIR / "feat42_shap_waterfall_drone.png", dpi=150, bbox_inches="tight")
        plt.close("all")
        waterfall_info = {"test_row": i, "p_drone": float(proba[i, drone_idx])}
    else:
        waterfall_info = None
        print("no correctly-detected drone found for waterfall plot")

    print("\nrunning LIME (independent method, locally-weighted linear surrogate)...")
    train_bg = pd.read_parquet(C.DATA_DIR / "feat42_train.parquet").sample(
        min(3000, len(bg) * 3), random_state=C.SEED)[FEAT42_COLUMNS]
    lime_explainer = LimeTabularExplainer(
        train_bg.values, feature_names=FEAT42_COLUMNS, class_names=classes_,
        mode="classification", discretize_continuous=True, random_state=C.SEED)

    lime_single = None
    if waterfall_info is not None:
        row_vals = X.iloc[i].values
        exp = lime_explainer.explain_instance(
            row_vals, model.predict_proba, labels=(drone_idx,),
            num_features=15, num_samples=LIME_NUM_SAMPLES)
        fig = exp.as_pyplot_figure(label=drone_idx)
        fig.suptitle(f"LIME: why this detection is a drone (42-feature model, row {i})")
        fig.tight_layout()
        fig.savefig(C.PLOTS_DIR / "feat42_lime_single_drone.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        lime_single = {"test_row": i, "weights": exp.as_list(label=drone_idx)}
        print(f"LIME single-instance explanation (row {i}): {lime_single['weights'][:5]}...")

    drone_test_idx = test.index[test[C.LABEL_COLUMN] == "drone"].tolist()
    rng = np.random.default_rng(C.SEED)
    sample_idx = rng.choice(drone_test_idx, size=min(LIME_AGGREGATE_N, len(drone_test_idx)), replace=False)
    lime_feature_weight_sum = {f: 0.0 for f in FEAT42_COLUMNS}
    lime_feature_appearances = {f: 0 for f in FEAT42_COLUMNS}
    for idx in sample_idx:
        row_vals = X.loc[idx].values
        exp = lime_explainer.explain_instance(
            row_vals, model.predict_proba, labels=(drone_idx,),
            num_features=15, num_samples=LIME_NUM_SAMPLES)
        for feat_desc, weight in exp.as_list(label=drone_idx):
            matched = next((f for f in FEAT42_COLUMNS if f in feat_desc), None)
            if matched:
                lime_feature_weight_sum[matched] += abs(weight)
                lime_feature_appearances[matched] += 1
    print(f"LIME aggregate ({len(sample_idx)} drone detections) done")

    lime_ranked = sorted(lime_feature_weight_sum.items(), key=lambda kv: -kv[1])
    lime_top10 = [f for f, _ in lime_ranked[:10] if lime_feature_appearances[f] > 0][:10]
    shap_top10_set, lime_top10_set = set(top10), set(lime_top10)
    overlap = shap_top10_set & lime_top10_set
    print(f"LIME top-10 (aggregate |weight|): {lime_top10}")
    print(f"SHAP/LIME top-10 overlap: {len(overlap)}/10 features in common: {sorted(overlap)}")

    lime_physics_share = sum(lime_feature_weight_sum[f] for f in PHYSICS_FEATURES)
    lime_waveform_share = sum(lime_feature_weight_sum[f] for f in WAVEFORM_STAT_FEATURES)
    lime_snr_share = lime_feature_weight_sum["snr_db"]
    lime_total = lime_physics_share + lime_waveform_share + lime_snr_share
    lime_split = {
        "physics_11feature_share": lime_physics_share / lime_total if lime_total else None,
        "waveform_stat_31feature_share": lime_waveform_share / lime_total if lime_total else None,
        "snr_db_shared_share": lime_snr_share / lime_total if lime_total else None,
    }
    print(f"LIME aggregate feature-set split: physics={lime_split['physics_11feature_share']:.1%} "
          f"waveform-stat={lime_split['waveform_stat_31feature_share']:.1%} "
          f"(SHAP was physics={split['physics_11feature_share']:.1%} "
          f"waveform-stat={split['waveform_stat_31feature_share']:.1%})")

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_feats = [f for f, _ in lime_ranked[:15]]
    plot_vals = [lime_feature_weight_sum[f] for f in plot_feats]
    colors = ["#D65F5F" if f in PHYSICS_FEATURES else ("#4878CF" if f in WAVEFORM_STAT_FEATURES else "#6ACC64")
             for f in plot_feats]
    ax.barh(plot_feats[::-1], plot_vals[::-1], color=colors[::-1])
    ax.set_xlabel(f"summed |LIME weight| across {len(sample_idx)} drone detections")
    ax.set_title("LIME aggregate feature importance -- drone class (42-feature model)")
    fig.tight_layout()
    fig.savefig(C.PLOTS_DIR / "feat42_lime_aggregate.png", dpi=150)
    plt.close(fig)

    (C.REPORTS_DIR / "feat42_shap_report.json").write_text(json.dumps({
        "shap": {
            "feature_set_importance_split": split,
            "drone_class_top10_features": list(zip(top10, top10_source)),
            "physically_plausible_matches": plausible,
            "waterfall_instance": waterfall_info,
        },
        "lime": {
            "n_aggregate_samples": len(sample_idx),
            "num_samples_per_call": LIME_NUM_SAMPLES,
            "single_instance": lime_single,
            "aggregate_top10": lime_top10,
            "feature_set_importance_split": lime_split,
        },
        "shap_lime_agreement": {
            "top10_overlap_count": len(overlap),
            "top10_overlap_features": sorted(overlap),
        },
        "note": ("Answers whether the combined model genuinely uses both feature sets, via "
                 "TWO independent methods: SHAP (exact tree attribution) says physics="
                 f"{split['physics_11feature_share']:.1%} / waveform-stat="
                 f"{split['waveform_stat_31feature_share']:.1%} of importance despite "
                 f"{split['physics_feature_count']}/{split['waveform_stat_feature_count']} "
                 "feature counts; LIME (locally-weighted linear surrogate, aggregated over "
                 f"{len(sample_idx)} drone detections) says physics="
                 f"{lime_split['physics_11feature_share']:.1%} / waveform-stat="
                 f"{lime_split['waveform_stat_31feature_share']:.1%}. "
                 f"{len(overlap)}/10 top features agree between the two methods."),
    }, indent=2))
    print("SHAP artifacts -> feat42_shap_summary_drone.png, feat42_shap_summary_global.png, "
          "feat42_shap_waterfall_drone.png")
    print("LIME artifacts -> feat42_lime_single_drone.png, feat42_lime_aggregate.png")
    print("combined report -> feat42_shap_report.json")


if __name__ == "__main__":
    main()
