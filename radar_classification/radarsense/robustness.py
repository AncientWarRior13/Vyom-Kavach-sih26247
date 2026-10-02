"""Stage 4c: robustness — SNR-stratified performance + adversarial stress tests.

1. SNR-stratified recall curves: per-class recall in 2 dB SNR buckets on the
   untouched test set (snr_db is in the schema precisely for this).
2. Stress tests on the drone test samples (feature-space perturbations that
   mimic countermeasures; the model is frozen):
     - RCS suppression: rcs_estimate reduced by 3/6/9 dBsm (stealth coating)
     - jamming: snr_db floored toward -20 plus proportional feature noise
     - micro-Doppler masking: bandwidth/jitter attenuated (shrouded rotors)
"""

import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from radarsense import config as C
from radarsense.optimize import threshold_predict


def load_final():
    model = joblib.load(C.MODELS_DIR / "final_model.joblib")
    meta = json.loads((C.MODELS_DIR / "final_model_meta.json").read_text())
    return model, meta["drone_threshold_deployed"]


def drone_recall(model, t, X, is_drone):
    proba = model.predict_proba(X[C.FEATURE_COLUMNS])
    pred = threshold_predict(proba, t)
    return float((pred[is_drone] == C.MINORITY_CLASS).mean())


def main():
    model, t = load_final()
    test = pd.read_parquet(C.DATA_DIR / "radar_test.parquet")
    proba = model.predict_proba(test[C.FEATURE_COLUMNS])
    pred = threshold_predict(proba, t)
    report = {"threshold_used": t}

    edges = np.arange(-20, -9.9, 2.0)
    buckets = pd.cut(test["snr_db"], edges)
    snr_table = {}
    fig, ax = plt.subplots(figsize=(8, 5))
    for cls in C.CLASSES:
        recalls, centers = [], []
        for b in buckets.cat.categories:
            mask = (buckets == b) & (test[C.LABEL_COLUMN] == cls)
            if mask.sum() < 10:
                continue
            recalls.append(float((pred[mask.values] == cls).mean()))
            centers.append(b.mid)
        snr_table[cls] = dict(zip([str(c) for c in centers], recalls))
        ax.plot(centers, recalls, "o-", label=cls,
                lw=2.5 if cls == "drone" else 1.2)
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("recall")
    ax.set_title("Per-class recall vs SNR (held-out test, cost-optimal threshold)")
    ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(C.PLOTS_DIR / "robustness_snr_curves.png", dpi=150)
    plt.close(fig)
    report["snr_stratified_recall"] = snr_table

    drones = test[test[C.LABEL_COLUMN] == C.MINORITY_CLASS].reset_index(drop=True)
    is_drone = np.ones(len(drones), dtype=bool)
    rng = np.random.default_rng(C.SEED)
    base = drone_recall(model, t, drones, is_drone)
    stress = {"unperturbed": base}

    for db in (3, 6, 9):
        z = drones.copy()
        z["rcs_estimate"] = np.clip(z["rcs_estimate"] - db, -20, 20)
        stress[f"rcs_suppression_-{db}dBsm"] = drone_recall(model, t, z, is_drone)

    for floor in (-16, -18, -20):
        z = drones.copy()
        squeeze = np.clip(z["snr_db"] - floor, 0, None) / 10.0
        z["snr_db"] = np.minimum(z["snr_db"], floor)
        for col, scale in [("doppler_velocity", 4), ("rcs_estimate", 3),
                           ("micro_doppler_bandwidth", 60),
                           ("micro_doppler_jitter", 0.12)]:
            z[col] = z[col] + squeeze * scale * rng.standard_normal(len(z))
        z["micro_doppler_jitter"] = z["micro_doppler_jitter"].clip(0, 1)
        z["micro_doppler_bandwidth"] = z["micro_doppler_bandwidth"].clip(0, 500)
        stress[f"jamming_snr_floor_{floor}dB"] = drone_recall(model, t, z, is_drone)

    for att in (0.7, 0.5, 0.3):
        z = drones.copy()
        z["micro_doppler_bandwidth"] *= att
        z["micro_doppler_jitter"] *= att
        stress[f"micro_doppler_masking_x{att}"] = drone_recall(model, t, z, is_drone)

    report["drone_stress_recall"] = stress

    fig, ax = plt.subplots(figsize=(9, 5))
    names = list(stress)
    ax.barh(names[::-1], [stress[k] for k in names[::-1]], color="#4878CF")
    ax.axvline(base, color="k", ls="--", lw=0.8, label=f"unperturbed {base:.2f}")
    ax.set_xlabel("drone recall under perturbation")
    ax.set_title("Adversarial stress tests (frozen model, drone test samples)")
    ax.legend(); fig.tight_layout()
    fig.savefig(C.PLOTS_DIR / "robustness_stress_tests.png", dpi=150)
    plt.close(fig)

    (C.REPORTS_DIR / "robustness_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(stress, indent=2))
    print(f"report -> {C.REPORTS_DIR / 'robustness_report.json'}")


if __name__ == "__main__":
    main()
