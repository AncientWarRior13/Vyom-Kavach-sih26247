"""Robustness stress tests for the 31-feature model. Same structure as
radarsense/robustness.py (SNR-stratified recall + adversarial stress
tests), adapted to perturb the RAW WAVEFORMS (not just the extracted
statistics) and re-extract features after each perturbation -- this is the
correct level to perturb at, analogous to how the 11-feature schema's
stress tests perturb the underlying physics quantities, not the model's
inputs directly.

Stress test mechanics, each a genuine physical analog of the 11-feature
schema's tests:
  - RCS suppression (stealth coating): amplitude ~ sqrt(RCS_linear) by
    construction (feat31_simulator.py), so reducing RCS by X dB scales
    amplitude (and therefore energy = amplitude^2) by 10^(-X/20)
  - jamming: floors snr_db and adds proportional noise to all 3 waveforms,
    same mechanism as the simulator's own noise injection
  - micro-Doppler masking (shrouded rotors): smooths the amplitude/energy
    waveforms (moving-average), blunting the spectral bandwidth / dominant
    frequency features that carry the rotor-blade modulation signal --
    the direct analog of attenuating micro_doppler_bandwidth/jitter
"""

import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from radarsense import config as C
from radarsense.feat31_features import FEAT31_COLUMNS, extract_features
from radarsense.optimize import threshold_predict


def load_feat31():
    model = joblib.load(C.MODELS_DIR / "feat31_model.joblib")
    results = json.loads((C.REPORTS_DIR / "feat31_results.json").read_text())
    return model, results["threshold_fbeta2_optimal"]


def drone_recall(model, t, raw_df):
    feats = extract_features(raw_df)
    proba = model.predict_proba(feats[FEAT31_COLUMNS])
    proba = proba[:, [list(model.classes_).index(c) for c in C.CLASSES]]
    pred = threshold_predict(proba, t)
    return float((pred == C.MINORITY_CLASS).mean())


def main():
    model, t = load_feat31()
    test_raw = pd.read_parquet(C.DATA_DIR / "feat31_test.parquet")
    test_feats = extract_features(test_raw)

    proba = model.predict_proba(test_feats[FEAT31_COLUMNS])
    proba = proba[:, [list(model.classes_).index(c) for c in C.CLASSES]]
    pred = threshold_predict(proba, t)
    report = {"threshold_used": t}

    edges = np.arange(-25, -9.9, 2.5)
    buckets = pd.cut(test_raw["snr_db"], edges)
    snr_table = {}
    fig, ax = plt.subplots(figsize=(8, 5))
    for cls in C.CLASSES:
        recalls, centers = [], []
        for b in buckets.cat.categories:
            mask = (buckets == b) & (test_raw[C.LABEL_COLUMN] == cls)
            if mask.sum() < 10:
                continue
            recalls.append(float((pred[mask.values] == cls).mean()))
            centers.append(b.mid)
        snr_table[cls] = dict(zip([str(c) for c in centers], recalls))
        ax.plot(centers, recalls, "o-", label=cls, lw=2.5 if cls == "drone" else 1.2)
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("recall")
    ax.set_title("Per-class recall vs SNR, 31-feature model (tuned threshold)")
    ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(C.PLOTS_DIR / "feat31_robustness_snr_curves.png", dpi=150)
    plt.close(fig)
    report["snr_stratified_recall"] = snr_table

    drones = test_raw[test_raw[C.LABEL_COLUMN] == "drone"].reset_index(drop=True)
    rng = np.random.default_rng(C.SEED)
    base = drone_recall(model, t, drones)
    stress = {"unperturbed": base}

    for db in (3, 6, 9):
        z = drones.copy()
        scale = 10 ** (-db / 20.0)
        z["amplitude_waveform"] = z["amplitude_waveform"].apply(lambda w: w * scale)
        z["energy_waveform"] = z["amplitude_waveform"].apply(lambda w: w ** 2)
        stress[f"rcs_suppression_-{db}dBsm"] = drone_recall(model, t, z)

    for floor in (-20, -22, -25):
        z = drones.copy()
        old_snr = z["snr_db"].values
        new_snr = np.minimum(old_snr, floor)
        noise_add = np.clip((old_snr - new_snr) / 15.0, 0, 1)
        z["snr_db"] = new_snr
        for col, scale in [("amplitude_waveform", 0.25), ("velocity_waveform", 4.0),
                           ("energy_waveform", 0.35)]:
            wf = np.stack(z[col].values)
            wf_scale = (np.abs(wf).mean(axis=1, keepdims=True) + 1e-6) * scale
            wf = wf + noise_add[:, None] * wf_scale * rng.standard_normal(wf.shape)
            z[col] = list(wf)
        stress[f"jamming_snr_floor_{floor}dB"] = drone_recall(model, t, z)

    for window in (3, 7, 15):
        z = drones.copy()
        kernel = np.ones(window) / window
        def smooth(w):
            return np.convolve(w, kernel, mode="same")
        z["amplitude_waveform"] = z["amplitude_waveform"].apply(smooth)
        z["energy_waveform"] = z["amplitude_waveform"].apply(lambda w: w ** 2)
        stress[f"micro_doppler_masking_smooth_w{window}"] = drone_recall(model, t, z)

    report["drone_stress_recall"] = stress

    fig, ax = plt.subplots(figsize=(9, 5))
    names = list(stress)
    ax.barh(names[::-1], [stress[k] for k in names[::-1]], color="#4878CF")
    ax.axvline(base, color="k", ls="--", lw=0.8, label=f"unperturbed {base:.2f}")
    ax.set_xlabel("drone recall under perturbation")
    ax.set_title("Adversarial stress tests, 31-feature model (drone test samples)")
    ax.legend(); fig.tight_layout()
    fig.savefig(C.PLOTS_DIR / "feat31_robustness_stress_tests.png", dpi=150)
    plt.close(fig)

    (C.REPORTS_DIR / "feat31_robustness_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(stress, indent=2))
    print(f"report -> {C.REPORTS_DIR / 'feat31_robustness_report.json'}")


if __name__ == "__main__":
    main()
