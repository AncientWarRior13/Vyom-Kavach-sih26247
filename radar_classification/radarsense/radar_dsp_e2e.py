"""End-to-end on REAL I/Q: raw 77GHz complex samples -> range-Doppler map ->
features extracted from THE MAP ITSELF (not the pre-computed bridge features)
-> drone-vs-rest classification. This closes the full radar chain on real
data: the classifier consumes the DSP output directly, so the whole pipeline
(range-Doppler, CFAR, feature extraction, classification) is demonstrated as
one connected system on real measurements, not as disconnected stages.

Features are derived purely from the Doppler power spectrum of each segment:
peak radial velocity, peak SNR, Doppler spectral centroid/bandwidth (the
micro-Doppler spread), CFAR detection count (micro-Doppler richness), and the
clutter-to-signal ratio. Multi-seed for significance.
"""

import json

import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from radarsense import config as C
from radarsense.radar_dsp import (RAW, N_SLOW, PRF, LAMBDA, range_doppler_map,
                                  notch_dc, os_cfar_1d, doppler_bin_to_velocity)
from radarsense.raddar_cnn import binary_metrics

DRONE_TYPES = {"D1", "D2", "D3", "D4", "D5", "D6"}
SEEDS = list(range(5))
SEGMENTS_PER_ROW = 60


def map_features(seg1280):
    """Features from the range-Doppler map only (the DSP output)."""
    power = notch_dc(range_doppler_map(seg1280))
    prof = power.sum(axis=0)
    freqs = np.fft.fftshift(np.fft.fftfreq(N_SLOW, d=1.0 / PRF))
    vel = freqs * LAMBDA / 2.0
    p = prof / (prof.sum() + 1e-12)
    peak = int(np.argmax(prof))
    centroid = float(np.sum(vel * p))
    bandwidth = float(np.sqrt(np.sum((vel - centroid) ** 2 * p)))
    n_det = int(sum(os_cfar_1d(power[r])[0].sum() for r in range(power.shape[0])))
    floor = np.median(prof[prof > 0]) if (prof > 0).any() else 1e-12
    return {
        "peak_velocity": float(vel[peak]),
        "peak_snr_db": 10 * np.log10(max(prof[peak], 1e-12) / max(floor, 1e-12)),
        "doppler_centroid": centroid,
        "doppler_bandwidth": bandwidth,
        "cfar_det_count": n_det,
        "clutter_ratio": float(floor / (prof[peak] + 1e-12)),
        "spectral_entropy": float(-np.sum(p * np.log(p + 1e-12))),
    }


def main():
    data = np.load(RAW, allow_pickle=True)
    rng = np.random.default_rng(C.SEED)

    print("--- extracting map-features from real I/Q, end-to-end ---")
    X, y = [], []
    for r in range(data.shape[0]):
        label = str(data[r, 0][0])
        is_drone = 1 if label in DRONE_TYPES else 0
        seg_cube = data[r, 1]
        idx = rng.choice(seg_cube.shape[1], size=min(SEGMENTS_PER_ROW, seg_cube.shape[1]), replace=False)
        for s in idx:
            X.append(map_features(seg_cube[:, s]))
            y.append(is_drone)
    feat_names = list(X[0].keys())
    Xarr = np.array([[f[k] for k in feat_names] for f in X])
    y = np.array(y)
    print(f"  {len(y)} real segments, {Xarr.shape[1]} map-features, drone frac {y.mean():.3f}")

    accs, recs, praucs = [], [], []
    for seed in SEEDS:
        Xtr, Xte, ytr, yte = train_test_split(Xarr, y, test_size=0.3, stratify=y, random_state=seed)
        sw = compute_sample_weight("balanced", ytr)
        clf = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1,
                                 random_seed=seed, verbose=0, allow_writing_files=False)
        clf.fit(Xtr, ytr, sample_weight=sw)
        p = clf.predict_proba(Xte)[:, 1]
        m = binary_metrics(yte, (p >= 0.5).astype(int), p)
        accs.append(m["accuracy"]); recs.append(m["recall_drone"]); praucs.append(m["pr_auc_drone"])

    print(f"\n=== end-to-end (raw I/Q -> DSP -> classify), {len(SEEDS)} seeds ===")
    print(f"  accuracy    : {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(f"  drone recall: {np.mean(recs):.4f} +/- {np.std(recs):.4f}")
    print(f"  drone PR-AUC: {np.mean(praucs):.4f} +/- {np.std(praucs):.4f}")

    out = {
        "pipeline": "raw 77GHz I/Q -> range-Doppler -> map-features -> CatBoost, end-to-end on real data",
        "n_segments": len(y), "n_features": Xarr.shape[1], "features": feat_names,
        "drone_fraction": float(y.mean()), "n_seeds": len(SEEDS),
        "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
        "recall_drone_mean": float(np.mean(recs)), "recall_drone_std": float(np.std(recs)),
        "pr_auc_mean": float(np.mean(praucs)), "pr_auc_std": float(np.std(praucs)),
    }
    (C.REPORTS_DIR / "radar_dsp_e2e_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nresults -> {C.REPORTS_DIR / 'radar_dsp_e2e_results.json'}")


if __name__ == "__main__":
    main()
