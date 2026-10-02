"""Feature drift: does incoming data still resemble the training distribution?

Reference distribution = radar_train.parquet (what the deployed model was
trained on). For each comparison batch, per-feature drift is a z-score:

    drift_z = (batch_mean - train_mean) / train_std

Classification: |z| < 1 "in range" (green), 1 <= |z| < 2 "near boundary"
(yellow), |z| >= 2 "out of distribution" (red). This is the standard,
simplest input-drift diagnostic (compares first-moment shift in units of
training standard deviations) -- not a full distributional test (e.g.
Kolmogorov-Smirnov), which is noted as a limitation, not hidden.

Two kinds of comparison batch, both real data, nothing synthetic:
  1. radar_test.parquet -- SANITY CHECK. Same simulator, held-out split.
     Expected result: near-zero drift on every feature. If this shows large
     drift, something is wrong with the split, not with "real-world" data.
  2. Each of the four real-sensor bridges' extracted features
     (real_saab77 / real_raddar / real_tsms_cw / real_diat_usat parquet).
     This is the SAME sim-to-real gap already reported in
     artifacts/reports/{real_data,raddar,tsms,diat_usat}_results.json,
     viewed through a per-feature drift lens instead of an end-to-end
     accuracy lens -- both are honest readings of the same underlying gap,
     not two different claims.

Note (schema mismatch across bridges, documented per-bridge in their own
result JSONs): several real-sensor bridges hold some features constant
(e.g. azimuth_deg/altitude_m = 0, unmeasurable from that sensor) -- those
show up as extreme drift here (correctly) because the training data has
real variance on them and the real sensor has none. That is a genuine
finding, not a bug: it means those specific features are not measurable
from that real sensor, which is exactly the kind of thing this diagnostic
is supposed to surface.
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from radarsense import config as C

COMPARISON_DATASETS = {
    "radar_test (sanity check -- same simulator, held-out split)": "radar_test.parquet",
    "real_saab77 (77GHz FMCW)": "real_saab77.parquet",
    "real_raddar (8.75GHz FMCW)": "real_raddar.parquet",
    "real_tsms_cw (2.5GHz CW)": "real_tsms_cw.parquet",
    "real_diat_usat (X-band, tabular bridge)": "real_diat_usat.parquet",
}


def classify(z):
    az = abs(z)
    if az < 1:
        return "green"
    if az < 2:
        return "yellow"
    return "red"


def compute_drift(train_stats, batch, name):
    out = {}
    for feat in C.FEATURE_COLUMNS:
        mu, sigma = train_stats[feat]
        batch_mean = float(batch[feat].mean())
        z = (batch_mean - mu) / sigma if sigma > 1e-9 else (
            0.0 if abs(batch_mean - mu) < 1e-9 else np.sign(batch_mean - mu) * 999.0)
        out[feat] = {"train_mean": mu, "train_std": sigma,
                      "batch_mean": batch_mean, "drift_z": float(z),
                      "status": classify(z)}
    n_red = sum(1 for v in out.values() if v["status"] == "red")
    n_yellow = sum(1 for v in out.values() if v["status"] == "yellow")
    print(f"{name}: {n_red} features RED, {n_yellow} YELLOW, "
          f"{len(out) - n_red - n_yellow} green (n={len(batch)})")
    return out


def main():
    train = pd.read_parquet(C.DATA_DIR / "radar_train.parquet")
    train_stats = {f: (float(train[f].mean()), float(train[f].std())) for f in C.FEATURE_COLUMNS}

    all_results = {}
    for label, fname in COMPARISON_DATASETS.items():
        path = C.DATA_DIR / fname
        if not path.exists():
            print(f"skipping {label}: {fname} not found")
            continue
        batch = pd.read_parquet(path)
        all_results[label] = compute_drift(train_stats, batch, label)

    n_panels = len(all_results)
    fig, axes = plt.subplots(n_panels, 1, figsize=(9, 2.6 * n_panels))
    if n_panels == 1:
        axes = [axes]
    color_map = {"green": "tab:green", "yellow": "tab:orange", "red": "tab:red"}
    for ax, (label, feats) in zip(axes, all_results.items()):
        names = list(feats.keys())
        zs = [feats[f]["drift_z"] for f in names]
        zs_clipped = [max(-5, min(5, z)) for z in zs]
        colors = [color_map[feats[f]["status"]] for f in names]
        ax.barh(names, zs_clipped, color=colors)
        ax.axvline(0, color="black", lw=0.8)
        ax.axvline(1, color="gray", lw=0.5, ls="--"); ax.axvline(-1, color="gray", lw=0.5, ls="--")
        ax.axvline(2, color="gray", lw=0.5, ls=":"); ax.axvline(-2, color="gray", lw=0.5, ls=":")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("drift z-score (clipped to +-5 for display)", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(C.PLOTS_DIR / "feature_drift.png", dpi=150)
    plt.close(fig)

    out = {
        "method": "z-score of comparison-batch feature mean vs. training mean/std "
                  "(first-moment drift; NOT a full distributional test like KS -- "
                  "documented limitation, not hidden)",
        "thresholds": {"green": "|z| < 1", "yellow": "1 <= |z| < 2", "red": "|z| >= 2"},
        "results": all_results,
    }
    (C.REPORTS_DIR / "feature_drift_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nresults -> {C.REPORTS_DIR / 'feature_drift_results.json'}")
    print(f"plot -> {C.PLOTS_DIR / 'feature_drift.png'}")


if __name__ == "__main__":
    main()
