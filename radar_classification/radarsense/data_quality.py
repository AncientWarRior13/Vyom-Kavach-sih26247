"""Data quality checks (spec section 2.3) + plots + QC report JSON."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from radarsense import config as C

KEY_FEATURES = ["doppler_velocity", "rcs_estimate", "micro_doppler_bandwidth",
                "micro_doppler_jitter", "velocity_variance", "snr_db"]


def main():
    full = pd.read_parquet(C.DATA_DIR / "radar_full.parquet")
    train = pd.read_parquet(C.DATA_DIR / "radar_train.parquet")
    val = pd.read_parquet(C.DATA_DIR / "radar_val.parquet")
    test = pd.read_parquet(C.DATA_DIR / "radar_test.parquet")

    report = {}

    counts = full[C.LABEL_COLUMN].value_counts()
    maj = counts.drop(C.MINORITY_CLASS).max()
    ratio = maj / counts[C.MINORITY_CLASS]
    report["class_counts"] = counts.to_dict()
    report["imbalance_ratio_majority_to_drone"] = round(float(ratio), 2)
    report["imbalance_in_spec_band_15_25"] = bool(15 <= ratio <= 25)

    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot.bar(ax=ax, color="#4878CF")
    ax.set_title(f"Class distribution (majority:drone = {ratio:.1f}:1)")
    ax.set_ylabel("samples")
    fig.tight_layout()
    fig.savefig(C.PLOTS_DIR / "qc_class_distribution.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, feat in zip(axes.flat, KEY_FEATURES):
        for cls in C.CLASSES:
            vals = full.loc[full[C.LABEL_COLUMN] == cls, feat]
            ax.hist(vals, bins=60, density=True, alpha=0.45, label=cls)
        ax.set_title(feat)
    axes.flat[0].legend()
    fig.suptitle("Per-class feature distributions")
    fig.tight_layout()
    fig.savefig(C.PLOTS_DIR / "qc_feature_distributions.png", dpi=150)
    plt.close(fig)

    for feat in ["micro_doppler_bandwidth", "micro_doppler_jitter", "rcs_estimate"]:
        d = full.loc[full[C.LABEL_COLUMN] == "drone", feat]
        b = full.loc[full[C.LABEL_COLUMN] == "bird", feat]
        pooled = np.sqrt((d.var() + b.var()) / 2)
        report[f"cohens_d_drone_vs_bird__{feat}"] = round(float(abs(d.mean() - b.mean()) / pooled), 3)

    snr = full["snr_db"]
    report["snr_min"] = round(float(snr.min()), 2)
    report["snr_max"] = round(float(snr.max()), 2)
    report["snr_in_band"] = bool(snr.between(*C.SNR_RANGE_DB).all())
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(snr, bins=80, color="#4878CF")
    ax.set_title("Realized SNR distribution")
    ax.set_xlabel("SNR (dB)")
    fig.tight_layout()
    fig.savefig(C.PLOTS_DIR / "qc_snr_distribution.png", dpi=150)
    plt.close(fig)

    dup_total = int(full.duplicated().sum())
    tr = set(map(tuple, train[C.FEATURE_COLUMNS].round(9).values[:, :4]))
    te = set(map(tuple, test[C.FEATURE_COLUMNS].round(9).values[:, :4]))
    va = set(map(tuple, val[C.FEATURE_COLUMNS].round(9).values[:, :4]))
    report["duplicate_rows_full"] = dup_total
    report["train_test_overlap"] = len(tr & te)
    report["train_val_overlap"] = len(tr & va)

    for name, df in [("train", train), ("val", val), ("test", test)]:
        report[f"{name}_class_frac"] = {
            k: round(v, 4) for k, v in
            (df[C.LABEL_COLUMN].value_counts(normalize=True)).to_dict().items()}

    report["schema_columns_ok"] = list(full.columns) == C.FEATURE_COLUMNS + [C.LABEL_COLUMN]

    out = C.REPORTS_DIR / "data_quality_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nQC report -> {out}")


if __name__ == "__main__":
    main()
