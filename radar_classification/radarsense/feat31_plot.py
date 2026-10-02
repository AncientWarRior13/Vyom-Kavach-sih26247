"""Plots accuracy/drone-recall vs. SNR for the 31-feature model (from
feat31_results.json's SNR-stratified slices) with the frozen 11-feature
model's out-of-distribution -25dB point overlaid for reference."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from radarsense import config as C


def main():
    feat31 = json.loads((C.REPORTS_DIR / "feat31_results.json").read_text())
    comp = json.loads((C.REPORTS_DIR / "feat31_vs_11feature_comparison.json").read_text())

    slices = feat31["snr_stratified"]
    midpoints, accs, recalls = [], [], []
    for key, m in slices.items():
        lo, hi = key.split("_to_")
        midpoints.append((float(lo) + float(hi)) / 2)
        accs.append(m["accuracy"])
        recalls.append(m["recall_drone"])
    order = sorted(range(len(midpoints)), key=lambda i: midpoints[i])
    midpoints = [midpoints[i] for i in order]
    accs = [accs[i] for i in order]
    recalls = [recalls[i] for i in order]

    m11 = comp["model_11feature_frozen_ood"]
    m31_deep = comp["model_31feature_trained_for_regime"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(midpoints, accs, "o-", label="31-feature model", color="tab:purple")
    axes[0].scatter([-24], [m11["accuracy"]], color="tab:blue", s=80, zorder=5,
                    label="11-feature FROZEN (out-of-distribution)")
    axes[0].scatter([-24], [m31_deep["accuracy"]], color="tab:red", s=80, zorder=5, marker="x",
                    label="31-feature (deep low-SNR slice)")
    axes[0].set_xlabel("SNR (dB)"); axes[0].set_ylabel("accuracy")
    axes[0].set_title("Accuracy vs SNR"); axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)

    axes[1].plot(midpoints, recalls, "o-", label="31-feature model", color="tab:purple")
    axes[1].scatter([-24], [m11["recall_drone"]], color="tab:blue", s=80, zorder=5,
                    label="11-feature FROZEN (out-of-distribution)")
    axes[1].scatter([-24], [m31_deep["recall_drone"]], color="tab:red", s=80, zorder=5, marker="x",
                    label="31-feature (deep low-SNR slice)")
    axes[1].set_xlabel("SNR (dB)"); axes[1].set_ylabel("drone recall")
    axes[1].set_title("Drone recall vs SNR"); axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(C.PLOTS_DIR / "feat31_snr_sweep_comparison.png", dpi=150)
    plt.close(fig)
    print(f"plot -> {C.PLOTS_DIR / 'feat31_snr_sweep_comparison.png'}")


if __name__ == "__main__":
    main()
