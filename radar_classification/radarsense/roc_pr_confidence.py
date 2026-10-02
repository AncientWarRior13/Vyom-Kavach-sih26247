"""ROC/PR curves and per-class confidence histograms for the deployed model.

Uses the final trained model (artifacts/models/final_model.joblib) on the
real held-out test set (radar_test.parquet, n=7,800) -- the same set every
other headline number in this project is measured on. One-vs-rest ROC and
PR curves per class (4 classes: aircraft/anomaly/bird/drone), plus, per
class, the distribution of the model's predicted probability for that
class split by whether the sample truly belongs to it -- showing whether
confidence actually separates the class from everything else, not just a
single aggregate accuracy number.
"""

import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)

from radarsense import config as C


def main():
    model = joblib.load(C.MODELS_DIR / "final_model.joblib")
    test = pd.read_parquet(C.DATA_DIR / "radar_test.parquet")
    X, y = test[C.FEATURE_COLUMNS], test[C.LABEL_COLUMN]

    proba = model.predict_proba(X)
    classes_ = list(model.classes_)
    proba = proba[:, [classes_.index(c) for c in C.CLASSES]]

    roc_data, pr_data = {}, {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, cls in enumerate(C.CLASSES):
        y_bin = (y == cls).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, proba[:, i])
        auc = roc_auc_score(y_bin, proba[:, i])
        prec, rec, _ = precision_recall_curve(y_bin, proba[:, i])
        ap = average_precision_score(y_bin, proba[:, i])

        roc_data[cls] = {"auc": float(auc)}
        pr_data[cls] = {"average_precision": float(ap)}

        axes[0].plot(fpr, tpr, label=f"{cls} (AUC={auc:.3f})")
        axes[1].plot(rec, prec, label=f"{cls} (AP={ap:.3f})")

    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC curves (one-vs-rest), held-out test n=7,800")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall curves (one-vs-rest)")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(C.PLOTS_DIR / "roc_pr_curves.png", dpi=150)
    plt.close(fig)
    print("ROC AUC per class:", {k: round(v["auc"], 4) for k, v in roc_data.items()})
    print("Average Precision per class:", {k: round(v["average_precision"], 4) for k, v in pr_data.items()})

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    separation = {}
    for ax, (i, cls) in zip(axes.ravel(), enumerate(C.CLASSES)):
        is_cls = (y == cls).values
        p_true = proba[is_cls, i]
        p_rest = proba[~is_cls, i]
        ax.hist(p_rest, bins=30, alpha=0.6, label=f"true label != {cls}", color="tab:gray", density=True)
        ax.hist(p_true, bins=30, alpha=0.6, label=f"true label == {cls}", color="tab:blue", density=True)
        ax.set_title(f"P({cls}) by true label"); ax.set_xlabel(f"predicted P({cls})")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        separation[cls] = {
            "mean_p_when_true_class": float(p_true.mean()),
            "mean_p_when_other_class": float(p_rest.mean()),
        }
    fig.tight_layout()
    fig.savefig(C.PLOTS_DIR / "per_class_confidence_histograms.png", dpi=150)
    plt.close(fig)
    print("\nmean confidence when true vs. when other class:")
    for cls, s in separation.items():
        print(f"  {cls}: true={s['mean_p_when_true_class']:.3f} "
              f"other={s['mean_p_when_other_class']:.3f}")

    out = {
        "n_test": len(test),
        "roc_auc_per_class": roc_data,
        "average_precision_per_class": pr_data,
        "confidence_separation_per_class": separation,
    }
    (C.REPORTS_DIR / "roc_pr_confidence_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nresults -> {C.REPORTS_DIR / 'roc_pr_confidence_results.json'}")
    print(f"plots -> {C.PLOTS_DIR / 'roc_pr_curves.png'}, {C.PLOTS_DIR / 'per_class_confidence_histograms.png'}")


if __name__ == "__main__":
    main()
