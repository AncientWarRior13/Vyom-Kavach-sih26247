"""Stage 4b: probability calibration for the final model.

The live demo shows confidence scores; uncalibrated scores are indefensible.
Isotonic calibration is fit per-class (one-vs-rest) on the VALIDATION set and
evaluated on test with reliability diagrams + Brier scores + ECE.
"""

import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

from radarsense import config as C


def ece(y_bin, p, n_bins=10):
    """Expected calibration error."""
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins[1:-1])
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.sum():
            total += mask.mean() * abs(y_bin[mask].mean() - p[mask].mean())
    return float(total)


def main():
    model = joblib.load(C.MODELS_DIR / "final_model.joblib")
    val = pd.read_parquet(C.DATA_DIR / "radar_val.parquet")
    test = pd.read_parquet(C.DATA_DIR / "radar_test.parquet")

    proba_val = model.predict_proba(val[C.FEATURE_COLUMNS])
    proba_test = model.predict_proba(test[C.FEATURE_COLUMNS])

    calibrators, report = {}, {}
    fig, axes = plt.subplots(1, len(C.CLASSES), figsize=(5 * len(C.CLASSES), 4.2))
    for i, cls in enumerate(C.CLASSES):
        yv = (val[C.LABEL_COLUMN] == cls).astype(int).values
        yt = (test[C.LABEL_COLUMN] == cls).astype(int).values
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        iso.fit(proba_val[:, i], yv)
        calibrators[cls] = iso
        p_raw, p_cal = proba_test[:, i], iso.predict(proba_test[:, i])
        report[cls] = {
            "brier_raw": float(brier_score_loss(yt, p_raw)),
            "brier_calibrated": float(brier_score_loss(yt, p_cal)),
            "ece_raw": ece(yt, p_raw),
            "ece_calibrated": ece(yt, p_cal),
        }
        for p, lbl in [(p_raw, "raw"), (p_cal, "isotonic")]:
            frac, mean_p = calibration_curve(yt, p, n_bins=10, strategy="quantile")
            axes[i].plot(mean_p, frac, "o-", label=lbl)
        axes[i].plot([0, 1], [0, 1], "k--", lw=0.8)
        axes[i].set_title(f"{cls}\nECE {report[cls]['ece_raw']:.3f} -> "
                          f"{report[cls]['ece_calibrated']:.3f}")
        axes[i].set_xlabel("predicted probability")
        if i == 0:
            axes[i].set_ylabel("observed frequency")
        axes[i].legend()
    fig.suptitle("Reliability diagrams (test set): raw vs isotonic-calibrated")
    fig.tight_layout()
    fig.savefig(C.PLOTS_DIR / "calibration_reliability.png", dpi=150)
    plt.close(fig)

    joblib.dump(calibrators, C.MODELS_DIR / "calibrators.joblib")
    (C.REPORTS_DIR / "calibration_report.json").write_text(json.dumps(report, indent=2))
    for cls, r in report.items():
        print(f"{cls:9s} Brier {r['brier_raw']:.4f}->{r['brier_calibrated']:.4f} "
              f"ECE {r['ece_raw']:.4f}->{r['ece_calibrated']:.4f}")
    print(f"calibrators + report saved")


if __name__ == "__main__":
    main()
