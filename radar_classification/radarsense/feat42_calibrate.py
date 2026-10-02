"""Isotonic calibration for the 42-feature combined model. Same methodology
as feat31_calibrate.py / calibrate.py: per-class isotonic regression fit
on validation, evaluated on test with reliability diagrams + Brier + ECE.
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
from radarsense.calibrate import ece
from radarsense.feat42_simulator import FEAT42_COLUMNS


def main():
    model = joblib.load(C.MODELS_DIR / "feat42_model.joblib")
    val = pd.read_parquet(C.DATA_DIR / "feat42_val.parquet")
    test = pd.read_parquet(C.DATA_DIR / "feat42_test.parquet")
    classes_ = list(model.classes_)

    proba_val = model.predict_proba(val[FEAT42_COLUMNS])
    proba_test = model.predict_proba(test[FEAT42_COLUMNS])

    calibrators, report = {}, {}
    fig, axes = plt.subplots(1, len(C.CLASSES), figsize=(5 * len(C.CLASSES), 4.2))
    for cls in C.CLASSES:
        i = classes_.index(cls)
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
        ax = axes[C.CLASSES.index(cls)]
        for p, lbl in [(p_raw, "raw"), (p_cal, "isotonic")]:
            frac, mean_p = calibration_curve(yt, p, n_bins=10, strategy="quantile")
            ax.plot(mean_p, frac, "o-", label=lbl)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_title(f"{cls}\nECE {report[cls]['ece_raw']:.3f} -> "
                     f"{report[cls]['ece_calibrated']:.3f}")
        ax.set_xlabel("predicted probability")
        if C.CLASSES.index(cls) == 0:
            ax.set_ylabel("observed frequency")
        ax.legend()
    fig.suptitle("Reliability diagrams, 42-feature combined model (test set): raw vs isotonic-calibrated")
    fig.tight_layout()
    fig.savefig(C.PLOTS_DIR / "feat42_calibration_reliability.png", dpi=150)
    plt.close(fig)

    joblib.dump(calibrators, C.MODELS_DIR / "feat42_calibrators.joblib")
    (C.REPORTS_DIR / "feat42_calibration_report.json").write_text(json.dumps(report, indent=2))
    for cls, r in report.items():
        print(f"{cls:9s} Brier {r['brier_raw']:.4f}->{r['brier_calibrated']:.4f} "
              f"ECE {r['ece_raw']:.4f}->{r['ece_calibrated']:.4f}")
    print("calibrators + report saved")


if __name__ == "__main__":
    main()
