"""Real-data experiments on the SAAB 77 GHz dataset (Zenodo 5896641).

Experiment A — domain-shift validation (the honest one):
    The simulator-trained FROZEN model runs unchanged on real extracted
    features. No refitting, no threshold retuning. Whatever the number is,
    that's the measured sim-to-real gap. Note: this dataset has no aircraft;
    any 'aircraft' prediction counts as an error against 3 real classes.

Experiment B — train on real data:
    Same architecture (CatBoost + balanced class weights) retrained from
    scratch on the dataset's OFFICIAL train split (1), threshold F2-tuned on
    the official val split (2), reported once on the official test split (3).
    3-class task: drone / bird / anomaly (humans + corner reflector).

Leakage note: the official splits are segment-level from [1]; segments of one
measurement can span splits, so B's numbers are comparable to the published
benchmark but not a strict independent-flight test — stated in the report.
"""

import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

from radarsense import config as C
from radarsense.metrics import full_metrics
from radarsense.optimize import threshold_predict, tune_threshold

REAL_CLASSES = ["anomaly", "bird", "drone"]


def experiment_a(df):
    model = joblib.load(C.MODELS_DIR / "final_model.joblib")
    meta = json.loads((C.MODELS_DIR / "final_model_meta.json").read_text())
    X = df[C.FEATURE_COLUMNS]
    y = df[C.LABEL_COLUMN]
    proba = model.predict_proba(X)
    pred = threshold_predict(proba, meta["drone_threshold_deployed"])
    m = full_metrics(y, pred, proba)
    cm = confusion_matrix(y, pred, labels=C.CLASSES).tolist()
    print("=== A: frozen sim-trained model on real data ===")
    print(f"drone recall {m['recall_drone']:.3f} precision {m['precision_drone']:.3f} "
          f"| bird recall {m['recall_bird']:.3f} | anomaly recall {m['recall_anomaly']:.3f} "
          f"| accuracy {m['accuracy']:.3f}")
    return {"metrics": m, "confusion_matrix_rows_true_cols_pred": cm,
            "labels": C.CLASSES,
            "note": ("Zero-shot sim-to-real transfer, frozen model + frozen "
                     "threshold; aircraft column = misfires on a class absent "
                     "from the real data.")}


def experiment_b(df):
    tr = df[df.split == 1]
    va = df[df.split == 2]
    te = df[df.split == 3]
    Xtr, ytr = tr[C.FEATURE_COLUMNS], tr[C.LABEL_COLUMN]

    model = CatBoostClassifier(iterations=400, depth=6, learning_rate=0.1,
                               random_seed=C.SEED, verbose=0,
                               allow_writing_files=False)
    model.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))

    classes = list(model.classes_)
    di = classes.index("drone")

    def pred_with_threshold(X, t):
        proba = model.predict_proba(X)
        out = []
        others = [i for i in range(len(classes)) if i != di]
        for row in proba:
            if row[di] >= t:
                out.append("drone")
            else:
                out.append(classes[others[int(np.argmax(row[others]))]])
        return np.array(out), proba

    proba_val = model.predict_proba(va[C.FEATURE_COLUMNS])
    pv = np.zeros((len(proba_val), len(C.CLASSES)))
    pv[:, C.MINORITY_IDX] = proba_val[:, di]
    t_f2, _, _ = tune_threshold(pv, va[C.LABEL_COLUMN])

    pred, proba = pred_with_threshold(te[C.FEATURE_COLUMNS], t_f2)
    yte = te[C.LABEL_COLUMN]
    prf = {}
    for cls in REAL_CLASSES:
        tp = ((pred == cls) & (yte == cls)).sum()
        prf[f"recall_{cls}"] = float(tp / max((yte == cls).sum(), 1))
        prf[f"precision_{cls}"] = float(tp / max((pred == cls).sum(), 1))
    acc = float((pred == yte).mean())
    cm = confusion_matrix(yte, pred, labels=REAL_CLASSES).tolist()
    print("=== B: retrained on real train split, official test split ===")
    print(f"threshold(F2,val)={t_f2:.2f} | accuracy {acc:.3f}")
    for cls in REAL_CLASSES:
        print(f"  {cls:8s} recall {prf['recall_'+cls]:.3f} "
              f"precision {prf['precision_'+cls]:.3f}")
    joblib.dump(model, C.MODELS_DIR / "real_saab77_model.joblib")
    return {"threshold_f2_val": float(t_f2), "accuracy": acc, **prf,
            "confusion_matrix_rows_true_cols_pred": cm, "labels": REAL_CLASSES,
            "split_sizes": {"train": len(tr), "val": len(va), "test": len(te)}}


def main():
    df = pd.read_parquet(C.DATA_DIR / "real_saab77.parquet")
    print(f"{len(df)} real segments | class counts: "
          f"{df[C.LABEL_COLUMN].value_counts().to_dict()}")

    res_a = experiment_a(df)
    res_b = experiment_b(df)

    opt = json.loads((C.REPORTS_DIR / "optimization_results.json").read_text())
    sim = opt["final_test_metrics"]["fbeta2_threshold"]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    bars = {
        "sim test\n(frozen model)": sim["recall_drone"],
        "REAL zero-shot\n(frozen model)": res_a["metrics"]["recall_drone"],
        "REAL retrained\n(official splits)": res_b["recall_drone"],
    }
    ax.bar(bars.keys(), bars.values(), color=["#4878CF", "#D65F5F", "#6ACC65"])
    for i, v in enumerate(bars.values()):
        ax.text(i, v + 0.02, f"{v:.1%}", ha="center", fontweight="bold")
    ax.set_ylabel("drone recall")
    ax.set_ylim(0, 1.12)
    ax.set_title("Sim-to-real: drone recall (SAAB 77 GHz real radar, n=%d)"
                 % len(df))
    fig.tight_layout()
    fig.savefig(C.PLOTS_DIR / "real_data_sim_to_real.png", dpi=150)
    plt.close(fig)

    out = {"dataset": "Zenodo 5896641 SAAB SIRS 77GHz FMCW (real measurements)",
           "experiment_a_zero_shot": res_a, "experiment_b_retrained": res_b}
    (C.REPORTS_DIR / "real_data_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nresults -> {C.REPORTS_DIR / 'real_data_results.json'}")


if __name__ == "__main__":
    main()
