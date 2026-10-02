"""RAD-DAR (Kaggle iroldan/real-doppler-raddar-database) bridge + experiments.

Real 8.75 GHz FMCW radar (UPM RAD-DAR), 17,485 range-Doppler matrices
(11 range x 61 Doppler cells, dBm), classes: Drones / Cars / People,
organized in measurement sessions (folder per session).

Axis calibration (from Roldan et al., DopplerNet):
  11x61 cells = 9.658 m range window, 20 km/h Doppler span around detection
  -> 61 bins span 5.556 m/s, one bin = 0.0911 m/s. ASSUMPTION documented:
  "20 km/h around detection" read as total span; if it is +-20 km/h all
  velocities scale x2 (does not affect retrain experiment, which is
  scale-invariant).

Feature notes vs schema:
- absolute range NOT provided (window is target-relative) -> range_m constant;
  rcs_estimate is an uncalibrated z-scored power proxy (label-free scaling).
- consecutive files in a session are consecutive frames -> kinematics from
  file order, assumed ~0.1 s/frame (documented).
- class map: Drones->drone, Cars+People->anomaly. No bird class exists.

Experiments:
  A) zero-shot: frozen sim-trained model on extracted features (weak schema
     overlap -> expected poor; measured anyway, honestly).
  B) retrain same architecture with SESSION-AWARE GroupShuffleSplit
     (no session spans splits — stricter than segment-level splits).
"""

import glob
import json
import re

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_sample_weight

from radarsense import config as C
from radarsense.metrics import full_metrics
from radarsense.optimize import threshold_predict

RADDAR = C.ROOT / "external_data/raddar"
F_RADDAR = 8.75e9
V_SPAN = 20.0 / 3.6
V_AXIS = np.linspace(-V_SPAN / 2, V_SPAN / 2, 61)
HZ_PER_MS = 2 * F_RADDAR / C.C_LIGHT
DT_FRAME = 0.1

CLASS_MAP = {"Drones": "drone", "Cars": "anomaly", "People": "anomaly"}


def matrix_features(m):
    """m: (11 range, 61 doppler) in dBm -> spectral features."""
    P = (10 ** (m / 10)).mean(axis=0)
    dc = len(P) // 2
    notched = P.copy()
    notched[dc - 1:dc + 2] = 0.0
    floor = np.median(P)
    peak_i = int(np.argmax(notched))
    snr_raw = 10 * np.log10(max(notched[peak_i], 1e-30) / max(floor, 1e-30))
    v = V_AXIS[peak_i]

    sig = notched > floor * 2
    if sig.sum() >= 2:
        vs, ps = V_AXIS[sig], notched[sig]
        cen = np.average(vs, weights=ps)
        bw_ms = np.sqrt(np.average((vs - cen) ** 2, weights=ps))
    else:
        bw_ms = 0.0
    bw10 = bw_ms * HZ_PER_MS * (C.CARRIER_FREQ_HZ / F_RADDAR)

    cents = []
    for r in range(m.shape[0]):
        p = 10 ** (m[r] / 10)
        p[dc - 1:dc + 2] = 0.0
        mask = p > np.median(p) * 2
        if mask.sum() >= 2:
            cents.append(np.average(V_AXIS[mask], weights=p[mask]))
    jitter = float(np.clip(np.std(cents) / 10.0, 0, 1)) if len(cents) >= 2 else 0.0

    return {"doppler_velocity": float(np.clip(v, -50, 50)),
            "micro_doppler_bandwidth": float(np.clip(bw10, 0, 500)),
            "micro_doppler_jitter": jitter,
            "_snr_raw": float(snr_raw),
            "_power_db": float(10 * np.log10(max((10 ** (m / 10)).sum(), 1e-30)))}


def build_dataframe():
    rows = []
    for cls_dir, label in CLASS_MAP.items():
        for sess_path in sorted(glob.glob(str(RADDAR / cls_dir / "*"))):
            sess = f"{cls_dir}/{sess_path.split('/')[-1]}"
            files = sorted(glob.glob(f"{sess_path}/*.csv"),
                           key=lambda p: int(re.findall(r"(\d+)\.csv$", p)[0]))
            feats = []
            for fp in files:
                m = np.loadtxt(fp, delimiter=",")
                if m.shape != (11, 61):
                    continue
                f = matrix_features(m)
                f["label"] = label
                f["session"] = sess
                feats.append(f)
            v = np.array([f["doppler_velocity"] for f in feats])
            for i, f in enumerate(feats):
                lo, hi = max(0, i - 5), min(len(v), i + 6)
                w = v[lo:hi]
                f["velocity_variance"] = float(np.var(w)) if len(w) >= 3 else 0.0
                f["acceleration"] = float(np.clip(
                    np.polyfit(np.arange(len(w)) * DT_FRAME, w, 1)[0], -10, 10)) \
                    if len(w) >= 3 else 0.0
            rows.extend(feats)
    df = pd.DataFrame(rows)

    p = df["_power_db"]
    df["rcs_estimate"] = ((p - p.mean()) / max(p.std(), 1e-9) * 5 - 5).clip(-20, 20)
    s = df["_snr_raw"]
    scale = 5.0 / max(float(s.quantile(0.95) - s.median()), 1e-6)
    df["snr_db"] = (-15.0 + (s - s.median()) * scale).clip(-20, -10)
    df["doppler_freq_shift"] = 2 * df["doppler_velocity"] * C.CARRIER_FREQ_HZ / C.C_LIGHT
    df["range_m"] = 500.0
    df["azimuth_deg"] = 0.0
    df["altitude_m"] = 0.0
    return df


def main():
    df = build_dataframe()
    df.to_parquet(C.DATA_DIR / "real_raddar.parquet")
    print(f"{len(df)} samples | {df.label.value_counts().to_dict()} | "
          f"{df.session.nunique()} sessions")

    model = joblib.load(C.MODELS_DIR / "final_model.joblib")
    meta = json.loads((C.MODELS_DIR / "final_model_meta.json").read_text())
    proba = model.predict_proba(df[C.FEATURE_COLUMNS])
    pred = threshold_predict(proba, meta["drone_threshold_deployed"])
    a = full_metrics(df["label"], pred, proba)
    print(f"A zero-shot: drone recall {a['recall_drone']:.3f} "
          f"precision {a['precision_drone']:.3f} acc {a['accuracy']:.3f}")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=C.SEED)
    tr_i, rest_i = next(gss.split(df, groups=df.session))
    rest = df.iloc[rest_i]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=C.SEED)
    va_i, te_i = next(gss2.split(rest, groups=rest.session))
    tr, va, te = df.iloc[tr_i], rest.iloc[va_i], rest.iloc[te_i]

    clf = CatBoostClassifier(iterations=400, depth=6, learning_rate=0.1,
                             random_seed=C.SEED, verbose=0,
                             allow_writing_files=False)
    clf.fit(tr[C.FEATURE_COLUMNS], tr["label"],
            sample_weight=compute_sample_weight("balanced", tr["label"]))
    classes = list(clf.classes_)
    di = classes.index("drone")

    pv = clf.predict_proba(va[C.FEATURE_COLUMNS])[:, di]
    best_t, best_f2 = 0.5, -1
    yv = (va["label"] == "drone").astype(int).values
    for t in np.linspace(0.05, 0.95, 91):
        yh = (pv >= t).astype(int)
        tp = int(((yh == 1) & (yv == 1)).sum())
        fp = int(((yh == 1) & (yv == 0)).sum())
        fn = int(((yh == 0) & (yv == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        f2 = 5 * prec * rec / (4 * prec + rec) if prec + rec else 0
        if f2 > best_f2:
            best_f2, best_t = f2, t

    pt = clf.predict_proba(te[C.FEATURE_COLUMNS])[:, di]
    pred_b = np.where(pt >= best_t, "drone", "anomaly")
    yte = te["label"]
    b = {
        "recall_drone": float(((pred_b == "drone") & (yte == "drone")).sum()
                              / max((yte == "drone").sum(), 1)),
        "precision_drone": float(((pred_b == "drone") & (yte == "drone")).sum()
                                 / max((pred_b == "drone").sum(), 1)),
        "accuracy": float((pred_b == yte).mean()),
        "threshold": float(best_t),
        "split_sessions": {"train": int(tr.session.nunique()),
                           "val": int(va.session.nunique()),
                           "test": int(te.session.nunique())},
        "confusion_matrix": confusion_matrix(
            yte, pred_b, labels=["anomaly", "drone"]).tolist(),
    }
    print(f"B session-aware retrain: drone recall {b['recall_drone']:.3f} "
          f"precision {b['precision_drone']:.3f} acc {b['accuracy']:.3f} "
          f"(t={best_t:.2f})")

    joblib.dump(clf, C.MODELS_DIR / "real_raddar_model.joblib")
    out = {"dataset": "Kaggle iroldan/real-doppler-raddar-database "
                      "(UPM RAD-DAR 8.75 GHz FMCW, real measurements)",
           "n_samples": len(df),
           "assumptions": ["doppler span 20 km/h total", "0.1 s frame interval",
                           "rcs/power proxy uncalibrated", "range/azimuth/altitude unavailable"],
           "experiment_a_zero_shot": a,
           "experiment_b_session_aware_retrain": b}
    (C.REPORTS_DIR / "raddar_results.json").write_text(json.dumps(out, indent=2))
    print(f"results -> {C.REPORTS_DIR / 'raddar_results.json'}")


if __name__ == "__main__":
    main()
