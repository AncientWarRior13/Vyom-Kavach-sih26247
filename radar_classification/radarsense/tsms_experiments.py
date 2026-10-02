"""TSMS-Drone CW radar (figshare 10.25452/figshare.plus.30027313) bridge + experiments.

Real S-band CW radar (2.495 GHz SDR, Han & Jung 2026): 16,384-sample I/Q frames
(0.25 s @ 65,536 Hz) with a 1,024 Hz baseband tone; micro-Doppler sidebands
around the tone. Targets: 4 real drones (Inspire 2, Matrice 30, Mavic 2 Pro,
Phantom 4 Pro) + tetrahedral corner reflector, at known 2-30 m distances.
Subset used: 30 frames per (distance, target), extracted from CW Radar.7z.

Schema mapping notes:
- doppler_velocity: spectral peak offset from the 1,024 Hz tone -> m/s at
  2.495 GHz (drones hover: near zero, correct).
- micro_doppler_bandwidth: power-weighted spread around the tone, carrier-
  scaled x(10/2.495) to the schema's 10 GHz.
- rcs_estimate: 10log10(power) + 40log10(R) calibrated so the corner
  reflector's median = +20 dBsm (same radiometric trick as the SAAB bridge).
- range_m: REAL (from folder name), floored at the schema's 50 m minimum
  (actual 2-30 m; documented).
- kinematics from consecutive frames per (distance, target).
- classes: drones -> drone, corner reflector -> anomaly.

Experiments:
  A) zero-shot frozen sim-trained model (drone recall).
  B) range-generalization: train on even distances, test on odd distances
     (drone vs anomaly) — does the classifier survive unseen ranges?
"""

import glob
import json
import re

import h5py
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.utils.class_weight import compute_sample_weight

from radarsense import config as C
from radarsense.metrics import full_metrics
from radarsense.optimize import threshold_predict

SUBSET = C.ROOT / "external_data/tsms_drone/subset"
FS = 65536.0
TONE = 1024.0
F_CW = 2.495e9
V_PER_HZ = C.C_LIGHT / (2 * F_CW)


def frame_features(x):
    x = x - x.mean()
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), d=1 / FS)
    band = (freqs > TONE - 600) & (freqs < TONE + 600)
    f, P = freqs[band], spec[band]
    tone_i = int(np.argmin(np.abs(f - TONE)))
    notched = P.copy()
    notched[max(0, tone_i - 2):tone_i + 3] = 0.0

    floor = np.median(P)
    peak_i = int(np.argmax(notched))
    snr_raw = 10 * np.log10(max(notched[peak_i], 1e-30) / max(floor, 1e-30))
    v = (f[peak_i] - TONE) * V_PER_HZ

    sig = notched > floor * 2
    if sig.sum() >= 2:
        fs_, ps = f[sig] - TONE, notched[sig]
        cen = np.average(fs_, weights=ps)
        bw = np.sqrt(np.average((fs_ - cen) ** 2, weights=ps))
    else:
        bw = 0.0
    bw10 = bw * (C.CARRIER_FREQ_HZ / F_CW)

    cents = []
    for k in range(4):
        seg = x[k * 4096:(k + 1) * 4096]
        sp = np.abs(np.fft.rfft(seg * np.hanning(4096))) ** 2
        fq = np.fft.rfftfreq(4096, d=1 / FS)
        b = (fq > TONE - 600) & (fq < TONE + 600)
        fb, pb = fq[b], sp[b]
        ti = int(np.argmin(np.abs(fb - TONE)))
        pb[max(0, ti - 1):ti + 2] = 0.0
        m = pb > np.median(pb) * 2
        if m.sum() >= 2:
            cents.append(np.average(fb[m] - TONE, weights=pb[m]) * V_PER_HZ)
    jitter = float(np.clip(np.std(cents) / 10.0, 0, 1)) if len(cents) >= 2 else 0.0

    return {"doppler_velocity": float(np.clip(v, -50, 50)),
            "micro_doppler_bandwidth": float(np.clip(bw10, 0, 500)),
            "micro_doppler_jitter": jitter,
            "_snr_raw": float(snr_raw),
            "_power_db": float(10 * np.log10(max(P.sum(), 1e-30)))}


def load_iq(fp):
    try:
        with h5py.File(fp) as h:
            d = h["data"][:]
        return (d["real"] + 1j * d["imag"]).ravel().astype(np.complex128)
    except OSError:
        import scipy.io as sio
        m = sio.loadmat(fp)
        for k, v in m.items():
            if not k.startswith("__"):
                return np.asarray(v).ravel().astype(np.complex128)
    return None


def build_dataframe():
    rows, skipped = [], 0
    for fp in sorted(glob.glob(str(SUBSET / "*/*/Raw File/*.mat"))):
        m = re.search(r"subset/(\d+)m/([^/]+)/", fp)
        dist, target = float(m.group(1)), m.group(2)
        try:
            x = load_iq(fp)
        except Exception:
            x = None
        if x is None or len(x) < 8192:
            skipped += 1
            continue
        f = frame_features(x)
        f["distance_m"] = dist
        f["target"] = target
        f["label"] = "anomaly" if target == "Corner Reflector" else "drone"
        rows.append(f)
    if skipped:
        print(f"skipped {skipped} unreadable/short files")
    df = pd.DataFrame(rows)

    for (_, _), g in df.groupby(["distance_m", "target"]):
        v = g["doppler_velocity"].values
        idx = g.index.to_numpy()
        for j, i in enumerate(idx):
            lo, hi = max(0, j - 3), min(len(v), j + 4)
            w = v[lo:hi]
            df.loc[i, "velocity_variance"] = float(np.var(w))
            df.loc[i, "acceleration"] = float(np.clip(
                np.polyfit(np.arange(len(w)) * 0.25, w, 1)[0], -10, 10)) \
                if len(w) >= 3 else 0.0

    cr = df[df.target == "Corner Reflector"]
    pm = df["_power_db"] + 40 * np.log10(df["distance_m"])
    offset = float((cr["_power_db"] + 40 * np.log10(cr["distance_m"])).median() - 20.0)
    df["rcs_estimate"] = (pm - offset).clip(-20, 20)
    s = df["_snr_raw"]
    scale = 5.0 / max(float(s.quantile(0.95) - s.median()), 1e-6)
    df["snr_db"] = (-15.0 + (s - s.median()) * scale).clip(-20, -10)
    df["doppler_freq_shift"] = 2 * df["doppler_velocity"] * C.CARRIER_FREQ_HZ / C.C_LIGHT
    df["range_m"] = df["distance_m"].clip(lower=50.0)
    df["azimuth_deg"] = 0.0
    df["altitude_m"] = 1.5
    return df


def main():
    df = build_dataframe()
    df.to_parquet(C.DATA_DIR / "real_tsms_cw.parquet")
    print(f"{len(df)} frames | {df.label.value_counts().to_dict()} | "
          f"distances {sorted(df.distance_m.unique())}")

    model = joblib.load(C.MODELS_DIR / "final_model.joblib")
    meta = json.loads((C.MODELS_DIR / "final_model_meta.json").read_text())
    proba = model.predict_proba(df[C.FEATURE_COLUMNS])
    pred = threshold_predict(proba, meta["drone_threshold_deployed"])
    a = full_metrics(df["label"], pred, proba)
    print(f"A zero-shot: drone recall {a['recall_drone']:.3f} "
          f"precision {a['precision_drone']:.3f} acc {a['accuracy']:.3f}")

    even = df[df.distance_m % 4 == 0]
    odd = df[df.distance_m % 4 != 0]
    clf = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1,
                             random_seed=C.SEED, verbose=0,
                             allow_writing_files=False)
    clf.fit(even[C.FEATURE_COLUMNS], even["label"],
            sample_weight=compute_sample_weight("balanced", even["label"]))
    pb = clf.predict_proba(odd[C.FEATURE_COLUMNS])[:, list(clf.classes_).index("drone")]
    pred_b = np.where(pb >= 0.5, "drone", "anomaly")
    y = odd["label"]
    b = {"recall_drone": float(((pred_b == "drone") & (y == "drone")).sum()
                               / max((y == "drone").sum(), 1)),
         "precision_drone": float(((pred_b == "drone") & (y == "drone")).sum()
                                  / max((pred_b == "drone").sum(), 1)),
         "accuracy": float((pred_b == y).mean()),
         "train_distances": sorted(even.distance_m.unique().tolist()),
         "test_distances": sorted(odd.distance_m.unique().tolist())}
    print(f"B range-generalization: drone recall {b['recall_drone']:.3f} "
          f"precision {b['precision_drone']:.3f} acc {b['accuracy']:.3f}")

    joblib.dump(clf, C.MODELS_DIR / "real_tsms_model.joblib")

    out = {"dataset": "TSMS-Drone CW radar subset (figshare, real 2.495 GHz "
                      "S-band CW measurements, Han & Jung 2026)",
           "n_frames": len(df),
           "experiment_a_zero_shot": a,
           "experiment_b_range_generalization": b,
           "summary": (f"{len(df)} real CW frames (4 drones + corner reflector, "
                       f"2-30 m): zero-shot drone recall "
                       f"{a['recall_drone']:.1%}; range-generalization retrain "
                       f"(train even / test odd distances) drone recall "
                       f"{b['recall_drone']:.1%}, accuracy {b['accuracy']:.1%}.")}
    (C.REPORTS_DIR / "tsms_results.json").write_text(json.dumps(out, indent=2))
    print(f"results -> {C.REPORTS_DIR / 'tsms_results.json'}")


if __name__ == "__main__":
    main()
