"""Bridge: SAAB SIRS 77 GHz FMCW raw I/Q segments -> RadarSense feature schema.

Dataset: Zenodo 5896641 (CC-BY-4.0), Karlsson et al., RadarConf 2022.
75,868 real scan segments: 6 drone types, 5+ bird species, humans, corner reflector.
Each segment: 1280 complex samples = 5 range cells x 256 azimuth/time samples
(15 ms dwell, PRF ~17 kHz), plus range, timestamp, and an official 1/2/3
train/val/test split indicator.

Feature extraction (documented honesty notes):
- doppler_velocity: peak of the Doppler spectrum (DC+-1 bin notched: stationary
  clutter and antenna-scan leakage live at DC), converted to m/s at 77 GHz.
  Carrier-independent in m/s.
- doppler_freq_shift: RE-DERIVED at the schema's 10 GHz carrier (2*v*f10/c) so
  the column means the same thing as in simulation.
- micro_doppler_bandwidth: power-weighted spectral std (Hz) of above-noise bins
  around the centroid, scaled by 10/77 to the schema carrier. Blade-flash
  bandwidth scales linearly with carrier frequency.
- micro_doppler_jitter: std of per-subwindow (4x64) spectral-centroid velocity,
  normalized by 10 m/s and clipped to [0,1].
- rcs_estimate: 10log10(signal power) + 40log10(R), radiometrically calibrated
  so the corner-reflector class median = +20 dBsm (its known RCS: 100 m^2).
- range_m: provided by the dataset.
- velocity_variance / acceleration: computed from consecutive segments of the
  same measurement within +-0.6 s (real kinematics, ~100 ms between segments).
- snr_db: measured peak/floor SNR, affinely mapped so the dataset median sits
  at -15 dB in the schema band [-20,-10] (indicative only; documented).
- azimuth_deg, altitude_m: NOT MEASURABLE from this sensor -> constant 0
  (uninformative by construction; documented in every report using this data).

Class mapping: D1..D6 -> drone; all bird species -> bird;
human_walk/human_run/CR -> anomaly. No aircraft class exists in this dataset.
"""

import json

import numpy as np
import pandas as pd

from radarsense import config as C

RAW = C.ROOT / "external_data/saab_77ghz/data_SAAB_SIRS_77GHz_FMCW.npy"
OUT = C.DATA_DIR / "real_saab77.parquet"

F77 = 77e9
PRF = 256 / 0.015
FREQS = np.fft.fftshift(np.fft.fftfreq(256, d=1 / PRF))
V_PER_HZ = C.C_LIGHT / (2 * F77)

LABEL_MAP = {}
for d in ("D1", "D2", "D3", "D4", "D5", "D6"):
    LABEL_MAP[d] = "drone"
for b in ("seagull", "pigeon", "raven", "black-headed gull",
          "seagull and black-headed gull", "heron"):
    LABEL_MAP[b] = "bird"
for a in ("human_walk", "human_run", "CR"):
    LABEL_MAP[a] = "anomaly"


def segment_features(seg, rng_m):
    """seg: (1280,) complex -> dict of spectral features (kinematics added later)."""
    cells = seg.reshape(5, 256)
    win = np.hanning(256)
    spec = np.abs(np.fft.fftshift(np.fft.fft(cells * win, axis=1), axes=1)) ** 2
    P = spec.mean(axis=0)

    dc = len(P) // 2
    notched = P.copy()
    notched[dc - 1:dc + 2] = 0.0

    floor = np.median(P)
    peak_i = int(np.argmax(notched))
    peak = notched[peak_i]
    snr_raw = 10 * np.log10(max(peak, 1e-12) / max(floor, 1e-12))

    v = FREQS[peak_i] * V_PER_HZ

    sig = notched > (floor * 2.0)
    if sig.sum() >= 2:
        f_sig, p_sig = FREQS[sig], notched[sig]
        centroid = np.average(f_sig, weights=p_sig)
        bw77 = np.sqrt(np.average((f_sig - centroid) ** 2, weights=p_sig))
    else:
        bw77 = 0.0
    bw10 = bw77 * (C.CARRIER_FREQ_HZ / F77)

    cents = []
    subf = np.fft.fftshift(np.fft.fftfreq(64, d=1 / PRF))
    for k in range(4):
        sub = cells[:, k * 64:(k + 1) * 64]
        sp = np.abs(np.fft.fftshift(np.fft.fft(sub * np.hanning(64), axis=1),
                                    axes=1)) ** 2
        pp = sp.mean(axis=0)
        pp[len(pp) // 2] = 0.0
        m = pp > np.median(pp) * 2
        if m.sum() >= 2:
            cents.append(np.average(subf[m], weights=pp[m]) * V_PER_HZ)
    jitter = float(np.clip(np.std(cents) / 10.0, 0, 1)) if len(cents) >= 2 else 0.0

    sig_power = float(spec.sum())
    power_metric = 10 * np.log10(max(sig_power, 1e-12)) + 40 * np.log10(max(rng_m, 1.0))

    return {
        "doppler_velocity": float(np.clip(v, -50, 50)),
        "micro_doppler_bandwidth": float(np.clip(bw10, 0, 500)),
        "micro_doppler_jitter": jitter,
        "range_m": float(np.clip(rng_m, 50, 5000)),
        "_snr_raw": float(snr_raw),
        "_power_metric": power_metric,
    }


def main():
    print("loading raw npy (1.5 GB)...")
    data = np.load(RAW, allow_pickle=True)
    rows = []
    for mi in range(data.shape[0]):
        raw_label = str(np.ravel(np.asarray(data[mi, 0], dtype=object))[0])
        label = LABEL_MAP.get(raw_label)
        if label is None:
            continue
        segs = data[mi, 1]
        rngs = np.asarray(data[mi, 2], dtype=float).ravel()
        times = np.asarray(data[mi, 3], dtype=float).ravel()
        split = np.asarray(data[mi, 4], dtype=int).ravel()
        n = segs.shape[1]
        if len(rngs) == 1:
            rngs = np.full(n, rngs[0])
        feats = []
        for si in range(n):
            f = segment_features(segs[:, si], rngs[si] if si < len(rngs) else rngs[-1])
            f["_t"] = times[si] if si < len(times) else np.nan
            f["_split"] = int(split[si]) if si < len(split) else 1
            f["_measurement"] = mi
            f["raw_label"] = raw_label
            f[C.LABEL_COLUMN] = label
            feats.append(f)
        t = np.array([f["_t"] for f in feats])
        v = np.array([f["doppler_velocity"] for f in feats])
        for si, f in enumerate(feats):
            m = np.abs(t - t[si]) <= 0.6
            if m.sum() >= 3:
                tv, vv = t[m], v[m]
                f["velocity_variance"] = float(np.clip(np.var(vv), 0, None))
                A = np.polyfit(tv, vv, 1)[0]
                f["acceleration"] = float(np.clip(A, -10, 10))
            else:
                f["velocity_variance"] = 0.0
                f["acceleration"] = 0.0
        rows.extend(feats)
        if (mi + 1) % 20 == 0:
            print(f"  measurement {mi + 1}/130, segments so far: {len(rows)}")

    df = pd.DataFrame(rows)

    cr = df[df.raw_label == "CR"]["_power_metric"]
    offset = float(cr.median() - 20.0) if len(cr) else float(df["_power_metric"].median())
    df["rcs_estimate"] = (df["_power_metric"] - offset).clip(-20, 20)

    snr = df["_snr_raw"]
    scale = 5.0 / max(float(snr.quantile(0.95) - snr.median()), 1e-6)
    df["snr_db"] = (-15.0 + (snr - snr.median()) * scale).clip(-20, -10)

    df["doppler_freq_shift"] = 2 * df["doppler_velocity"] * C.CARRIER_FREQ_HZ / C.C_LIGHT
    df["azimuth_deg"] = 0.0
    df["altitude_m"] = 0.0

    keep = C.FEATURE_COLUMNS + [C.LABEL_COLUMN, "raw_label", "_split", "_measurement"]
    df = df[keep].rename(columns={"_split": "split", "_measurement": "measurement"})
    df.to_parquet(OUT)
    print(f"\nwrote {len(df)} real segments -> {OUT}")
    print(df[C.LABEL_COLUMN].value_counts().to_string())
    print(df.groupby(C.LABEL_COLUMN)[["doppler_velocity", "micro_doppler_bandwidth",
                                      "rcs_estimate"]].median().round(2).to_string())
    meta = {"source": "Zenodo 5896641, SAAB SIRS 1600 77GHz FMCW (CC-BY-4.0)",
            "rcs_calibration_offset_db": offset,
            "n_segments": len(df),
            "not_measurable_constant_zero": ["azimuth_deg", "altitude_m"]}
    (C.REPORTS_DIR / "real_data_bridge_meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
