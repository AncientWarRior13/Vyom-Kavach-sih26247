"""Extracts the 31 named statistical features (FEATURE_SCHEMAS.md Section 3)
from the raw 100-sample amplitude/velocity/energy waveforms produced by
feat31_simulator.py.

10 statistics per channel (amplitude, velocity, energy) x 3 channels = 30,
plus snr_db = 31 total, matching the schema doc's column list exactly:
  mean, std, min, max, range, energy(sum of squares), zcr (zero-crossing
  rate around the mean), dominant_freq (DFT peak), spectral_bandwidth
  (power-weighted spread around the spectral centroid -- the same
  centroid-based definition used elsewhere in this project's real-sensor
  bridges, e.g. diat_usat_bridge.py's micro-Doppler bandwidth), diff_var
  (variance of the first-difference signal).

Sample-rate assumption: fs = 100 Hz (100 samples / 1.0s), matching
feat31_simulator.py's DT=0.01s -- documented there, reused here for the
DFT frequency axis.
"""

import numpy as np
import pandas as pd

from radarsense import config as C
from radarsense.feat31_simulator import DT, N_STEPS

FREQS = np.fft.rfftfreq(N_STEPS, d=DT)

FEAT31_COLUMNS = ["snr_db"] + [
    f"{ch}_{stat}" for ch in ("amplitude", "velocity", "energy")
    for stat in ("mean", "std", "min", "max", "range", "energy", "zcr",
                 "dominant_freq", "spectral_bandwidth", "diff_var")
]


def _channel_stats(x, prefix):
    """x: (n, N_STEPS) waveform batch -> dict of 10 named stats for this channel."""
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    mn = x.min(axis=1)
    mx = x.max(axis=1)
    rng = mx - mn
    energy = (x ** 2).sum(axis=1)

    centered = x - mean[:, None]
    signs = np.sign(centered)
    signs[signs == 0] = 1
    zcr = (np.diff(signs, axis=1) != 0).sum(axis=1) / (x.shape[1] - 1)

    spec = np.abs(np.fft.rfft(x, axis=1)) ** 2
    dom_idx = spec.argmax(axis=1)
    dominant_freq = FREQS[dom_idx]
    power_sum = spec.sum(axis=1)
    centroid = (spec * FREQS[None, :]).sum(axis=1) / np.maximum(power_sum, 1e-12)
    spectral_bandwidth = np.sqrt(
        (spec * (FREQS[None, :] - centroid[:, None]) ** 2).sum(axis=1) / np.maximum(power_sum, 1e-12))

    diff_var = np.var(np.diff(x, axis=1), axis=1)

    return {
        f"{prefix}_mean": mean, f"{prefix}_std": std, f"{prefix}_min": mn,
        f"{prefix}_max": mx, f"{prefix}_range": rng, f"{prefix}_energy": energy,
        f"{prefix}_zcr": zcr, f"{prefix}_dominant_freq": dominant_freq,
        f"{prefix}_spectral_bandwidth": spectral_bandwidth, f"{prefix}_diff_var": diff_var,
    }


def extract_features(df):
    """df: rows with amplitude_waveform/velocity_waveform/energy_waveform
    (each a length-N_STEPS array) + snr_db + label -> tabular 31-feature df."""
    amp = np.stack(df["amplitude_waveform"].values)
    vel = np.stack(df["velocity_waveform"].values)
    ene = np.stack(df["energy_waveform"].values)

    out = {"snr_db": df["snr_db"].values}
    out.update(_channel_stats(amp, "amplitude"))
    out.update(_channel_stats(vel, "velocity"))
    out.update(_channel_stats(ene, "energy"))
    result = pd.DataFrame(out)
    result[C.LABEL_COLUMN] = df[C.LABEL_COLUMN].values
    return result[FEAT31_COLUMNS + [C.LABEL_COLUMN]]


def main():
    for split in ("train", "val", "test"):
        raw = pd.read_parquet(C.DATA_DIR / f"feat31_{split}.parquet")
        feats = extract_features(raw)
        feats.to_parquet(C.DATA_DIR / f"feat31_{split}_features.parquet")
        print(f"{split}: {feats.shape}")
    print(f"columns ({len(FEAT31_COLUMNS)}): {FEAT31_COLUMNS}")


if __name__ == "__main__":
    main()
