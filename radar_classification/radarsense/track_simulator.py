"""Track-level (multi-frame) simulator: generates K consecutive noisy
detections of the SAME underlying target, sharing one clean physics draw
(reusing simulator.py's _class_block, unmodified) but with INDEPENDENT
per-frame measurement noise -- mimicking consecutive radar returns of one
physical target, where the target's true kinematics/RCS/micro-Doppler
don't change frame-to-frame but the noise realization does.

This does NOT touch or modify simulator.py, config.py, or the frozen
single-snapshot dataset -- purely additive, for the track-fusion experiment
in track_fusion.py.

Split discipline: split by TRACK, never by frame -- a track's frames must
never span train/test (the obvious leakage risk here, analogous to the
real-sensor bridges' session-aware splits).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from radarsense import config as C
from radarsense.simulator import _class_block

K_FRAMES = 5
N_TRACKS = 10000


def generate_tracks(n_tracks=N_TRACKS, k_frames=K_FRAMES, seed=C.SEED):
    rng = np.random.default_rng(seed)
    rows = []
    track_id = 0
    for cls, frac in C.CLASS_PROPORTIONS.items():
        n = max(1, int(round(n_tracks * frac)))
        block = _class_block(rng, cls, n)
        for i in range(n):
            clean = {k: v[i] for k, v in block.items()}
            for frame in range(k_frames):
                row = dict(clean)
                row["track_id"] = track_id
                row["frame"] = frame
                row[C.LABEL_COLUMN] = cls
                rows.append(row)
            track_id += 1

    df = pd.DataFrame(rows)

    raw = df["rcs_estimate"] - 40 * np.log10(df["range_m"])
    lo, hi = np.percentile(raw, [2, 98])
    snr = C.SNR_RANGE_DB[0] + (raw - lo) / (hi - lo) * (C.SNR_RANGE_DB[1] - C.SNR_RANGE_DB[0])
    df["snr_db"] = np.clip(snr + rng.normal(0, 1.0, len(df)), *C.SNR_RANGE_DB)

    noise_frac = (C.SNR_RANGE_DB[1] - df["snr_db"]) / (C.SNR_RANGE_DB[1] - C.SNR_RANGE_DB[0])
    noise_scales = {
        "doppler_velocity": 4.0, "rcs_estimate": 3.0,
        "micro_doppler_bandwidth": 60.0, "micro_doppler_jitter": 0.12,
        "range_m": 30.0, "azimuth_deg": 2.0, "altitude_m": 50.0,
        "velocity_variance": 2.0, "acceleration": 1.5,
    }
    for feat, scale in noise_scales.items():
        df[feat] = df[feat] + noise_frac * scale * rng.standard_normal(len(df))

    df["doppler_velocity"] = df["doppler_velocity"].clip(-50, 50)
    df["rcs_estimate"] = df["rcs_estimate"].clip(-20, 20)
    df["micro_doppler_bandwidth"] = df["micro_doppler_bandwidth"].clip(0, 500)
    df["micro_doppler_jitter"] = df["micro_doppler_jitter"].clip(0, 1)
    df["range_m"] = df["range_m"].clip(50, 5000)
    df["azimuth_deg"] = df["azimuth_deg"].clip(-180, 180)
    df["altitude_m"] = df["altitude_m"].clip(0, 3000)
    df["velocity_variance"] = df["velocity_variance"].clip(lower=0)
    df["acceleration"] = df["acceleration"].clip(-10, 10)
    df["doppler_freq_shift"] = 2 * df["doppler_velocity"] * C.CARRIER_FREQ_HZ / C.C_LIGHT

    cols = ["track_id", "frame"] + C.FEATURE_COLUMNS + [C.LABEL_COLUMN]
    return df[cols].astype({c: np.float64 for c in C.FEATURE_COLUMNS})


def make_track_splits(df, seed=C.SEED):
    """Split by TRACK id, not by frame -- a track never spans splits."""
    tracks = df[["track_id", C.LABEL_COLUMN]].drop_duplicates("track_id")
    train_t, rest_t = train_test_split(tracks, test_size=C.VAL_FRAC + C.TEST_FRAC,
                                        stratify=tracks[C.LABEL_COLUMN], random_state=seed)
    val_t, test_t = train_test_split(rest_t, test_size=C.TEST_FRAC / (C.VAL_FRAC + C.TEST_FRAC),
                                      stratify=rest_t[C.LABEL_COLUMN], random_state=seed)
    train = df[df.track_id.isin(train_t.track_id)].reset_index(drop=True)
    val = df[df.track_id.isin(val_t.track_id)].reset_index(drop=True)
    test = df[df.track_id.isin(test_t.track_id)].reset_index(drop=True)
    return train, val, test


def main():
    df = generate_tracks()
    train, val, test = make_track_splits(df)
    for name, sub in [("train", train), ("val", val), ("test", test)]:
        sub.to_parquet(C.DATA_DIR / f"tracks_{name}.parquet")
    print(f"generated {df.track_id.nunique()} tracks x {K_FRAMES} frames = {len(df)} rows")
    print(df.groupby(C.LABEL_COLUMN).track_id.nunique().to_string())
    print(f"splits (rows): train={len(train)} val={len(val)} test={len(test)}")
    print(f"splits (tracks): train={train.track_id.nunique()} "
          f"val={val.track_id.nunique()} test={test.track_id.nunique()}")


if __name__ == "__main__":
    main()
