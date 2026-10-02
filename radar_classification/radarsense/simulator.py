"""Physics-grounded radar return simulator for RadarSense.

Statistical simulator — models published statistical properties of radar
returns (RCS ranges, micro-Doppler structure, kinematics), NOT full
electromagnetic physics. Real-sensor generalization is unverified.

Design notes (defensible in Q&A):
- RCS ranges per class from open literature: small quadcopter ~ -20..-8 dBsm,
  bird ~ -20..-10 dBsm (starling..raptor), light/commercial aircraft +5..+20 dBsm.
- Micro-Doppler: drone rotor blade-flash bandwidth = n_blades * blade_rate
  harmonics (2-blade props at 80-160 rev/s -> 160..500 Hz spread). Bird
  wing-beats are 2-15 Hz quasi-periodic -> narrow (10-80 Hz) spread.
- Kinematics are derived from an 8-step simulated radial-velocity trajectory
  per target (OU-style dynamics per class), so velocity_variance and
  acceleration are genuinely computed, not sampled.
- SNR is coupled to RCS and range via the radar equation form
  SNR ~ RCS_dB - 40*log10(R), then affinely mapped into the spec band
  [-20, -10] dB. Lower SNR inflates measurement noise on every observed
  feature, which is what makes low-SNR classification genuinely harder.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from radarsense import config as C

DT = 0.1
TRAJ_STEPS = 8


def _ou_trajectory(rng, n, v0, theta, mu, sigma, jerk_prob=0.0, jerk_scale=0.0):
    """Ornstein-Uhlenbeck radial-velocity track with optional jerk events."""
    v = np.empty((n, TRAJ_STEPS))
    v[:, 0] = v0
    for t in range(1, TRAJ_STEPS):
        dv = theta * (mu - v[:, t - 1]) * DT + sigma * np.sqrt(DT) * rng.standard_normal(n)
        if jerk_prob > 0:
            jerks = rng.random(n) < jerk_prob
            dv += jerks * rng.normal(0, jerk_scale, n)
        v[:, t] = v[:, t - 1] + dv
    return v


def _class_block(rng, cls, n):
    """Generate n 'true' (pre-noise) samples for one class."""
    if cls == "aircraft":
        v0 = rng.uniform(-50, 50, n)
        traj = _ou_trajectory(rng, n, v0, theta=0.05, mu=v0, sigma=0.4)
        rcs = rng.normal(12, 4, n)
        md_bw = np.abs(rng.normal(15, 10, n))
        md_jit = rng.beta(1.5, 10, n)
        rng_m = rng.uniform(800, 5000, n)
        alt = rng.uniform(500, 3000, n)
    elif cls == "bird":
        v0 = rng.uniform(-20, 20, n)
        traj = _ou_trajectory(rng, n, v0, theta=0.6, mu=v0 * 0.8, sigma=2.5)
        beat_hz = rng.uniform(2, 15, n)[:, None]
        phase = rng.uniform(0, 2 * np.pi, n)[:, None]
        traj = traj + 1.5 * np.sin(2 * np.pi * beat_hz * np.arange(TRAJ_STEPS) * DT + phase)
        rcs = rng.normal(-16, 3, n)
        md_bw = np.abs(rng.normal(45, 20, n))
        md_jit = rng.beta(4, 5, n)
        rng_m = rng.uniform(50, 2500, n)
        alt = rng.uniform(0, 1000, n)
    elif cls == "drone":
        hover = rng.random(n) < 0.35
        v0 = np.where(hover, rng.normal(0, 1.5, n), rng.uniform(-15, 15, n))
        traj = _ou_trajectory(rng, n, v0, theta=0.9, mu=0.0, sigma=3.0,
                              jerk_prob=0.25, jerk_scale=4.0)
        rcs = rng.normal(-14, 3.5, n)
        blade_rate = rng.uniform(80, 160, n)
        n_blades = rng.choice([2, 3, 4], n, p=[0.6, 0.2, 0.2])
        md_bw = np.clip(blade_rate * n_blades * rng.uniform(0.6, 1.1, n), 0, 500)
        md_jit = rng.beta(6, 3, n)
        rng_m = rng.uniform(50, 2000, n)
        alt = rng.uniform(0, 500, n)
    else:
        kind = rng.choice(3, n, p=[0.4, 0.35, 0.25])
        v0 = np.select(
            [kind == 0, kind == 1, kind == 2],
            [rng.normal(0, 2, n), rng.uniform(-10, 10, n), rng.uniform(-35, 35, n)])
        sig = np.select([kind == 0, kind == 1, kind == 2], [0.3, 5.0, 1.0])
        traj = _ou_trajectory(rng, n, v0, theta=0.3, mu=v0 * 0.5, sigma=1.0)
        traj += sig[:, None] * rng.standard_normal((n, TRAJ_STEPS)) * np.sqrt(DT)
        rcs = np.select(
            [kind == 0, kind == 1, kind == 2],
            [rng.normal(-2, 4, n), rng.normal(-8, 5, n), rng.normal(4, 4, n)])
        md_bw = np.select(
            [kind == 0, kind == 1, kind == 2],
            [np.abs(rng.normal(5, 4, n)),
             np.abs(rng.normal(250, 90, n)),
             np.abs(rng.normal(60, 30, n))])
        md_jit = np.select(
            [kind == 0, kind == 1, kind == 2],
            [rng.beta(1.2, 12, n), rng.beta(7, 2.5, n), rng.beta(3, 6, n)])
        rng_m = rng.uniform(50, 5000, n)
        alt = np.select(
            [kind == 0, kind == 1, kind == 2],
            [rng.uniform(100, 3000, n), rng.uniform(0, 400, n), rng.uniform(100, 1500, n)])

    v_r = traj[:, -1]
    accel = np.clip(np.diff(traj, axis=1).mean(axis=1) / DT, -10, 10)
    v_var = traj.var(axis=1)
    rcs = np.clip(rcs, -20, 20)
    return {
        "doppler_velocity": np.clip(v_r, -50, 50),
        "rcs_estimate": rcs,
        "micro_doppler_bandwidth": np.clip(md_bw, 0, 500),
        "micro_doppler_jitter": np.clip(md_jit, 0, 1),
        "range_m": rng_m,
        "azimuth_deg": rng.uniform(-180, 180, n),
        "altitude_m": np.clip(alt, 0, 3000),
        "velocity_variance": v_var,
        "acceleration": accel,
    }


def generate_dataset(n_samples=C.N_SAMPLES, seed=C.SEED):
    rng = np.random.default_rng(seed)
    frames = []
    for cls, frac in C.CLASS_PROPORTIONS.items():
        n = int(round(n_samples * frac))
        block = _class_block(rng, cls, n)

        raw = block["rcs_estimate"] - 40 * np.log10(block["range_m"])
        lo, hi = np.percentile(raw, [2, 98])
        snr = C.SNR_RANGE_DB[0] + (raw - lo) / (hi - lo) * (C.SNR_RANGE_DB[1] - C.SNR_RANGE_DB[0])
        snr = np.clip(snr + rng.normal(0, 1.0, n), *C.SNR_RANGE_DB)
        block["snr_db"] = snr

        noise_frac = (C.SNR_RANGE_DB[1] - snr) / (C.SNR_RANGE_DB[1] - C.SNR_RANGE_DB[0])
        noise_scales = {
            "doppler_velocity": 4.0, "rcs_estimate": 3.0,
            "micro_doppler_bandwidth": 60.0, "micro_doppler_jitter": 0.12,
            "range_m": 30.0, "azimuth_deg": 2.0, "altitude_m": 50.0,
            "velocity_variance": 2.0, "acceleration": 1.5,
        }
        for feat, scale in noise_scales.items():
            block[feat] = block[feat] + noise_frac * scale * rng.standard_normal(n)

        block["doppler_velocity"] = np.clip(block["doppler_velocity"], -50, 50)
        block["rcs_estimate"] = np.clip(block["rcs_estimate"], -20, 20)
        block["micro_doppler_bandwidth"] = np.clip(block["micro_doppler_bandwidth"], 0, 500)
        block["micro_doppler_jitter"] = np.clip(block["micro_doppler_jitter"], 0, 1)
        block["range_m"] = np.clip(block["range_m"], 50, 5000)
        block["azimuth_deg"] = np.clip(block["azimuth_deg"], -180, 180)
        block["altitude_m"] = np.clip(block["altitude_m"], 0, 3000)
        block["velocity_variance"] = np.clip(block["velocity_variance"], 0, None)
        block["acceleration"] = np.clip(block["acceleration"], -10, 10)

        block["doppler_freq_shift"] = (
            2 * block["doppler_velocity"] * C.CARRIER_FREQ_HZ / C.C_LIGHT)

        df = pd.DataFrame(block)
        df[C.LABEL_COLUMN] = cls
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    data = data[C.FEATURE_COLUMNS + [C.LABEL_COLUMN]]
    data = data.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    data[C.FEATURE_COLUMNS] = data[C.FEATURE_COLUMNS].astype(np.float64)
    return data


def make_splits(data, seed=C.SEED):
    """Stratified 70/15/15 train/val/test."""
    train, rest = train_test_split(
        data, test_size=C.VAL_FRAC + C.TEST_FRAC,
        stratify=data[C.LABEL_COLUMN], random_state=seed)
    val, test = train_test_split(
        rest, test_size=C.TEST_FRAC / (C.VAL_FRAC + C.TEST_FRAC),
        stratify=rest[C.LABEL_COLUMN], random_state=seed)
    return (train.reset_index(drop=True),
            val.reset_index(drop=True),
            test.reset_index(drop=True))


def main():
    data = generate_dataset()
    train, val, test = make_splits(data)
    data.to_parquet(C.DATA_DIR / "radar_full.parquet")
    data.to_csv(C.DATA_DIR / "radar_full.csv", index=False)
    for name, df in [("train", train), ("val", val), ("test", test)]:
        df.to_parquet(C.DATA_DIR / f"radar_{name}.parquet")
    counts = data[C.LABEL_COLUMN].value_counts()
    print(f"generated {len(data)} samples")
    print(counts.to_string())
    maj = counts.drop(C.MINORITY_CLASS).max()
    print(f"imbalance (largest majority : drone) = {maj / counts[C.MINORITY_CLASS]:.1f}:1")
    print(f"splits: train={len(train)} val={len(val)} test={len(test)}")


if __name__ == "__main__":
    main()
