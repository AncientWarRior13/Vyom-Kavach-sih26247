"""300-step variant of feat31_simulator.py -- "longer waveforms" experiment.

Per-class physics (distributions, RCS/micro-Doppler/OU parameters) are
IDENTICAL to feat31_simulator.py -- this file exists only because N_STEPS
is baked into feat31_simulator.py's module-level constants and several
downstream computations (t_axis, DFT frequency axis), and feat31_simulator.py
is itself treated as a stable, already-tested artifact that feat31_train.py/
feat31_compare.py/feat31_quantum_snr.py depend on -- not something to modify
even additively while those depend on its exact current behavior. Following
the same "duplicate rather than risk the frozen artifact" philosophy
feat31_simulator.py itself already uses for _ou_trajectory relative to
simulator.py.

DT is kept at 0.01s (same sample RATE as the 100-step schema) and the
window is extended to 3.0s (300 steps) rather than resampled faster within
the same 1.0s window -- "longer waveforms" means observing for longer, not
sampling more finely over the same duration. This is the more informative
choice for the specific features SHAP flagged as most degraded by noise
(amplitude_spectral_bandwidth, amplitude_dominant_freq, energy_
spectral_bandwidth): DFT frequency resolution is 1/T, so a 3x longer window
gives 3x finer frequency resolution (0.33 Hz vs 1.0 Hz bins), which is
exactly what should make dominant-frequency and spectral-bandwidth
estimates better-conditioned under noise -- the mechanism this experiment
is testing, not asserting.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from radarsense import config as C

N_STEPS = 300
DT = 0.01
SNR_RANGE_DB = (-25.0, -10.0)


def _ou_waveform(rng, n, v0, theta, mu, sigma, jerk_prob=0.0, jerk_scale=0.0, steps=N_STEPS):
    v = np.empty((n, steps))
    v[:, 0] = v0
    for t in range(1, steps):
        dv = theta * (mu - v[:, t - 1]) * DT + sigma * np.sqrt(DT) * rng.standard_normal(n)
        if jerk_prob > 0:
            jerks = rng.random(n) < jerk_prob
            dv += jerks * rng.normal(0, jerk_scale, n)
        v[:, t] = v[:, t - 1] + dv
    return v


def _class_waveforms(rng, cls, n):
    """Identical per-class distributions to feat31_simulator.py's
    _class_waveforms -- only N_STEPS/DT differ."""
    if cls == "aircraft":
        v0 = rng.uniform(-50, 50, n)
        vel = _ou_waveform(rng, n, v0, theta=0.05, mu=v0, sigma=0.4)
        rcs = np.clip(rng.normal(12, 4, n), -20, 20)
        md_bw = np.clip(np.abs(rng.normal(15, 10, n)), 0, 500)
        md_jit = rng.beta(1.5, 10, n)
        rng_m = rng.uniform(800, 5000, n)
    elif cls == "bird":
        v0 = rng.uniform(-20, 20, n)
        vel = _ou_waveform(rng, n, v0, theta=0.6, mu=v0 * 0.8, sigma=2.5)
        beat_hz = rng.uniform(2, 15, n)[:, None]
        phase = rng.uniform(0, 2 * np.pi, n)[:, None]
        vel = vel + 1.5 * np.sin(2 * np.pi * beat_hz * np.arange(N_STEPS) * DT + phase)
        rcs = np.clip(rng.normal(-16, 3, n), -20, 20)
        md_bw = np.clip(np.abs(rng.normal(45, 20, n)), 0, 500)
        md_jit = rng.beta(4, 5, n)
        rng_m = rng.uniform(50, 2500, n)
    elif cls == "drone":
        hover = rng.random(n) < 0.35
        v0 = np.where(hover, rng.normal(0, 1.5, n), rng.uniform(-15, 15, n))
        vel = _ou_waveform(rng, n, v0, theta=0.9, mu=0.0, sigma=3.0, jerk_prob=0.25, jerk_scale=4.0)
        rcs = np.clip(rng.normal(-14, 3.5, n), -20, 20)
        blade_rate = rng.uniform(80, 160, n)
        n_blades = rng.choice([2, 3, 4], n, p=[0.6, 0.2, 0.2])
        md_bw = np.clip(blade_rate * n_blades * rng.uniform(0.6, 1.1, n), 0, 500)
        md_jit = rng.beta(6, 3, n)
        rng_m = rng.uniform(50, 2000, n)
    else:
        kind = rng.choice(3, n, p=[0.4, 0.35, 0.25])
        v0 = np.select([kind == 0, kind == 1, kind == 2],
                        [rng.normal(0, 2, n), rng.uniform(-10, 10, n), rng.uniform(-35, 35, n)])
        sig = np.select([kind == 0, kind == 1, kind == 2], [0.3, 5.0, 1.0])
        vel = _ou_waveform(rng, n, v0, theta=0.3, mu=v0 * 0.5, sigma=1.0)
        vel = vel + sig[:, None] * rng.standard_normal((n, N_STEPS)) * np.sqrt(DT)
        rcs = np.clip(np.select([kind == 0, kind == 1, kind == 2],
                                  [rng.normal(-2, 4, n), rng.normal(-8, 5, n), rng.normal(4, 4, n)]), -20, 20)
        md_bw = np.clip(np.select([kind == 0, kind == 1, kind == 2],
                                    [np.abs(rng.normal(5, 4, n)), np.abs(rng.normal(250, 90, n)),
                                     np.abs(rng.normal(60, 30, n))]), 0, 500)
        md_jit = np.select([kind == 0, kind == 1, kind == 2],
                             [rng.beta(1.2, 12, n), rng.beta(7, 2.5, n), rng.beta(3, 6, n)])
        rng_m = rng.uniform(50, 5000, n)

    base_amp = np.sqrt(10 ** (rcs / 10.0))[:, None]
    mod_freq = (md_bw / 4.0)[:, None]
    mod_depth = md_jit[:, None]
    t_axis = np.arange(N_STEPS) * DT
    envelope = 1.0 + mod_depth * np.sin(2 * np.pi * mod_freq * t_axis + rng.uniform(0, 2 * np.pi, n)[:, None])
    amp = base_amp * np.clip(envelope, 0.05, None)
    amp = amp + 0.03 * base_amp * rng.standard_normal((n, N_STEPS))
    energy = amp ** 2

    return {"amplitude": amp, "velocity": vel, "energy": energy,
            "rcs_estimate": rcs, "range_m": rng_m}


def generate_dataset(n_samples=C.N_SAMPLES, seed=C.SEED):
    rng = np.random.default_rng(seed)
    rows = []
    for cls, frac in C.CLASS_PROPORTIONS.items():
        n = int(round(n_samples * frac))
        block = _class_waveforms(rng, cls, n)

        raw = block["rcs_estimate"] - 40 * np.log10(block["range_m"])
        lo, hi = np.percentile(raw, [2, 98])
        snr = SNR_RANGE_DB[0] + (raw - lo) / (hi - lo) * (SNR_RANGE_DB[1] - SNR_RANGE_DB[0])
        snr = np.clip(snr + rng.normal(0, 1.0, n), *SNR_RANGE_DB)

        noise_frac = (SNR_RANGE_DB[1] - snr) / (SNR_RANGE_DB[1] - SNR_RANGE_DB[0])
        for key, scale in [("amplitude", 0.25), ("velocity", 4.0), ("energy", 0.35)]:
            wf = block[key]
            wf_scale = (np.abs(wf).mean(axis=1, keepdims=True) + 1e-6) * scale
            block[key] = wf + noise_frac[:, None] * wf_scale * rng.standard_normal(wf.shape)

        for i in range(n):
            rows.append({
                "amplitude_waveform": block["amplitude"][i],
                "velocity_waveform": block["velocity"][i],
                "energy_waveform": block["energy"][i],
                "snr_db": float(snr[i]),
                C.LABEL_COLUMN: cls,
            })

    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def make_splits(df, seed=C.SEED):
    train, rest = train_test_split(df, test_size=C.VAL_FRAC + C.TEST_FRAC,
                                    stratify=df[C.LABEL_COLUMN], random_state=seed)
    val, test = train_test_split(rest, test_size=C.TEST_FRAC / (C.VAL_FRAC + C.TEST_FRAC),
                                  stratify=rest[C.LABEL_COLUMN], random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def main():
    df = generate_dataset()
    train, val, test = make_splits(df)
    for name, sub in [("full", df), ("train", train), ("val", val), ("test", test)]:
        sub.to_parquet(C.DATA_DIR / f"feat31_300_{name}.parquet")
    counts = df[C.LABEL_COLUMN].value_counts()
    print(f"generated {len(df)} raw-waveform samples ({N_STEPS} steps, {N_STEPS*DT:.1f}s window)")
    print(counts.to_string())
    print(f"SNR range: {df.snr_db.min():.1f} to {df.snr_db.max():.1f} dB")
    print(f"splits: train={len(train)} val={len(val)} test={len(test)}")


if __name__ == "__main__":
    main()
