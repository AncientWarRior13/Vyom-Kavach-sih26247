"""Standalone combined-schema (11 physics + 31 waveform-statistic) generator
-- Option C, built on top of the two already-separate 11-feature (frozen,
simulator.py) and 31-feature (feat31_*.py) pipelines.

Why a new generator rather than concatenating radar_train.parquet and
feat31_train_features.parquet columns: those are two INDEPENDENTLY sampled
datasets -- row i in one has no relationship to row i in the other, so
naively joining columns would fabricate combined-feature rows describing
two different targets as if they were one. A genuine combined dataset
needs every row to describe ONE consistent physical target.

This generator draws each instance's per-class LATENT physical parameters
(initial radial velocity v0, RCS, micro-Doppler bandwidth/jitter, blade
rate/count or anomaly kind, slant range, altitude) exactly once, using the
IDENTICAL per-class distributions as both frozen sources (confirmed by
inspection: simulator.py's _class_block and feat31_simulator.py's
_class_waveforms already use the same per-class distributions -- this
generator is the first place that's made explicit, by drawing them from
one shared block instead of two separately-seeded ones).

From that one shared instance, two trajectories are generated at each
schema's own native time resolution:
  - simulator.py's OWN frozen _ou_trajectory (imported unmodified, 8-step,
    DT=0.1) -> the 11 physics features, statistically identical in
    distribution to the frozen 11-feature schema's own features.
  - feat31_simulator.py's OWN _ou_waveform (imported unmodified, 100-step,
    DT=0.01) -> via feat31_features.extract_features (imported unmodified),
    the 31 waveform-statistic features, statistically identical in
    distribution to the 31-feature schema's own features.

The two trajectories share the same v0/theta/mu/sigma (same target
identity, same dynamics regime, same RCS/micro-Doppler/range) but are
independent noise realizations at their respective timescales -- not the
same numbers resampled, which would require reconciling two DT/step-count
conventions that were each separately tuned for their own schema. Disclosed
plainly: "same target, same dynamics parameters, two independently
time-sampled observations" is the honest description of what's combined
here, not "the exact same waveform downsampled."

SNR: ONE snr_db value per instance (same radar-equation coupling form used
by both frozen sources), used to set noise severity on BOTH feature sets
consistently -- an improvement over the two schemas' current decoupled SNR
realizations when generated separately. Uses the EXTENDED [-25,-10] dB band
(matching feat31_simulator.py, not the narrower frozen 11-feature band) so
the combined model can be evaluated at -25dB on equal footing with both
individual models.

Column count: 41 unique features (11 + 31, minus snr_db counted once, not
twice) + label. Called "combined 42-feature" informally since 11+31=42,
but training on a literal duplicate snr_db column would be meaningless --
disclosed here rather than silently padding to a round number.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from radarsense import config as C
from radarsense.feat31_features import FEAT31_COLUMNS, extract_features
from radarsense.feat31_simulator import _ou_waveform
from radarsense.simulator import _ou_trajectory

DT8 = 0.1
STEPS8 = 8
DT100 = 0.01
STEPS100 = 100
SNR_RANGE_DB = (-25.0, -10.0)

FEAT42_COLUMNS = C.FEATURE_COLUMNS + [c for c in FEAT31_COLUMNS if c != "snr_db"]


def _class_params_and_trajectories(rng, cls, n):
    """Draw per-class latent parameters ONCE (identical distributions to
    both frozen sources, confirmed by inspection), then generate both
    trajectories from the same v0/theta/mu/sigma."""
    if cls == "aircraft":
        v0 = rng.uniform(-50, 50, n)
        theta, mu, sigma = 0.05, v0, 0.4
        jerk_prob, jerk_scale = 0.0, 0.0
        rcs = rng.normal(12, 4, n)
        md_bw = np.abs(rng.normal(15, 10, n))
        md_jit = rng.beta(1.5, 10, n)
        rng_m = rng.uniform(800, 5000, n)
        alt = rng.uniform(500, 3000, n)
        beat_hz = None
    elif cls == "bird":
        v0 = rng.uniform(-20, 20, n)
        theta, mu, sigma = 0.6, v0 * 0.8, 2.5
        jerk_prob, jerk_scale = 0.0, 0.0
        rcs = rng.normal(-16, 3, n)
        md_bw = np.abs(rng.normal(45, 20, n))
        md_jit = rng.beta(4, 5, n)
        rng_m = rng.uniform(50, 2500, n)
        alt = rng.uniform(0, 1000, n)
        beat_hz = rng.uniform(2, 15, n)[:, None]
        beat_phase = rng.uniform(0, 2 * np.pi, n)[:, None]
    elif cls == "drone":
        hover = rng.random(n) < 0.35
        v0 = np.where(hover, rng.normal(0, 1.5, n), rng.uniform(-15, 15, n))
        theta, mu, sigma = 0.9, 0.0, 3.0
        jerk_prob, jerk_scale = 0.25, 4.0
        rcs = rng.normal(-14, 3.5, n)
        blade_rate = rng.uniform(80, 160, n)
        n_blades = rng.choice([2, 3, 4], n, p=[0.6, 0.2, 0.2])
        md_bw = np.clip(blade_rate * n_blades * rng.uniform(0.6, 1.1, n), 0, 500)
        md_jit = rng.beta(6, 3, n)
        rng_m = rng.uniform(50, 2000, n)
        alt = rng.uniform(0, 500, n)
        beat_hz = None
    else:
        kind = rng.choice(3, n, p=[0.4, 0.35, 0.25])
        v0 = np.select([kind == 0, kind == 1, kind == 2],
                       [rng.normal(0, 2, n), rng.uniform(-10, 10, n), rng.uniform(-35, 35, n)])
        sig = np.select([kind == 0, kind == 1, kind == 2], [0.3, 5.0, 1.0])
        theta, mu, sigma = 0.3, v0 * 0.5, 1.0
        jerk_prob, jerk_scale = 0.0, 0.0
        rcs = np.select([kind == 0, kind == 1, kind == 2],
                         [rng.normal(-2, 4, n), rng.normal(-8, 5, n), rng.normal(4, 4, n)])
        md_bw = np.select([kind == 0, kind == 1, kind == 2],
                           [np.abs(rng.normal(5, 4, n)), np.abs(rng.normal(250, 90, n)),
                            np.abs(rng.normal(60, 30, n))])
        md_jit = np.select([kind == 0, kind == 1, kind == 2],
                            [rng.beta(1.2, 12, n), rng.beta(7, 2.5, n), rng.beta(3, 6, n)])
        rng_m = rng.uniform(50, 5000, n)
        alt = np.select([kind == 0, kind == 1, kind == 2],
                         [rng.uniform(100, 3000, n), rng.uniform(0, 400, n), rng.uniform(100, 1500, n)])
        beat_hz = None

    traj8 = _ou_trajectory(rng, n, v0, theta=theta, mu=mu, sigma=sigma,
                            jerk_prob=jerk_prob, jerk_scale=jerk_scale)
    traj100 = _ou_waveform(rng, n, v0, theta=theta, mu=mu, sigma=sigma,
                            jerk_prob=jerk_prob, jerk_scale=jerk_scale, steps=STEPS100)

    if cls == "bird":
        traj8 = traj8 + 1.5 * np.sin(2 * np.pi * beat_hz * np.arange(STEPS8) * DT8 + beat_phase)
        traj100 = traj100 + 1.5 * np.sin(2 * np.pi * beat_hz * np.arange(STEPS100) * DT100 + beat_phase)
    elif cls == "anomaly":
        traj8 = traj8 + sig[:, None] * rng.standard_normal((n, STEPS8)) * np.sqrt(DT8)
        traj100 = traj100 + sig[:, None] * rng.standard_normal((n, STEPS100)) * np.sqrt(DT100)

    return {"traj8": traj8, "traj100": traj100, "rcs": rcs, "md_bw": md_bw, "md_jit": md_jit,
            "rng_m": rng_m, "alt": alt}


def _physics11(params, rng, n):
    """11-feature tail, matching simulator.py's _class_block exactly."""
    traj8 = params["traj8"]
    v_r = traj8[:, -1]
    accel = np.clip(np.diff(traj8, axis=1).mean(axis=1) / DT8, -10, 10)
    v_var = traj8.var(axis=1)
    return {
        "doppler_velocity": np.clip(v_r, -50, 50),
        "rcs_estimate": np.clip(params["rcs"], -20, 20),
        "micro_doppler_bandwidth": np.clip(params["md_bw"], 0, 500),
        "micro_doppler_jitter": np.clip(params["md_jit"], 0, 1),
        "range_m": params["rng_m"],
        "azimuth_deg": rng.uniform(-180, 180, n),
        "altitude_m": np.clip(params["alt"], 0, 3000),
        "velocity_variance": v_var,
        "acceleration": accel,
    }


def _waveforms31(params, rng, n):
    """31-feature waveform tail, matching feat31_simulator.py's
    _class_waveforms amplitude/energy derivation exactly."""
    rcs, md_bw, md_jit = params["rcs"], params["md_bw"], params["md_jit"]
    base_amp = np.sqrt(10 ** (rcs / 10.0))[:, None]
    mod_freq = (md_bw / 4.0)[:, None]
    mod_depth = md_jit[:, None]
    t_axis = np.arange(STEPS100) * DT100
    envelope = 1.0 + mod_depth * np.sin(2 * np.pi * mod_freq * t_axis + rng.uniform(0, 2 * np.pi, n)[:, None])
    amp = base_amp * np.clip(envelope, 0.05, None)
    amp = amp + 0.03 * base_amp * rng.standard_normal((n, STEPS100))
    energy = amp ** 2
    return {"amplitude": amp, "velocity": params["traj100"], "energy": energy}


def generate_dataset(n_samples=C.N_SAMPLES, seed=C.SEED):
    rng = np.random.default_rng(seed)
    phys_frames, wave_rows = [], []
    for cls, frac in C.CLASS_PROPORTIONS.items():
        n = int(round(n_samples * frac))
        params = _class_params_and_trajectories(rng, cls, n)

        raw = params["rcs"] - 40 * np.log10(params["rng_m"])
        lo, hi = np.percentile(raw, [2, 98])
        snr = SNR_RANGE_DB[0] + (raw - lo) / (hi - lo) * (SNR_RANGE_DB[1] - SNR_RANGE_DB[0])
        snr = np.clip(snr + rng.normal(0, 1.0, n), *SNR_RANGE_DB)
        noise_frac = (SNR_RANGE_DB[1] - snr) / (SNR_RANGE_DB[1] - SNR_RANGE_DB[0])

        phys = _physics11(params, rng, n)
        noise_scales = {
            "doppler_velocity": 4.0, "rcs_estimate": 3.0,
            "micro_doppler_bandwidth": 60.0, "micro_doppler_jitter": 0.12,
            "range_m": 30.0, "azimuth_deg": 2.0, "altitude_m": 50.0,
            "velocity_variance": 2.0, "acceleration": 1.5,
        }
        for feat, scale in noise_scales.items():
            phys[feat] = phys[feat] + noise_frac * scale * rng.standard_normal(n)
        phys["doppler_velocity"] = np.clip(phys["doppler_velocity"], -50, 50)
        phys["rcs_estimate"] = np.clip(phys["rcs_estimate"], -20, 20)
        phys["micro_doppler_bandwidth"] = np.clip(phys["micro_doppler_bandwidth"], 0, 500)
        phys["micro_doppler_jitter"] = np.clip(phys["micro_doppler_jitter"], 0, 1)
        phys["range_m"] = np.clip(phys["range_m"], 50, 5000)
        phys["azimuth_deg"] = np.clip(phys["azimuth_deg"], -180, 180)
        phys["altitude_m"] = np.clip(phys["altitude_m"], 0, 3000)
        phys["velocity_variance"] = np.clip(phys["velocity_variance"], 0, None)
        phys["acceleration"] = np.clip(phys["acceleration"], -10, 10)
        phys["doppler_freq_shift"] = 2 * phys["doppler_velocity"] * C.CARRIER_FREQ_HZ / C.C_LIGHT
        phys["snr_db"] = snr
        phys_df = pd.DataFrame(phys)
        phys_df[C.LABEL_COLUMN] = cls
        phys_frames.append(phys_df)

        wave = _waveforms31(params, rng, n)
        for key, scale in [("amplitude", 0.25), ("velocity", 4.0), ("energy", 0.35)]:
            wf = wave[key]
            wf_scale = (np.abs(wf).mean(axis=1, keepdims=True) + 1e-6) * scale
            wave[key] = wf + noise_frac[:, None] * wf_scale * rng.standard_normal(wf.shape)
        for i in range(n):
            wave_rows.append({
                "amplitude_waveform": wave["amplitude"][i],
                "velocity_waveform": wave["velocity"][i],
                "energy_waveform": wave["energy"][i],
                "snr_db": float(snr[i]),
                C.LABEL_COLUMN: cls,
            })

    phys_all = pd.concat(phys_frames, ignore_index=True)
    wave_all_raw = pd.DataFrame(wave_rows)
    wave_feats = extract_features(wave_all_raw)

    assert len(phys_all) == len(wave_feats)
    assert (phys_all[C.LABEL_COLUMN].values == wave_feats[C.LABEL_COLUMN].values).all()
    combined = pd.concat([
        phys_all[C.FEATURE_COLUMNS],
        wave_feats[[c for c in FEAT31_COLUMNS if c != "snr_db"]],
    ], axis=1)
    combined[C.LABEL_COLUMN] = phys_all[C.LABEL_COLUMN].values

    combined = combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    combined[FEAT42_COLUMNS] = combined[FEAT42_COLUMNS].astype(np.float64)
    return combined[FEAT42_COLUMNS + [C.LABEL_COLUMN]]


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
        sub.to_parquet(C.DATA_DIR / f"feat42_{name}.parquet")
    print(f"generated {len(df)} combined-feature samples, {len(FEAT42_COLUMNS)} unique features")
    print(df[C.LABEL_COLUMN].value_counts().to_string())
    print(f"SNR range: {df.snr_db.min():.1f} to {df.snr_db.max():.1f} dB")
    print(f"splits: train={len(train)} val={len(val)} test={len(test)}")


if __name__ == "__main__":
    main()
