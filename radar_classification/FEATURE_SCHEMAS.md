# RadarSense — Feature Engineering & Schemas

This document lists the engineered feature schemas used across the RadarSense target
classification pipeline.

## 1. Classical Model Schema (11 Features)
The deployed CatBoost classical model is trained on the following 11 engineered features
(frozen schema contract). **This is the incumbent, unchanged pipeline** — see `radarsense/config.py`.

| Column | Feature | Type | Unit | Range | Description |
|---|---|---|---|---|---|
| 1 | `doppler_velocity` | float64 | m/s | −50..50 | Radial velocity, last step of an 8-step simulated track |
| 2 | `doppler_freq_shift` | float64 | Hz | ≈ −3333..3333 | 2·v·f_c/c at f_c = 10 GHz, derived from observed velocity |
| 3 | `rcs_estimate` | float64 | dBsm | −20..20 | Radar cross-section proxy; smaller = stealthier |
| 4 | `micro_doppler_bandwidth` | float64 | Hz | 0..500 | Spectral spread from rotating/flapping parts — key drone discriminator |
| 5 | `micro_doppler_jitter` | float64 | — | 0..1 | Normalized micro-motion irregularity |
| 6 | `range_m` | float64 | m | 50..5000 | Target range |
| 7 | `azimuth_deg` | float64 | deg | −180..180 | Bearing |
| 8 | `altitude_m` | float64 | m | 0..3000 | Estimated altitude |
| 9 | `velocity_variance` | float64 | (m/s)² | 0+ | Variance of the simulated velocity track (derived, not sampled) |
| 10 | `acceleration` | float64 | m/s² | −10..10 | Mean derivative of the velocity track (derived, not sampled) |
| 11 | `snr_db` | float64 | dB | −20..−10 | Per-sample Signal-to-Noise Ratio; co-determined by range and RCS |

---

## 2. Quantum Model Schema (4 Features)
The Variational Quantum Classifier (VQC) and precomputed fidelity quantum kernel (QSVC)
models use a subset of 4 specific physical features to match the ZZFeatureMap dimensionality
(4 qubits).

| Column | Feature | Type | Unit | Range | Description |
|---|---|---|---|---|---|
| 1 | `doppler_velocity` | float64 | m/s | −50..50 | Radial velocity, last step of an 8-step simulated track |
| 2 | `rcs_estimate` | float64 | dBsm | −20..20 | Radar cross-section proxy; smaller = stealthier |
| 3 | `micro_doppler_bandwidth` | float64 | Hz | 0..500 | Spectral spread from rotating/flapping parts — key drone discriminator |
| 4 | `micro_doppler_jitter` | float64 | — | 0..1 | Normalized micro-motion irregularity |

---

## 3. Standalone 31-Feature Schema (Option B experiment, `feature31-standalone` branch)
A raw-signal statistical model generating 31 statistical features derived from 100-sample
time-series waveforms across Amplitude, Velocity, and Energy signals. See
`artifacts/reports/FEAT31_EXPERIMENT.md` for the full build and results — **this schema lost
to the incumbent 11-feature model at -25dB and is not deployed**, kept as a documented,
reproducible standalone experiment only.

| Column | Feature | Type | Unit | Description |
|---|---|---|---|---|
| 1 | `snr_db` | float64 | dB | Signal-to-Noise Ratio |
| 2 | `amplitude_mean` | float64 | V | Mean of received signal amplitude waveform |
| 3 | `amplitude_std` | float64 | V | Standard deviation of amplitude waveform |
| 4 | `amplitude_min` | float64 | V | Minimum amplitude value |
| 5 | `amplitude_max` | float64 | V | Maximum amplitude value |
| 6 | `amplitude_range` | float64 | V | Amplitude range (V_max − V_min) |
| 7 | `amplitude_energy` | float64 | V² | Total signal energy (Σx²) |
| 8 | `amplitude_zcr` | float64 | — | Zero-Crossing Rate of amplitude around mean |
| 9 | `amplitude_dominant_freq` | float64 | Hz | Peak dominant frequency from DFT |
| 10 | `amplitude_spectral_bandwidth` | float64 | Hz | Spectral bandwidth spread from DFT |
| 11 | `amplitude_diff_var` | float64 | V² | Variance of first-difference signal (var(Δx)) |
| 12 | `velocity_mean` | float64 | m/s | Mean of radial velocity signal waveform |
| 13 | `velocity_std` | float64 | m/s | Standard deviation of velocity waveform |
| 14 | `velocity_min` | float64 | m/s | Minimum velocity value |
| 15 | `velocity_max` | float64 | m/s | Maximum velocity value |
| 16 | `velocity_range` | float64 | m/s | Velocity range (v_max − v_min) |
| 17 | `velocity_energy` | float64 | (m/s)² | Total energy of velocity signal |
| 18 | `velocity_zcr` | float64 | — | Zero-Crossing Rate of velocity around mean |
| 19 | `velocity_dominant_freq` | float64 | Hz | Peak dominant frequency from velocity DFT |
| 20 | `velocity_spectral_bandwidth` | float64 | Hz | Spectral bandwidth spread from velocity DFT |
| 21 | `velocity_diff_var` | float64 | (m/s)² | Variance of first-difference velocity signal |
| 22 | `energy_mean` | float64 | J | Mean of energy signal waveform |
| 23 | `energy_std` | float64 | J | Standard deviation of energy waveform |
| 24 | `energy_min` | float64 | J | Minimum energy value |
| 25 | `energy_max` | float64 | J | Maximum energy value |
| 26 | `energy_range` | float64 | J | Energy range (E_max − E_min) |
| 27 | `energy_energy` | float64 | J² | Total signal energy of energy waveform |
| 28 | `energy_zcr` | float64 | — | Zero-Crossing Rate of energy around mean |
| 29 | `energy_dominant_freq` | float64 | Hz | Peak dominant frequency from energy DFT |
| 30 | `energy_spectral_bandwidth` | float64 | Hz | Spectral bandwidth spread from energy DFT |
| 31 | `energy_diff_var` | float64 | J² | Variance of first-difference energy signal |

---

## 4. Physical Grounding & Derivations

* **Kinematic Trajectories:** `velocity_variance` and `acceleration` (schema 1) / the velocity
  waveform (schema 3) are derived from an Ornstein–Uhlenbeck velocity track modeling
  class-specific kinematics (e.g. drone hover-and-jerk behavior vs. smooth aircraft velocity)
  — 8 steps for schema 1, extended to 100 steps for schema 3's full waveform.
* **SNR Coupling:** SNR follows the physical radar-equation coupling form:
  SNR ∝ RCS − 40·log10(Range), mapped affinely into each schema's spec band
  (schema 1: [-20,-10]dB; schema 3: [-25,-10]dB, extended for the -25dB comparison).
* **Measurement Noise:** Noise levels on all features/waveforms scale dynamically with the
  SNR deficit, making low-SNR targets much harder to predict for both schemas.
