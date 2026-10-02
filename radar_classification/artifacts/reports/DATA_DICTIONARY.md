# RadarSense Dataset — Data Dictionary & Datasheet

**Dataset:** `artifacts/data/radar_full.parquet` (+ CSV), splits in `radar_{train,val,test}.parquet`
**Samples:** 52,000 | **Imbalance:** 20:1 (aircraft:drone) | **SNR band:** −20 to −10 dB
**Generator:** `radarsense/simulator.py`, seed 42 — fully reproducible with `python -m radarsense.simulator`

## Motivation

Simulated X-band (10 GHz) radar returns for low-signature aerial target classification,
built as ground truth for the RadarSense pipeline (Quant-A-Thon'26). The minority target
class is **drone** (2.5% of samples) to reflect the operational reality that drone
incursions are rare relative to routine air traffic and biological clutter.

## Composition

| Class | Count | Share | Signature design |
|---|---|---|---|
| aircraft | 26,000 | 50.0% | High RCS (≈ +12 dBsm), fast smooth kinematics, narrow engine-modulation micro-Doppler |
| bird | 20,800 | 40.0% | Low RCS (≈ −16 dBsm), 2–15 Hz wing-beat micro-Doppler (10–80 Hz spread), irregular path |
| anomaly | 3,900 | 7.5% | Mixture: balloons (static, moderate RCS), rotor swarms (broad micro-Doppler), ultralights |
| drone | 1,300 | 2.5% | Low RCS (≈ −14 dBsm), hover-capable, jerky kinematics, broad rotor blade-flash micro-Doppler (160–500 Hz) |

## Feature schema (FROZEN — contract with Pair 2; column order matters)

| # | Feature | Type | Unit | Range | Description |
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
| 11 | `snr_db` | float64 | dB | −20..−10 | Per-sample SNR; also usable as a stress-test control variable |
| 12 | `label` | str | — | 4 classes | Ground truth: aircraft / anomaly / bird / drone |

## Generation process (physics grounding)

- **Kinematics:** each sample is the endpoint of an 8-step (0.1 s/step) Ornstein–Uhlenbeck
  radial-velocity trajectory with class-specific dynamics (drone: mean-reverting to hover
  with jerk events; aircraft: near-constant; bird: OU + sinusoidal wing-beat).
  `velocity_variance` and `acceleration` are computed from the track, not drawn independently.
- **SNR coupling:** SNR follows the radar-equation form RCS_dB − 40·log₁₀(R), affinely
  mapped into the −20..−10 dB spec band. Range and RCS therefore physically co-determine SNR.
- **Noise model:** measurement noise on every feature scales linearly with SNR deficit
  (zero extra noise at −10 dB, maximum at −20 dB), making low-SNR samples genuinely harder.
- **RCS values** parameterized from open literature ranges (small UAS ≈ −20..−8 dBsm,
  birds ≈ −20..−10 dBsm, light/commercial aircraft ≈ +5..+20 dBsm).

## Splits

Stratified 70/15/15 train/val/test (36,400 / 7,800 / 7,800), stratified on `label`,
seed 42. Verified: zero duplicate rows, zero cross-split overlap
(`artifacts/reports/data_quality_report.json`).

## Known limitations (state these in Q&A unprompted)

1. Statistical simulator — models statistical radar properties, **not full
   electromagnetic physics**. Real-sensor generalization is unverified, flagged as future work.
2. Micro-Doppler is summarized to two scalars (bandwidth, jitter); a real system would
   have the full spectrogram.
3. Class-conditional distributions are stationary; no weather, multipath, or terrain masking.
4. The anomaly class is a designed mixture (balloons/swarms/ultralights), not open-world.

## Maintenance

Schema is validated by `python -m radarsense.validate_schema <file>`; run it against any
data claiming this schema before feeding the model. Changes to the schema after Pair 2
handoff are prohibited.
