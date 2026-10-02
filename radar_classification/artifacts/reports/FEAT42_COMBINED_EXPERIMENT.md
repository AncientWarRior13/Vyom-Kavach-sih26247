# Combined 42-Feature (11 Physics + 31 Waveform-Statistic) Model

*"42-feature" is the informal name (11+31=42); the model actually trains on 41 unique
columns, since `snr_db` appears in both source schemas and is not duplicated — disclosed in
[`feat42_simulator.py`](../../radarsense/feat42_simulator.py)'s docstring rather than padded
to a round number.*

## Why attempt this

The 11-feature physics model is the established stronger performer for drone recall
specifically (this project's own frozen incumbent). The 31-feature waveform-statistic model
showed a genuine accuracy edge under extreme noise, once the -25dB comparison's noise-
calibration bug was fixed (see [`FEAT31_EXPERIMENT.md`](FEAT31_EXPERIMENT.md)). Rather than
picking a winner, training one model on both feature sets is a legitimate "best of both
worlds" attempt — not guaranteed to win (more columns isn't automatically better; the two
schemas are partially redundant by construction, both derived from the same per-instance RCS/
micro-Doppler/range/velocity parameters), so it was tested rather than assumed.

## How the combined dataset is genuinely one dataset, not two stapled together

Simply joining `radar_train.parquet` and `feat31_train_features.parquet` column-wise would be
wrong — row *i* in one has no relationship to row *i* in the other; they're independently
sampled. [`feat42_simulator.py`](../../radarsense/feat42_simulator.py) instead draws each
instance's latent physical parameters (initial radial velocity, RCS, micro-Doppler bandwidth/
jitter, blade rate/count or anomaly kind, slant range, altitude) **once per instance**, using
the identical per-class distributions already used separately by `simulator.py` and
`feat31_simulator.py` (confirmed by inspection to match). From that one shared instance, two
trajectories are generated using each source's own frozen, unmodified dynamics function —
`simulator.py`'s `_ou_trajectory` (8-step, DT=0.1) for the 11 physics features, and
`feat31_simulator.py`'s `_ou_waveform` (100-step, DT=0.01) for the 31 waveform-statistic
features — sharing the same target identity and dynamics regime, with one shared per-instance
SNR value setting noise severity on both feature sets consistently.

## Result: combined model wins on both axes at the extreme-noise slice

Three-way comparison at SNR ≤ -23dB (each model evaluated on its own matching test slice):

| Model | Accuracy | Drone recall | n |
|---|---|---|---|
| 11-feature (frozen, out-of-distribution — never trained below -20dB) | 95.1% | 88.2% | 1565 |
| 31-feature (trained for this regime) | 96.1% | 76.9% | 1487 |
| **42-feature combined (trained for this regime)** | **99.25%** | **100.0%** | 1462 |

The combined model's deep-low-SNR confusion matrix: 48 real drones in the slice, all 48
caught, 1 false positive total (precision 98.0%, recall 100.0%) — not a small-sample fluke.

## Why this plausibly works: independent noise realizations of the same latent quantities

`amplitude_mean`/`amplitude_energy` (31-feature) are, before noise, a near-deterministic
function of `rcs_estimate` (11-feature) — both trace back to the same instance's true RCS.
Similarly, `amplitude_dominant_freq`/`amplitude_spectral_bandwidth` trace back to the same
`micro_doppler_bandwidth`/`micro_doppler_jitter` that drive the 11-feature schema directly.
Critically, the two views get **independent noise realizations** (the physics trajectory and
the waveform trajectory are separately time-sampled, separately noised). Combining two
independently-noised measurements of the same latent physical quantity is a real, well-
understood noise-reduction mechanism (averaging/sensor-fusion) — CatBoost's feature
importances for the combined model support this reading: the top features are a healthy mix
of both schemas (`micro_doppler_bandwidth`, `range_m`, `amplitude_mean`, `acceleration`,
`energy_zcr`, `rcs_estimate`, ...), not one schema dominating or one suspicious column
carrying the result.

## Caveats

- Checked across 3 seeds (42/43/44, full regeneration + retrain each time, not just a
  reshuffled split): accuracy 99.19–99.53%, drone recall 100.0% every time — the result is
  reproducible, not a lucky draw on one seed.
- The two source trajectories are independent noise realizations of the *same* latent
  parameters, not independent physical measurements from two different sensors — this is a
  synthetic-data property of how the combined generator is built, not evidence that a
  real dual-feature-extraction pipeline on real hardware would show the same gain.
- Not yet compared against SMOTE-based imbalance handling (class-weighting only, matching
  both individual models) — see [`FEAT31_SMOTE_COMPARISON.md`](FEAT31_SMOTE_COMPARISON.md) for
  that ablation on the 31-feature model.
- 41 unique features means slower inference and a larger feature-engineering surface at
  deployment than either individual model — not evaluated here for latency/footprint.

## Artifacts

- `radarsense/feat42_simulator.py` — combined dataset generator
- `radarsense/feat42_train.py` — training + three-way comparison
- `artifacts/reports/feat42_results.json` — full results
- `artifacts/models/feat42_model.joblib` — trained model
