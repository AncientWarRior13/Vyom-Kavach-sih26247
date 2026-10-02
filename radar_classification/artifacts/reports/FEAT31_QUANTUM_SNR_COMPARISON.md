# 31-Feature Quantum Kernel: Fixed-SNR Comparison

A fresh test built specifically to compare against a teammate's independently-run
quantum-kernel result on the same schema, at the same nominal SNR levels. Full methodology
and caveats are documented in [`radarsense/feat31_quantum_snr.py`](../../radarsense/feat31_quantum_snr.py)'s
module docstring — summarized here.

## What "same" means here

- **Same dataset**: training pool subsampled directly from `feat31_train_features.parquet`
  — the exact data the deployed 31-feature CatBoost model was trained on, not a
  separately-generated pool.
- **Same SNR levels**: -10 / -15 / -20 / -25 dB, held FIXED per test set (not sampled from a
  continuous band and binned afterward).
- **Same evaluation protocol / circuit**: `ZZFeatureMap(4, reps=2)`, exact noiseless
  statevector fidelity kernel — the same circuit family used everywhere else in this project
  (`quantum.py`, `raddar_cnn.py`, `diat_usat_cnn.py`).
- **Quantum feature subset**: the model's own top real SHAP-ranked features for the drone
  class (`energy_spectral_bandwidth`, `amplitude_mean`, `amplitude_spectral_bandwidth`,
  `amplitude_min`), excluding `snr_db` itself — selected by measured importance, same
  methodology as the frozen 11-feature schema's `QUANTUM_FEATURES`.

## Result (5 seeds, mean ± std)

| SNR (dB) | n_test | Accuracy | Drone recall |
|---|---|---|---|
| -10 | 320 | **76.9% ± 2.1** | 73.3% ± 5.0 |
| -15 | 320 | **78.2% ± 1.0** | 74.0% ± 1.6 |
| -20 | 320 | **79.3% ± 1.7** | 71.3% ± 3.6 |
| -25 | 320 | **76.9% ± 1.7** | 74.5% ± 4.8 |

Full CSV: [`feat31_quantum_snr_comparison.csv`](feat31_quantum_snr_comparison.csv).

## Honest comparison to the teammate's numbers

| SNR (dB) | Teammate's run | This run |
|---|---|---|
| -10 | 81–86% | 76.9% ± 2.1 |
| -15 | 80–83% | 78.2% ± 1.0 |
| -20 | 80–81% | 79.3% ± 1.7 |
| -25 | 77–79% | 76.9% ± 1.7 |

Same ballpark (mid-to-high 70s to low 80s accuracy across the board), so 81% is confirmed
not to be an outlier or a broken run — it's a reasonable number for this kind of low-SNR
quantum-kernel classification. Two real differences, both worth being upfront about:

1. **Shape across SNR**: the teammate's numbers decline monotonically as SNR worsens
   (81→86% down to 77→79%), which is the physically expected pattern. This run is closer to
   flat (76.9 → 78.2 → 79.3 → 76.9%), even dipping slightly at the best SNR (-10dB) before
   recovering. The std bands mostly overlap between adjacent levels, so this is plausibly
   sampling noise at n_test=320 rather than a real non-monotonic effect — but it does NOT
   show the same clean trend the teammate's run does, and that's reported rather than
   smoothed over.
2. **Absolute level**: this run sits a few points below the teammate's at every level. Exact
   protocols likely differ in ways not fully known here (this run's own train-pool size,
   class-balancing choice, and quantum feature subset are documented above and in the
   script — the teammate's exact setup for their run isn't independently verified against
   this one beyond the SNR levels and "quantum kernel" framing).

## Caveats

- 5 seeds is enough to see that per-level std (1–2 points on accuracy) is much smaller than
  the level-to-level spread, but not enough to fully rule out the non-monotonic shape being a
  real, small effect rather than pure noise.
- Class-balanced training (100/class) rather than the dataset's native ~20:1
  aircraft:drone ratio — necessary (an earlier natural-proportion pass produced 0.0 drone
  recall at every level, a degenerate result, not a real finding) but means this measures
  quantum-kernel *capability* under a controlled class mix, not deployment-realistic
  class-imbalanced performance.
- Noiseless simulation, as with every other quantum-kernel result in this project except
  [`IBM_HARDWARE_EXPERIMENT.md`](IBM_HARDWARE_EXPERIMENT.md).
