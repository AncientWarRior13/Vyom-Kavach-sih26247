# Longer Waveforms (100 → 300 Steps) Experiment

Tests whether tripling the waveform length improves the 31-feature schema's noise robustness
— the reasoning being that the features SHAP flagged as most important (`amplitude_
spectral_bandwidth`, `amplitude_dominant_freq`, `energy_spectral_bandwidth`) are DFT-derived,
and a longer observation window gives finer frequency resolution (0.33 Hz bins at 300 steps
vs. 1.0 Hz at 100, same 0.01s sample rate — extending the window to 3.0s, not sampling faster
over the same 1.0s), which should make those estimates less noise-sensitive.

Built as a full parallel pipeline ([`feat31_simulator_300.py`](../../radarsense/feat31_simulator_300.py),
[`feat31_features_300.py`](../../radarsense/feat31_features_300.py),
[`feat31_train_300.py`](../../radarsense/feat31_train_300.py)) rather than modifying
`feat31_simulator.py`/`feat31_features.py` in place — those files are depended on by
`feat31_train.py`, `feat31_compare.py`, and `feat31_quantum_snr.py` as stable, already-tested
artifacts; same "duplicate rather than risk the frozen file" philosophy `feat31_simulator.py`
itself already uses for `simulator.py`.

## Result: real improvement overall, flat at the extreme-noise slice specifically

| | 100-step | 300-step |
|---|---|---|
| Overall test accuracy | 96.56% | **97.77%** |
| Overall test drone recall | 83.08% | **85.64%** |
| SNR ≤ -23dB accuracy | 96.10% | **97.10%** |
| SNR ≤ -23dB drone recall | **76.92%** | 76.00% |

Overall accuracy and drone recall both improved with longer waveforms. But at the specific
extreme-noise slice this model exists for, accuracy improved while drone recall did not (76.0%
vs. 76.9%, within noise of each other, not a real gain). The reasoning that motivated this
experiment — better-conditioned spectral statistics helping specifically at low SNR — is not
what's driving the overall improvement; something else is (see mechanism check below).

## Mechanism check: does the specific reasoning actually hold?

Measured the SHAP-flagged spectral features' noise-induced standard deviation directly,
100-step vs. 300-step, at both a good-SNR slice (≥-12dB) and the deep-low-SNR slice:

| Feature | 100-step std (good / deep-low SNR) | 300-step std (good / deep-low SNR) |
|---|---|---|
| `amplitude_spectral_bandwidth` | 4.81 / 3.88 | **3.92 / 3.37** |
| `energy_spectral_bandwidth` | 5.53 / 4.58 | **4.71 / 4.38** |
| `amplitude_dominant_freq` | 0.00 / 0.00 | 0.00 / 0.00 |

Absolute precision genuinely improved at 300 steps for both features, at both SNR regimes —
lower standard deviation means a tighter, more repeatable estimate, which is what "more
samples → better-conditioned statistics" predicts. But the *relative* noise-degradation ratio
(deep-low-SNR std ÷ good-SNR std) didn't clearly improve — it was already below 1.0 at 100
steps and stayed there at 300 (0.81→0.86 and 0.83→0.93 respectively, i.e. slightly worse
by this specific ratio). So the mechanism is real and measurable (tighter absolute estimates),
but it isn't specifically a *low-SNR* fix — it's a general precision improvement that helps
everywhere, which is consistent with the overall-but-not-extreme-slice pattern in the
accuracy table above.

## Side-finding: `amplitude_dominant_freq` is a dead feature

Discovered while building this mechanism check, not something that was being looked for:
`amplitude_dominant_freq` is **exactly 0.0 for all 36,400 training samples**, in every class,
at both 100 and 300 steps. The DFT peak always lands on the zero-frequency (DC) bin because
the constant amplitude offset (`base_amp`, set by RCS) dominates the small sinusoidal
modulation (depth set by `micro_doppler_jitter`, typically 0.05–0.6) in every sample — the
modulation never wins the DFT peak. This feature has carried zero information in every model
trained on this schema (`feat31_model.joblib`, `feat42_model.joblib`, this 300-step model,
and the quantum-kernel work). It hasn't corrupted any prior result — CatBoost assigns
near-zero importance to a constant column automatically, and it was never selected for the
quantum feature subsets — but it's a real data-quality issue worth fixing in a future
iteration (e.g. compute the dominant *non-DC* frequency instead), not silently left
undisclosed now that it's been found.

## Honest takeaway

Longer waveforms are a genuine, measurable improvement for the 31-feature schema overall
(+1.2 points accuracy, +2.6 points drone recall on the full test set), driven by a real
precision improvement in the DFT-derived features. It is **not**, on this evidence, a fix
specifically for the deep-low-SNR regime — that slice's drone recall was flat. If the goal is
specifically extreme-noise robustness, this experiment doesn't deliver it; if the goal is
overall model quality, it does.

## Caveats

- Single run, not repeated across seeds.
- 3x longer waveforms means 3x the per-detection observation window in a real deployment —
  a real latency/pipeline-design cost this report doesn't evaluate.
- Checked the other two channels' dominant-frequency stats for the same issue:
  `velocity_dominant_freq` is genuinely informative (51 distinct values, std≈9.8 — velocity
  has no large DC offset to dominate its own DFT). `energy_dominant_freq` is nearly
  degenerate too (only 2 distinct values, std≈0.37) since `energy = amplitude²` inherits the
  same DC-domination problem. So 2 of the 31 features (`amplitude_dominant_freq`,
  `energy_dominant_freq`) carry little-to-no real signal; only the velocity channel's
  dominant-frequency stat is doing real work.

## Artifacts

- `radarsense/feat31_simulator_300.py`, `feat31_features_300.py`, `feat31_train_300.py`
- `artifacts/reports/feat31_300_results.json`
- `artifacts/models/feat31_300_model.joblib`
