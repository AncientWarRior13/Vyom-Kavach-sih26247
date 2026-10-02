# 42-Feature Combined Model: Explainability (SHAP + LIME) and Calibration

Fills the gap flagged when the combined model was first reported: it's the star result (99%+
accuracy, 100% drone recall at extreme noise) but was the only deployed model with zero
interpretability or calibration work — the 11-feature and 31-feature models both have SHAP
summaries, reliability diagrams, and robustness sweeps.

## The core question: does it actually use both feature sets?

`FEAT42_COMBINED_EXPERIMENT.md` left this open. Answered here with **two independent
explanation methods**, not just one — SHAP (exact game-theoretic tree attribution) and LIME
(a genuinely different algorithm: a locally-weighted linear surrogate fit on perturbed
samples around each instance). Two different methods agreeing is real corroborating evidence
the finding isn't a SHAP-specific artifact.

| | SHAP | LIME (aggregated over 20 drone detections) |
|---|---|---|
| Physics (11-feature schema, 10 cols) | 38.8% | 42.7% |
| Waveform-stat (31-feature schema, 30 cols) | 57.6% | 55.2% |
| `snr_db` (shared) | 3.5% | 2.1% |

**Both methods agree: the model genuinely draws on both feature sets.** Physics features
carry more importance per-feature than their column count would suggest (38.8–42.7% of total
importance from only 10 of 41 columns — roughly 4% per feature vs. waveform-stat's ~1.8% per
feature), consistent with the physics schema's own strong standalone track record on drone
recall specifically. But waveform-stat features aren't just riding along — they carry the
majority of total importance and dominate the top-10 list (7 of 10 features).

**Top-10 feature overlap between the two methods: 7/10** (`range_m`, `micro_doppler_bandwidth`,
`energy_spectral_bandwidth`, `energy_zcr`, `amplitude_zcr`, `amplitude_mean`, `velocity_zcr`).
This is a strong agreement rate for two structurally different explanation methods — not
identical (LIME surfaces `velocity_dominant_freq`, `velocity_variance`, `amplitude_spectral_
bandwidth` where SHAP surfaces `altitude_m`, `velocity_energy`, `velocity_max` instead), which
is expected (LIME explains local linear behavior around specific instances; SHAP attributes
exact tree-ensemble output globally) — but the core finding replicates across both.

## Calibration: already well-calibrated, isotonic refinement helps a little more

| Class | Brier (raw → calibrated) | ECE (raw → calibrated) |
|---|---|---|
| aircraft | 0.0041 → 0.0038 | 0.0057 → 0.0021 |
| anomaly | 0.0066 → 0.0066 | 0.0058 → 0.0031 |
| bird | 0.0015 → 0.0015 | 0.0015 → 0.0009 |
| drone | 0.0013 → 0.0012 | 0.0014 → 0.0009 |

**Notably better raw calibration than the 31-feature model** — feat42's raw ECE (0.0014–0.0058
across classes) is roughly 3–10× lower than feat31's raw ECE (0.0143–0.0170 across the same
classes, from `feat31_calibration_report.json`). Isotonic calibration still helps (ECE roughly
halves across all four classes), so it's applied and saved as `feat42_calibrators.joblib`, but
the model's raw probability outputs were already meaningfully more trustworthy than either
individual schema's model before any post-hoc correction.

## Honest interpretation

- The combined model's strong performance is not an artifact of one schema dominating — real,
  corroborated-by-two-methods evidence that fusing physics and waveform-statistics genuinely
  produces a richer decision function, consistent with the "independent noise realizations of
  the same latent quantities" mechanism proposed in `FEAT42_COMBINED_EXPERIMENT.md`.
- The better raw calibration (vs. feat31) is a new, previously unmeasured advantage of the
  combined model worth factoring into any deployment decision — not just "more accurate," but
  "more trustworthy probability outputs out of the box."

## Caveats

- LIME's aggregate ranking is based on 20 drone detections (num_samples=1000 perturbations
  each) — a modest sample; a larger aggregate would tighten the ranking but the qualitative
  agreement with SHAP is already clear at this scale.
- LIME discretizes continuous features into quartile bins by default; matching its bin
  descriptions back to the underlying feature name (not the exact bin boundary) is how the
  aggregate ranking was built — the *feature* attribution is solid, the specific threshold
  values in LIME's bins are a separate, less load-bearing detail.
- Calibration was fit once on the validation split, not cross-validated.

## Artifacts

- `radarsense/feat42_explain.py`, `radarsense/feat42_calibrate.py`
- `artifacts/reports/feat42_shap_report.json` (SHAP + LIME combined), `feat42_calibration_report.json`
- `artifacts/plots/feat42_shap_summary_drone.png`, `feat42_shap_summary_global.png`,
  `feat42_shap_waterfall_drone.png`, `feat42_lime_single_drone.png`, `feat42_lime_aggregate.png`,
  `feat42_calibration_reliability.png`
- `artifacts/models/feat42_calibrators.joblib`
