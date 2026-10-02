# Does the Combined Model Resist Micro-Doppler Masking Better?

`THREAT_MODEL.md`'s one clean vulnerability in the 11-feature model: physically shrouding or
slowing rotors collapses drone recall from 91.8% to 12.3% at strong attenuation, because
`micro_doppler_bandwidth` is the dominant SHAP feature. The 42-feature combined model has a
second, independently-noised feature set derived from the same underlying physics — does that
redundancy actually help against an attack on the shared root cause, or is it useless because
both feature sets trace back to the same signal?

## Why this has to be tested fairly, not assumed

The 31-feature schema's amplitude/energy waveform statistics are themselves driven by
`micro_doppler_bandwidth`/`micro_doppler_jitter` (`feat42_simulator.py`: modulation frequency =
`md_bw/4`, modulation depth = `md_jit`). Masking only the 11-feature physics columns while
leaving the 31-feature waveform-stat columns untouched would make the combined model look more
robust than it really is — a real countermeasure (shrouding rotors) would suppress both. This
test regenerates drone instances with the *same* attenuated `md_bw`/`md_jit` fed consistently
into both feature derivations (`_physics11` and `_waveforms31`, imported unmodified), at the
same attenuation levels `robustness.py` used for the 11-feature model (×0.7, ×0.5, ×0.3).

## Result: genuinely more robust at every level

| Attenuation | 11-feature model (reported) | 42-feature model (measured) |
|---|---|---|
| Unperturbed | 91.8% | **98.5%** |
| ×0.7 | 76.4% | **93.1%** |
| ×0.5 | 51.8% | **63.1%** |
| ×0.3 | 12.3% | **16.8%** |

The combined model wins at every single attenuation level, including the severe ×0.3 case —
not solved (16.8% recall is still a real degradation from 98.5% baseline), but meaningfully
less severe than the 11-feature model's collapse to 12.3%. The gap is largest at moderate
attenuation (×0.7: +17pt, ×0.5: +11pt) and narrows at the most extreme level, consistent with
both feature sets sharing the same ultimate root cause — enough independent noise survives at
moderate masking to help substantially, but heavy enough masking eventually degrades both.

## Honest interpretation

This supports the "independent noise realizations of the same latent quantity" mechanism
already proposed in `FEAT42_COMBINED_EXPERIMENT.md`: even though the 31-feature waveform
statistics are driven by the same `md_bw`/`md_jit` values as the physics features, their
*independent* noise realization means they don't degrade in lockstep — when the physics-side
signal is heavily attenuated, the waveform-stat side still carries some residual, decorrelated
information the model can fall back on. That's a genuine, physically-grounded robustness
benefit from feature-set fusion, not just an accuracy benefit.

It is **not a fix for the underlying vulnerability** — a sufficiently strong real-world
countermeasure would still degrade both. But deploying the 42-feature model instead of (or
alongside) the 11-feature model is a concrete, already-available mitigation direction that
doesn't require building the kinematic-fallback system `THREAT_MODEL.md` proposes — it's
already sitting in this repository, tested and quantified.

## Caveats

- Compared against the 11-feature model's already-reported `robustness.py` numbers, not a
  freshly matched run on identical underlying drone instances — the two use different specific
  attack mechanics (post-hoc tabular attenuation for the 11-feature model vs. source-level
  attenuation before feature extraction for the 42-feature model), though both attenuate the
  same physical quantities by the same factors, making the comparison meaningful even if not
  bit-identical in mechanism.
- Single run per attenuation level, 1,000 drone samples each, not repeated across seeds.
- Only the micro-Doppler masking attack was re-tested here — RCS suppression and jamming
  (which don't evade the 11-feature model either) were not re-checked against the 42-feature
  model.

## Artifacts

- `radarsense/feat42_masking_robustness.py`
- `artifacts/reports/feat42_masking_robustness_results.json`
