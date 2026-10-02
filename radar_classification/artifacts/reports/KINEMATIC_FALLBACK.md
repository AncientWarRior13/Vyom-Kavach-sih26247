# Kinematic Fallback Classifier: A Second Mitigation Direction

`THREAT_MODEL.md` named two mitigation directions for micro-Doppler masking. `FEATURE_DROPOUT_TRAINING.md`
tested the first (retrain to reduce reliance on the vulnerable feature). This tests the second,
a genuinely different mechanism — detection + routing rather than retraining: a second
classifier trained on the 9 non-micro-Doppler features (structurally cannot be fooled by
masking, since it never sees `micro_doppler_bandwidth`/`micro_doppler_jitter` at all), blended
in with weight proportional to a "suspicion score" when the primary model's micro-Doppler
reading looks anomalously low.

## A real bug caught and fixed before trusting the result

First pass calibrated the suspicion cutoff on `micro_doppler_bandwidth`'s 20th percentile
across the *full* training population — which came out to exactly 0.0, because aircraft (50%
of the population) naturally has near-zero micro-Doppler (a residual engine line, not a broad
spread), pulling the overall percentile down to a degenerate value. The detector never
triggered (0.0% of samples), silently doing nothing while looking like it ran successfully.
Fixed by calibrating on the **drone class specifically** — "unusually low for a drone" is the
actual question, not "unusually low overall." Left visible here rather than silently corrected,
consistent with how this project handles every other self-caught bug.

## Result: real but modest, and not free

| Attenuation | Primary only | Blended with fallback |
|---|---|---|
| Unperturbed (overall accuracy) | 97.59% | 96.46% (**−1.13pt**) |
| ×0.7 | 76.4% | 79.0% |
| ×0.5 | 51.8% | 58.0% |
| ×0.3 | 12.3% | **29.2%** |

At the severe ×0.3 case, recall more than doubles (12.3%→29.2%) — a real improvement, but well
below feature-dropout training's 64.1% at the same attack level, and at a real accuracy cost
this project's other mitigations mostly avoided (feature-dropout training cost 3.6 points of
drone-specific recall; this costs 1.1 points of *overall* accuracy across all four classes,
since blending triggers on 95.6% of all test samples, not narrowly on masked drones).

**One genuinely good sign**: naturally-quiet-rotor real drones (bottom 20% of natural
micro-Doppler bandwidth, unperturbed) actually improve under blending — 76.9%→82.1% recall.
The fallback isn't just reacting to the specific attack pattern; it generalizes to a real,
naturally-occurring edge case too.

## Honest interpretation

This is a weaker single mitigation than feature-dropout training for this specific attack, and
it costs more (broader accuracy hit vs. a narrower recall-specific one). It's not a wasted
effort, though — it's a structurally different defense (an ensemble/routing approach vs. a
training-data approach), and the fact that it also helps genuinely quiet-rotor drones suggests
it's capturing something real about model uncertainty near the micro-Doppler decision boundary,
not just pattern-matching the attack. Combining it with feature-dropout training (not tested
here) is a plausible way to get both effects — untested.

The 95.6%-of-samples-trigger-blending number suggests the suspicion cutoff and blend curve
(`blend_power=2.0`, 20th percentile) weren't well-tuned — a proper sweep over the cutoff
percentile and blend sharpness would very plausibly find a configuration with a better
robustness/accuracy trade-off than this first pass. Not done here due to time; noted as the
clear next step if this direction is pursued further.

## Caveats

- No hyperparameter sweep (suspicion percentile, blend power) — first reasonable choice tested,
  not tuned.
- Single run, not repeated across seeds.
- Not combined with feature-dropout training or the 42-feature model's fusion-based robustness
  — testing all combinations was out of scope here.
- Fallback model saved separately (`kinematic_fallback_model.joblib`); the blending logic
  itself lives only in this script, not wired into `api/server.py` or `api/predict.py`.

## Artifacts

- `radarsense/kinematic_fallback.py`
- `artifacts/reports/kinematic_fallback_results.json`
- `artifacts/models/kinematic_fallback_model.joblib`
