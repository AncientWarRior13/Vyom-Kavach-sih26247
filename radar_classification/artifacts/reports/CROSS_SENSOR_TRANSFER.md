# Cross-Real-Sensor Zero-Shot Transfer

The one genuinely untested question left in this project: every real-sensor result elsewhere
is either **within-sensor** (train and test on the same sensor — `diat_usat_bridge.py`'s own
variant-holdout retrain, `raddar_experiments.py`'s own retrain-and-test) or **sim-to-real
zero-shot** (the frozen simulated-trained model tested on real data). Neither of those tests
whether a model that has only ever seen one real sensor's actual noise, calibration, and
physics characteristics generalizes to a **different real sensor**, with nothing in between.

## Setup

All four real-sensor bridges already in this project map into the identical 11-feature
schema, which makes this test possible without building anything new at the data layer:

| Sensor | Bridge | Physics | Classes available |
|---|---|---|---|
| SAAB SIRS 1600 | `real_data_bridge.py` | 77GHz FMCW | drone / bird / anomaly |
| RAD-DAR | `raddar_experiments.py` | 8.75GHz FMCW | drone / anomaly |
| TSMS-Drone | `tsms_experiments.py` | 2.5GHz CW | drone / anomaly |
| DIAT-uSAT | `diat_usat_bridge.py` | X-band CW | drone / bird / aircraft / anomaly |

Drone-vs-anomaly is the only task all four share (RAD-DAR and TSMS have neither bird nor
aircraft), so that's the test: for each sensor, one CatBoost model trained on an 80% split of
*that sensor's* real data, then evaluated — zero retraining, zero fine-tuning — on (a) its
own held-out 20% (in-distribution reference) and (b) every other sensor's full real data
(cross-sensor). Using the identical trained model for both is what makes the gap a fair
comparison.

## Result: transfer degrades substantially, but the story is more nuanced than "it fails"

In-distribution, every sensor's model is strong (85–99% accuracy). Naive cross-sensor
transfer (0.5 threshold) looks much worse — average accuracy drops 25 to 63 points depending
on source sensor:

| Source sensor | In-distribution acc | Mean cross-sensor acc (0.5 threshold) | Mean cross-sensor acc (oracle threshold) | Mean cross-sensor PR-AUC |
|---|---|---|---|---|
| SAAB | 99.2% | 73.7% | 78.4% | 0.749 |
| RAD-DAR | 88.9% | **26.4%** | 54.1% | 0.802 |
| TSMS | 85.1% | 52.8% | 67.2% | 0.544 |
| DIAT-uSAT | 90.2% | 44.4% | 58.6% | 0.617 |

**The naive-threshold number alone overstates the failure.** Re-evaluated at each target
sensor's own optimal threshold (an oracle diagnostic — not deployable, since a real zero-shot
setting has no target labels to pick it with, but useful for separating two different
failure modes):

- **Recalibration recovers most of the gap** when the underlying PR-AUC is already high.
  RAD-DAR→SAAB: 29.6% → 73.7% accuracy once re-thresholded (PR-AUC was 0.972 the whole time —
  the model's drone/anomaly *ranking* transferred almost perfectly; the fixed 0.5 cutoff was
  just badly miscalibrated for SAAB's probability distribution, not the model itself).
- **Recalibration barely helps** when the PR-AUC itself is low. RAD-DAR→TSMS: 19.8% → 22.9%
  even at the oracle threshold (PR-AUC 0.170 — barely better than random ranking).
  DIAT-uSAT→RAD-DAR: 29.0% → 29.0%, unchanged (PR-AUC 0.332). These are genuine distributional
  mismatches, not calibration artifacts — no threshold choice fixes a model that isn't
  ranking the two classes correctly in the first place.

So "does it generalize across sensors" doesn't have one answer — it depends on *which*
sensor pair, and specifically on whether the miscalibration is in the decision threshold
(fixable, cheaply, without ever touching model weights) or in the underlying feature-space
separation the model learned (not fixable without retraining or domain adaptation).

## Honest interpretation

- This is a real negative-leaning result, consistent with the RAD-DAR CNN replication finding
  elsewhere in this project (`RADDAR_CNN_REPLICATION.md`): patterns learned on one real sensor
  do not automatically hold on another. Zero-shot cross-sensor deployment of any single-sensor
  model in this project would be unreliable without either target-sensor recalibration (best
  case) or retraining (necessary in the worst pairs).
- SAAB is the most "central" sensor — its model transfers best to others (mean PR-AUC 0.749),
  and other sensors' models transfer to it best too (RAD-DAR→SAAB has the highest single-pair
  PR-AUC, 0.972). Plausibly because it's by far the largest, most diverse real dataset here
  (68K samples vs. 2–17K for the others), giving its model the most robust decision boundary.
- RAD-DAR and TSMS are the hardest targets/sources in most pairings — both are CW/FMCW sensors
  with comparatively small real datasets and (per their own bridge docstrings) coarser or more
  assumption-laden feature calibration than SAAB's radiometrically-calibrated pipeline.
- Zero-shot is a genuinely hard bar, and a more realistic deployment question — how much a
  small labeled calibration set recovers — is answered separately in
  [`CROSS_SENSOR_FEW_SHOT.md`](CROSS_SENSOR_FEW_SHOT.md): just 50 labeled target examples
  recover roughly 21 accuracy points on average (48.8%→69.8%), a substantial but incomplete
  recovery.

## Caveats

- Each sensor's `snr_db` is independently affine-mapped into the schema's band per its own
  bridge's documented "indicative only" calibration — not a physically identical quantity
  across sensors. This test exercises all 11 schema features, not `snr_db` alone, so it isn't
  invalidated by this, but it's a real limit on how literally "the same feature means the same
  thing across sensors" should be read. **Resolved, not just flagged**: tested directly in
  [`CROSS_SENSOR_CALIBRATION_ABLATION.md`](CROSS_SENSOR_CALIBRATION_ABLATION.md) — dropping
  `snr_db` and `rcs_estimate` (the two features with confirmed inconsistent cross-sensor
  calibration) improves mean cross-sensor accuracy by 10.7 points (49.3%→60.0%), confirming
  calibration inconsistency was a real, measurable, but partial contributor to the transfer
  failure — most of the gap remains even after removing them, so the rest is plausibly genuine
  physics/hardware domain shift.
- `azimuth_deg`/`altitude_m` are constant 0.0 in most bridges (documented as unmeasurable at
  each source) — CatBoost assigns them near-zero importance automatically, same as the dead
  `amplitude_dominant_freq` feature found in `FEAT31_300STEP_EXPERIMENT.md`; noted for
  completeness, not a new finding.
- Single run per pair, not repeated across seeds or alternate 80/20 splits.
- The oracle threshold is a diagnostic to separate failure modes, explicitly not a deployable
  number — it uses target-sensor labels a real zero-shot deployment would not have.

## Artifacts

- `radarsense/cross_sensor_transfer.py`
- `artifacts/reports/cross_sensor_transfer_results.json` — full 4×4 matrix, both threshold
  regimes, all metrics
