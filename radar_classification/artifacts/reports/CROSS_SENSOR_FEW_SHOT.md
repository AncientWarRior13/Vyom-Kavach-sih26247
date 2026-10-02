# Few-Shot Cross-Sensor Adaptation: A More Realistic Deployment Question

`CROSS_SENSOR_TRANSFER.md` found pure zero-shot transfer (train on one real sensor, test on
another, zero target-domain information) degrades substantially. That's a harder bar than most
real deployments actually face — a new sensor install can usually collect at least a small
labeled calibration set. This asks the more realistic question: how many labeled target
examples does it take to recover most of the gap?

## Method

Reuses `cross_sensor_transfer.py`'s exact data loading, model training, and metrics. Each
sensor's data is split once into a 200-sample adaptation pool (source of the few-shot examples)
and a final-eval pool (never touched by adaptation, used at every shot budget so the comparison
stays apples-to-apples as the budget grows). Two strategies tested at each budget (0/10/20/50
shots, all 12 source→target pairs):

1. **Threshold-only recalibration** — pick the best decision threshold using only the small
   few-shot sample (a realistic version of the original report's oracle-threshold diagnostic,
   which used the *full* target set; this uses a genuinely small, deployable one).
2. **Full retraining** — mix the few-shot target examples into the source training data and
   retrain from scratch.

## Result: 50 labeled examples recover meaningfully — not fully, but substantially

| Shots | Zero-shot (0) | Threshold-only | Full retrain |
|---|---|---|---|
| Accuracy | 48.8% | — | — |
| 10 | | 58.5% | 59.6% |
| 20 | | 57.2% | 63.2% |
| 50 | | 57.6% | **69.8%** |

Full retraining with just 50 labeled examples brings average cross-sensor accuracy from 48.8%
to 69.8% — a real, substantial 21-point recovery, though still well below the 85–99%
in-distribution numbers each sensor's own model achieves on its own data.

**The same accuracy-vs-recall trade-off pattern from the classical+quantum ensemble work shows
up here too**: threshold-only recalibration plateaus quickly on accuracy (~57–58% regardless of
shot count) but gives *higher* drone recall than full retraining at every budget (72–78% vs.
62–66%). Full retraining keeps improving accuracy with more shots but doesn't catch up on
recall. Neither strategy dominates the other — which metric matters more depends on whether the
deployment cares more about overall correctness or specifically not missing drones.

## Honest interpretation

Zero-shot transfer is a genuinely hard bar, and this shows why it doesn't need to be the bar a
real deployment is held to: a small, realistic calibration effort (50 labeled detections from
the new sensor) recovers a meaningful fraction of the cross-sensor gap. This is a more useful
answer for an actual deployment plan than the zero-shot number alone — "this doesn't generalize
with zero labels, but does noticeably better with a modest calibration set" is a complete,
actionable finding, not a excuse for the zero-shot result.

It is **not full recovery** — even at 50 shots, cross-sensor accuracy (69.8%) is well below
in-distribution performance (85–99%). More shots would likely help further (untested here,
budget capped at 50); so would the strategies from `CLASSICAL_QUANTUM_ENSEMBLE.md`'s
calibration work, not combined with this.

## Caveats

- Aggregated across all 12 pairs, which include both "easy" pairs (recoverable by
  recalibration alone, per the original report's oracle-threshold diagnostic) and "hard" pairs
  (genuine distributional mismatch) — the average masks real per-pair variation; some pairs
  likely recover much more than 21 points, others much less.
- Single run, single seed for shot sampling.
- Shot budget capped at 50 — the recovery curve was still rising at that point (59.6%→63.2%→69.8%
  for full retrain), so more shots would plausibly recover further; not tested.
- `snr_db`'s cross-sensor calibration inconsistency (each bridge's own "indicative only"
  mapping, disclosed in `CROSS_SENSOR_TRANSFER.md`) applies here too — not addressed by this
  experiment.

## Artifacts

- `radarsense/cross_sensor_few_shot.py`
- `artifacts/reports/cross_sensor_few_shot_results.json`
