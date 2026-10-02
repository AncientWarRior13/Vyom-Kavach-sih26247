# "Unified Calibration Protocol" — What's Actually Achievable, and What It Found

`CROSS_SENSOR_TRANSFER.md` flagged as a real, questionable weak spot: each real-sensor bridge
does its own "indicative only" SNR calibration, and asked whether transfer failure reflects a
genuine physics finding or an artifact of inconsistent feature engineering. This resolves that
question directly, rather than assuming an answer.

## What's actually true, checked precisely, not assumed

**All four bridges already use the identical `snr_db` formula** — `scale = 5.0 /
(quantile(0.95) − median)`, `snr_db = −15.0 + (raw − median) × scale`, clipped to [−20,−10].
The inconsistency isn't the formula; it's what it's anchored to. Each sensor's own empirical
median self-defines as −15dB — a relative, self-referential calibration, not a shared physical
reference. A genuinely absolute cross-sensor SNR calibration would require each sensor's real
system parameters (transmit power, antenna gain, noise figure) — not available for these
third-party public datasets. Building a fake "unified" absolute calibration without that data
would be unfounded, not a fix, so it wasn't attempted.

**RCS calibration status, checked precisely**: SAAB and TSMS both have *real* radiometric
calibration against a known corner-reflector reference (+20 dBsm) — verified by reading their
bridge code, not assumed. RAD-DAR and DIAT-uSAT use an uncalibrated z-scored power proxy (their
own docstrings already disclose this). So `rcs_estimate` is consistently anchored for half the
sensors and not the other half — a real, different kind of inconsistency than `snr_db`'s.

## The direct, falsifiable test: does dropping the inconsistent features help?

Reused `cross_sensor_transfer.py`'s exact methodology across 3 feature-set variants:

| Variant | Features | In-distribution accuracy | Mean cross-sensor accuracy | Mean cross-sensor PR-AUC |
|---|---|---|---|---|
| All 11 features | 11 | 90.8% | 49.3% | 0.678 |
| Drop `snr_db` | 10 | 89.6% | **53.8%** | 0.694 |
| Drop `snr_db` + `rcs_estimate` | 9 | 84.8% | **60.0%** | 0.687 |

**Confirmed**: dropping the two inconsistently-calibrated features improves mean cross-sensor
accuracy by 10.7 points (49.3%→60.0%), at a real but smaller cost to in-distribution accuracy
(90.8%→84.8%, −6.0pt). This directly answers the question the original report left open — a
meaningful fraction of the cross-sensor transfer failure genuinely was a calibration-consistency
artifact, not purely a physics domain-shift finding.

## Honest interpretation

This is real, useful evidence, not a complete resolution. Cross-sensor accuracy even at 9
features (60.0%) is still far below in-distribution performance (84.8%) — most of the
transfer gap remains even after removing the two features known to be inconsistently
calibrated. So the honest reading is: calibration inconsistency is **a real, measurable, but
partial** contributor. The rest is plausibly genuine physics/hardware domain shift (different
carrier frequencies, different waveform types — FMCW vs. CW — genuinely produce different
feature distributions for the same physical target class), which no feature-dropping can fix.

The trade-off is real too: dropping features to gain cross-sensor robustness costs
in-distribution performance. Whether that trade is worth it depends on whether a deployment
prioritizes single-sensor accuracy or cross-sensor portability — not a question this experiment
answers unilaterally.

## Caveats

- Only `snr_db` and `rcs_estimate` were tested for ablation — the other 9 features weren't
  individually checked for their own calibration consistency (e.g. `range_m`/`altitude_m` are
  literal constants for some sensors per their bridges' own "not measurable" disclosures,
  which is a different, more obvious kind of inconsistency not addressed here).
- Single run, single seed.
- Does not attempt the harder, more complete fix (a shared physical calibration reference) —
  explicitly ruled out here as unfounded given the available data, not attempted and abandoned.

## Artifacts

- `radarsense/cross_sensor_calibration_ablation.py`
- `artifacts/reports/cross_sensor_calibration_ablation_results.json`
