# Defending Against Model Extraction: What Actually Works

Follow-up to `MODEL_EXTRACTION_ATTACK.md`, which found a surrogate trained on 5,000 black-box
queries reaches 99.2% drone-gate fidelity, and evasions crafted entirely offline against it
transferred to fool the real model 100/100 times. This tests candidate defenses — measured
before being applied, not assumed to work.

## What was tried and DIDN'T work: hardening the response content

Tested four response modes (`extraction_defense_test.py`): full probabilities (baseline),
rounded to 2 decimals, rounded to 1 decimal, and label-only (no probability at all). **All four
gave identical extraction fidelity (95.4% overall, 99.15% drone-gate) and identical transfer
rate (99/100).**

Why: the surrogate was trained on the returned *hard label*, not the probability value — which
is what any real classifier API has to return to be useful at all. Coarsening or hiding the
probability doesn't touch what the attacker actually needed. This is a real negative result,
not a shortcut skipped — it directly rules out a whole category of "just hide the confidence
score" defenses that might otherwise have seemed like the obvious fix.

**Also checked**: how few queries does an attacker actually need? Drone-gate fidelity was
already **97.0% at just 10 queries** (20: 97.0%, 50: 97.1%, 100: 98.2%). The decision boundary
is cheap enough to learn that rate-limiting alone would need to be so aggressive (single-digit
requests) it would break legitimate use of the API too.

## What DID work: filtering queries that don't look like real detections

An `IsolationForest` fit on the real training distribution (`extraction_defense.py`) — real
detections follow the simulator's per-class physical correlations (RCS, range, velocity,
micro-Doppler co-varying the way real targets' do); an attacker's i.i.d.-uniform sampling over
the documented physically-plausible bounds does not.

Calibrated to reject only the bottom 1st percentile of real training scores, verified against
held-out real test data (never used to calibrate the threshold): **1.10% false-reject rate on
genuine detections.** Against the extraction attack's actual query distribution: **95.2%
caught and rejected.**

**End-to-end result**: an attacker sending the same 5,000-query budget as the original attack
gets only 199 queries actually answered. Rebuilding the surrogate from just those 199:

| | Before (no defense) | After (anomaly filter) |
|---|---|---|
| Queries answered (of 5,000 sent) | 5,000 | 199 |
| Surrogate drone-gate fidelity | 99.2% | 97.3% |
| Transfer attack success rate | **100/100 (100%)** | **84/100 (84%)** |

## Deployed

Both measured defenses are now live in `api/server.py`:

1. **Query-pattern anomaly filter** (`extraction_anomaly_filter.joblib`) — `/predict` and
   `/batch` reject samples that don't resemble real detections (HTTP 400) before they ever
   reach the model.
2. **Rate limiting** (`slowapi`, 100 requests/minute per client) — caps total attempts,
   answered or not.

Verified live, not just in a script: started the real server, confirmed (a) a genuine test-set
detection still gets classified correctly, (b) a synthetic extraction-style query gets
rejected with a 400, (c) the 101st request within a minute gets a 429.

## Honest interpretation

This is a real, substantial improvement — roughly 25x fewer usable queries per attempt, and
transfer success drops 16 points — **not a complete fix.** The 199 queries that do pass the
filter are, by construction, the ones that look most like real detections, which also makes
them informative about the actual decision boundary near real data — exactly where an attacker
most wants information. An attacker willing to spend more attempts (the filter doesn't block
retrying) or willing to more carefully hand-craft physically-plausible-looking queries instead
of uniform sampling could likely do better than this measurement shows. This is disclosed as
a meaningful deterrent that raises the cost of the attack, not a guarantee that closes it.

## Caveats

- Tested against the SAME response-hardening/anomaly-filter combination on the 11-feature
  model's API only — not yet repeated on the 31-feature, 42-feature, or quantum backends.
- The IsolationForest is fit on the same simulator distribution the model itself was trained
  on — an attacker with access to real deployment data (not just the public schema
  documentation) could plausibly craft queries that pass the filter more often.
- Single run, single seed for the query sampling.

## Artifacts

- `radarsense/extraction_defense_test.py` — response-hardening negative result
- `radarsense/extraction_defense.py` — anomaly filter training
- `artifacts/models/extraction_anomaly_filter.joblib`, `extraction_anomaly_filter_meta.json`
- `artifacts/reports/extraction_defense_test_results.json`
- `api/server.py` — the deployed defenses
