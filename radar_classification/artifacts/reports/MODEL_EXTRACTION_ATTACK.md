# Model Extraction / Inversion Attack

Fills the last well-specified, testable gap `THREAT_MODEL.md` names: "No evaluation of
whether repeated queries to the deployed ONNX model (via the `/predict` or `/batch` API)
could allow an adversary to reconstruct decision-boundary information useful for evasion."
Full methodology in
[`radarsense/model_extraction_attack.py`](../../radarsense/model_extraction_attack.py)'s
docstring.

## Threat model

An adversary with **query access only** — no training data, no model internals, no
architecture knowledge. Confirmed the live `/predict` endpoint (`api/server.py`) is a thin
wrapper around `api.predict.predict_one` with no rate-limiting or response filtering, via one
actual live HTTP request matched bit-for-bit against a direct function call — so the bulk of
the attack runs via direct calls for speed, not HTTP overhead, without changing what's being
measured. Query inputs are sampled uniformly within the schema's *publicly documented*
physical bounds, not the real training distribution a genuine black-box attacker wouldn't
have.

**Disclosed leak that makes this easier than it could be**: `/predict` returns full class
probabilities, not just the hard label. A label-only API would make extraction meaningfully
harder — this measures risk under the API's actual current behavior, not a hardened
hypothetical.

## Stage 1: extraction — surrogate fidelity vs. query budget

A surrogate CatBoost model trained purely on (random query, API response) pairs, evaluated
against the real target on a held-out set (unavailable to a real attacker, used here only to
measure ground-truth extraction quality):

| Query budget | Overall label agreement | Drone-gate agreement |
|---|---|---|
| 200 | 54.3% | 93.0% |
| 500 | 80.8% | 98.7% |
| 1,000 | 82.9% | 99.0% |
| 2,000 | 85.3% | 98.9% |
| 5,000 | 93.4% | **99.2%** |

The full 4-class boundary is harder to extract than the one decision that actually matters
for evasion — **drone-gate agreement is already 93% at just 200 queries**, an extremely low
bar, and above 99% by 1,000 queries. Full 4-class fidelity climbs more slowly but still
reaches 93.4% by 5,000 queries — a modest budget against a public, unthrottled API.

## Stage 2: transfer attack — the actual security question

Extraction alone isn't dangerous; extraction *enabling evasion crafted entirely offline* is.
Using the 5,000-query surrogate, ran the same black-box search as
[`ADVERSARIAL_ROBUSTNESS.md`](ADVERSARIAL_ROBUSTNESS.md) — but every query during the search
hit the **surrogate only**. Zero further live queries. Then checked, once, whether each
crafted evasion also fooled the **real** target:

| | Result |
|---|---|
| Evasions crafted (surrogate-vs-surrogate) | 100/100 (100%) |
| **Transferred to fool the real target** | **100/100 (100%)** |

Every single surrogate-crafted evasion also fooled the real model. Zero live queries were
spent during attack development — only the one-time extraction budget (5,000 queries) and
then unlimited offline iteration against a private copy.

## Honest interpretation

This is a **materially more severe threat model than the direct adversarial search** in
`ADVERSARIAL_ROBUSTNESS.md`, not because the underlying vulnerability is different (both
exploit the same joint-feature decision-boundary slack), but because of *how* the attack
would actually be executed: an adversary doesn't need sustained, detectable, high-volume
query access to the live system while iterating on an attack — a single moderate extraction
run, done once, produces a private offline copy good enough to develop and refine evasions
indefinitely with no further signal reaching the target system. Any defense that only
monitors for "many failed evasion attempts against the live API" would not see this coming —
the live API only ever sees one clean-looking batch of feature-space queries, then the
attack itself is executed with zero live footprint.

Combined with `ADVERSARIAL_ROBUSTNESS.md`'s finding, the full picture now spans three of the
five items `THREAT_MODEL.md` originally listed as untested: spoofing (evasion via crafted
feature vectors), optimized adversarial search, and model extraction. Two of the remaining
two (training-data poisoning, multi-target/clutter confusion) are noted there as not
meaningfully testable on synthetic data or already partially addressed by the schema's
`anomaly` class respectively.

## Mitigation directions (not implemented, noted as concrete next steps)

- **Rate-limiting / query budgeting** on the public API — extraction needed only ~500-5,000
  queries here; throttling to a much smaller daily budget per client would raise the cost.
- **Return the hard label only, not full probabilities** — the disclosed leak above; would
  slow (not necessarily stop) extraction, since label-only extraction is a well-studied,
  still-viable attack in the literature, just less sample-efficient.
- **Query-pattern anomaly detection** — a burst of thousands of near-uniform-random,
  physically-implausible-in-combination queries (e.g. rapid range/RCS/velocity sweeps with no
  temporal/kinematic coherence) is a detectable signature distinct from real target
  detections, which arrive from genuine physical trajectories.

## Caveats

- Surrogate uses the same model family (CatBoost) as the real target — a genuine black-box
  attacker might not know this; a mismatched surrogate family could extract less efficiently.
  Not tested here.
- Single run, single random seed for query sampling and attack crafting.
- Tested only against the 11-feature model's public API — not yet repeated against the
  31-feature, 42-feature, or quantum-kernel backends.

## Artifacts

- `radarsense/model_extraction_attack.py`
- `artifacts/reports/model_extraction_attack_results.json`
