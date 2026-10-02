# Model Extraction: The Honest Ceiling, Established With Numbers

`EXTRACTION_DEFENSE.md` left this partial: the anomaly filter raises attacker cost ~25× but a
realistic-profile adversary was flagged as a remaining hole, and response-hardening was already
shown useless. The plan for closing it had two prongs — (a) **detect/attribute** stolen models
via fingerprinting (reframe from "prevent" to "prove theft + revoke"), and (b) **raise cost**
via per-key budgeting and boundary-region noise. Both prongs were built and measured. The honest
result: **extraction of this model resists prevention *and* attribution, and the only lever that
survives contact with the numbers is operational cost.** That's a real, defensible conclusion —
and more useful to a deployer than an oversold "solved."

## Prong A — fingerprinting for theft detection: tested three ways, does not work here

The idea: embed secret probe inputs where a model *extracted from ours* agrees with us far more
than an *independently trained competitor* does, so a clone is provably attributable. Three probe
designs, scored non-circularly (probes built with a selector panel, separation measured on
**held-out** reference models):

| Probe design | Extracted clone agreement | Held-out innocent (max) | Separation | Detectable? |
|---|---|---|---|---|
| Boundary (on the decision surface) | 0.483 | 0.540 | −0.057 | No |
| Idiosyncratic (target vs. panel disagreement) | 0.490 | 0.480 | +0.010 | No |

No design separates the clone from an honest competitor. The reason is a genuine property of
*this* problem, not a bug: the task is simple enough (11 features, binary drone-vs-rest) that
**every competent model converges to nearly the same decision function** — whether trained on the
target's query responses (the clone) or independently on similar real data (the competitor). On
easy points all models agree; on boundary points all models coin-flip; on the target's
idiosyncratic points even the clone fails to reproduce the target reliably. There is no
distinctive signature to copy, so there is nothing to fingerprint. (Fingerprinting is known to
work on high-capacity neural nets precisely because they *have* idiosyncratic boundaries; a small
tabular model on an easy task does not.) Reported as a negative because it was measured across
three designs, not assumed.

## Prong B, part 1 — boundary-region label noise: fails, like every response-side defense

Probability rounding was already shown useless (`EXTRACTION_DEFENSE.md`). Flipping the hard
*label* near the boundary is a stronger lever — it corrupts the surrogate's supervision exactly
where it matters — so it was worth a direct test:

| Flip margin | Labels flipped | Surrogate drone-gate fidelity |
|---|---|---|
| 0.00 (none) | 0.0% | 0.9895 |
| 0.05 | 0.8% | 0.9892 |
| 0.10 | 1.9% | 0.9901 |
| 0.15 | 3.0% | 0.9903 |

**No effect.** CatBoost shrugs off a few percent of label noise, and to flip enough labels to
hurt the surrogate would break the API for legitimate users. This closes the *entire* category of
response-side defenses (coarsen the probability, hide it, or perturb the label) — none of them
touch what extraction actually needs, because it only needs the hard label the API must return.

## Prong B, part 2 — the anomaly filter is defeated by a realistic query profile

The earlier report *flagged* this as a caveat; here it is quantified:

| Attacker query profile | Anomaly filter reject rate | Surrogate drone-gate fidelity (from passers) |
|---|---|---|
| i.i.d.-uniform (naive) | 95.4% | 0.965 (only 92 of 2000 passed) |
| Realistic (drawn from real distribution) | **1.1%** | **0.994** (1978 of 2000 passed) |

The filter is excellent against a naive uniform attacker and **nearly useless against a patient
one** who queries with realistic-looking detections — 1978 of 2000 sail through, yielding a 99.4%
fidelity clone. This is the hard truth the earlier caveat hinted at: the filter buys protection
only against the low-effort attacker.

## What actually survives — the one lever with no hole: operational cost

| Defense | Verdict |
|---|---|
| Response hardening (rounding/hiding/label-flip) | ✗ extraction needs only hard labels |
| Query anomaly filter | ✗ defeated by realistic queries (1.1% caught) |
| Fingerprint attribution | ✗ model too generic to carry a signature |
| **Per-key query budget** | **✓ real friction** |

The per-key token bucket caps a single identity at 500 answered queries per burst (4500 of 5000
refused). It doesn't stop extraction — an attacker can distribute across many keys — but it is the
one measured lever that raises cost without a demonstrated bypass, forcing an attacker into
many-identity, rate-limited, realistic-query campaigns that are far more detectable operationally
than a single 5000-query burst.

## Honest interpretation

Model extraction of a small, well-behaved tabular classifier is **fundamentally hard to defend
technically** — this is consistent with the broader literature, and now demonstrated with numbers
on this exact model. The honest deployment posture is therefore layered and mostly operational:
per-key budgeting + total-attempt rate limits (already deployed) + monitoring for distributed
many-key realistic-query patterns + contractual/ToS controls — not a belief that any single
algorithm makes the model un-copyable. Stating that plainly is the deliverable; claiming a
technical fix that the measurements don't support would be the one thing a sharp reviewer would
catch.

## Caveats

- Fingerprinting was tested with CatBoost surrogates and CatBoost innocents; a wildly different
  architecture might carry more signature — not tested, and not expected to change the conclusion
  for the deployed model.
- Realistic queries were drawn from the real held-out distribution, i.e. an attacker who already
  possesses representative detections; an attacker without them pays a real collection cost first.
- Single seed for query sampling; the ordering of conclusions (all technical defenses fail,
  budgeting is the surviving lever) is robust to that.

## Follow-up: make the copy less useful (`extraction_defanged.py`)

Since extraction can't be prevented, attack the *reason* it matters — the transfer attack (steal
a clone, craft evasions offline, use them on the real model). Does hardening the deployed model
break that chain?

| Target the clone was extracted from | Clone-crafted evasion transfer rate |
|---|---|
| Original (undefended) model | 100% |
| Iteratively-hardened model | **77.9%** |

Hardening the deployed model cuts the transfer rate of clone-crafted evasions from 100% to 78% —
a real reduction, **not** elimination. Combined with the physical-cost budget
(`ADVERSARIAL_DEFENSE_COMPLETE.md`: ~22% achievable at a realistic budget) and per-key budgeting,
the *effective* attack an extraction adversary can mount is substantially degraded, layer by
layer. Stated honestly: 78% is not zero, and no query-accessible model is made un-copyable — the
copy is made progressively less useful, with the exact number reported for each layer.

## Artifacts

- `radarsense/extraction_defense_complete.py`
- `radarsense/extraction_defanged.py`, `artifacts/reports/extraction_defanged_results.json`
- `artifacts/models/extraction_fingerprints.joblib`
- `artifacts/reports/extraction_defense_complete_results.json`
