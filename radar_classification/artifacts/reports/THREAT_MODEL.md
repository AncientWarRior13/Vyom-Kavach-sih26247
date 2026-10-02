# Threat Model — Adversarial & Operational Robustness

Radar target classification for a counter-UAS/defense-sensing context has an
adversarial dimension beyond ordinary ML robustness: a real drone operator
can deliberately try to evade detection. This document states plainly what's
been tested (with real measured numbers), and what hasn't (flagged
honestly as future work, not silently omitted).

## Tested (real numbers, `radarsense/robustness.py` / `robustness_report.json`)

| Countermeasure simulated | Perturbation | Drone recall (baseline 91.8%) | Evades detection? |
|---|---|---|---|
| RCS suppression (stealth coating) | −3 dBsm | 93.3% | No — recall unaffected |
| RCS suppression | −6 dBsm | 92.8% | No |
| RCS suppression | −9 dBsm | 93.8% | No |
| Jamming (SNR floor + proportional noise) | floor −16 dB | 91.8% | No |
| Jamming | floor −18 dB | 94.9% | No |
| Jamming | floor −20 dB | 96.4% | No |
| Micro-Doppler masking (shrouded/slowed rotors) | ×0.7 attenuation | 76.4% | Partial degradation |
| Micro-Doppler masking | ×0.5 attenuation | 51.8% | Substantial degradation |
| **Micro-Doppler masking** | **×0.3 attenuation** | **12.3%** | **Yes — known, documented vulnerability** |

**Honest reading**: RCS suppression and jamming do not evade this classifier — the
model relies on kinematic/micro-Doppler structure more than raw signal strength or
RCS magnitude, so those two countermeasures don't work against it. Micro-Doppler
masking (physically shrouding or dramatically slowing rotors to suppress the
propeller signature) is the one countermeasure that works, and it works well —
recall collapses to 12.3% at strong attenuation. This is stated as a real,
unresolved vulnerability, not minimized. It's a direct consequence of
`micro_doppler_bandwidth` being the top SHAP feature for the drone class
(`shap_report.json`) — the model's strongest signal is also its weakest point under
this specific countermeasure.

## Tested since the above: black-box adversarial search (`ADVERSARIAL_ROBUSTNESS.md`)

The two items below — previously listed as future work — have since been tested with real
measured numbers. Full writeup: [`ADVERSARIAL_ROBUSTNESS.md`](ADVERSARIAL_ROBUSTNESS.md).

**Result: 100% evasion success (179/179 real drones) at a small joint perturbation** (median
1.04 std-units across 6 attackable features) — a materially more severe finding than any
single-feature countermeasure above. `rcs_estimate` is the most-exploited feature in
combination, despite RCS suppression *alone* (tested above) not evading detection — a real
interaction effect a one-feature-at-a-time sweep can't see. Read as an upper bound under an
idealized attacker with full sensor-reading control, not a demonstrated real-world attack
(closing that gap needs a per-feature achievability *cost* model, not yet built — see the
report's caveats).

## Tested since the above: model extraction / inversion (`MODEL_EXTRACTION_ATTACK.md`)

**Result: a surrogate trained on 5,000 black-box queries to the public API reaches 99.2%
drone-gate fidelity (93.0% at just 200 queries) — and evasions crafted entirely against that
surrogate, with ZERO further live queries, transferred to fool the real target 100/100
(100%).** This is a more severe threat model than the direct adversarial search above, not
because the underlying vulnerability differs, but because of execution: an adversary needs
only one moderate, one-time extraction run against the live system, then can develop and
refine attacks offline indefinitely with no further signal reaching the target. A defense
watching only for repeated failed evasion attempts against the live API would not catch this.
Full writeup: [`MODEL_EXTRACTION_ATTACK.md`](MODEL_EXTRACTION_ATTACK.md).

## Defended since the above: extraction mitigation deployed and measured (`EXTRACTION_DEFENSE.md`)

Response-hardening (rounding or withholding the probability) was tried first and **measured to
do nothing** — all four variants gave identical extraction fidelity, since the attack only ever
needed the hard label. What worked: a query-pattern anomaly filter (an `IsolationForest` on the
real training distribution) that catches 95.2% of the attack's synthetic queries while wrongly
rejecting only 1.1% of real detections, combined with rate limiting (100 req/min). **Deployed
live in `api/server.py`, verified against a running server** — real detections still classify
correctly, synthetic queries get HTTP 400, excess requests get HTTP 429.

At the same 5,000-query attempt budget, this drops the transfer attack's success rate from
**100% to 84%** — a real, substantial improvement, honestly **not a complete fix**: the queries
that pass the filter are, by construction, the ones that most resemble real detections, which
are also the most informative ones for an attacker. Full writeup:
[`EXTRACTION_DEFENSE.md`](EXTRACTION_DEFENSE.md).

## Defended since the above: adversarial training deployed and measured (`ADVERSARIAL_TRAINING.md`)

Harvested 300 real evasion perturbations from the black-box search (run against TRAINING
drones, test set kept untouched), augmented the training data with them (still correctly
labeled "drone" — the physical target didn't change, only the measured feature vector),
retrained. **Evasion success drops from 100% to 80.1%, and the median perturbation needed by
the attacks that still work rises from 1.04 to 1.55 std-units (+50%)** — a real, measured
improvement, not a complete fix. Normal test performance held up (97.1% accuracy, drone recall
actually improved slightly, 91.8%→92.8%).

Honest limit: this is one round of the standard attack→retrain→attack arms race, not run to
convergence — an attacker re-running the search against this retrained model would very
plausibly find a new (larger, but real) set of successful perturbations. Full writeup:
[`ADVERSARIAL_TRAINING.md`](ADVERSARIAL_TRAINING.md).

## Mitigated since the above: the 42-feature combined model is more robust to masking, measured (`FEAT42_MASKING_ROBUSTNESS.md`)

Tested — not assumed — whether the 42-feature combined model's second, independently-noised
feature set actually helps against micro-Doppler masking, given both feature sets trace back to
the same `micro_doppler_bandwidth`/`micro_doppler_jitter` root cause. Regenerated masked drone
instances with the *same* attenuation fed consistently into both feature derivations (a fair
test — masking only the physics side while leaving the correlated waveform-stat side untouched
would have been misleading).

**Result: the 42-feature model wins at every attenuation level** — 98.5% vs. 91.8% unperturbed,
93.1% vs. 76.4% at ×0.7, 63.1% vs. 51.8% at ×0.5, and even at the severe ×0.3 case, 16.8% vs.
12.3%. Not a fix (16.8% is still a real degradation), but a genuine, already-available
mitigation: deploying the 42-feature model instead of the 11-feature model measurably reduces
this vulnerability's impact, without needing to build the kinematic-fallback system described
below. Full writeup: [`FEAT42_MASKING_ROBUSTNESS.md`](FEAT42_MASKING_ROBUSTNESS.md).

## Mitigated further: feature-dropout training, a direct fix (`FEATURE_DROPOUT_TRAINING.md`)

The strongest mitigation tested against micro-Doppler masking so far. Directly targets the
diagnosed mechanism (over-reliance on `micro_doppler_bandwidth` as a near-sufficient statistic)
rather than working around it: 50% of drone training rows get a masked duplicate (same
attenuation range the attack uses, 0.3–0.9×), still correctly labeled "drone," mixed into
training.

**Result: at the severe ×0.3 masking level, drone recall goes from 12.3% to 64.1% — more than
5× better** — at a real, disclosed cost of 3.6 points of unperturbed-case recall (91.8%→88.2%).
Bigger effect than the 42-feature model's fusion-based robustness (16.8% at ×0.3) — worth
combining the two in a future pass, not yet done. Full writeup:
[`FEATURE_DROPOUT_TRAINING.md`](FEATURE_DROPOUT_TRAINING.md).

## The second mitigation direction, also tested: kinematic fallback (`KINEMATIC_FALLBACK.md`)

A structurally different mechanism from feature-dropout training — detection + routing rather
than retraining: a second classifier trained on the 9 non-micro-Doppler features (cannot
structurally be fooled by masking), blended in when the primary model's micro-Doppler reading
looks anomalously low for a drone.

**Result: real but modest** — ×0.3 masking recall improves from 12.3% to 29.2% (more than 2×,
but well below feature-dropout training's 64.1%), at a broader accuracy cost (−1.1pt overall,
vs. feature-dropout's −3.6pt drone-specific cost). One good sign: it also helps real,
naturally-quiet-rotor drones (76.9%→82.1%), not just the attacked case. First reasonable
configuration tested, not tuned — a hyperparameter sweep would likely improve the trade-off.
Full writeup: [`KINEMATIC_FALLBACK.md`](KINEMATIC_FALLBACK.md).

## COMPLETED: every vulnerability above pushed from "partial fix" to its honest ceiling

A dedicated cycle took the four mitigations above from single measured points to their full,
honest limits. Each has its own report; summarized here so the threat model stays the index.

**Micro-Doppler masking — now a tunable frontier, not a single point of failure**
(`MASKING_DEFENSE_COMPLETE.md`). The plan was "stack feature-dropout + fallback + fusion"; built
honestly, stacking *lost* (probability-blending averages the strong model down; `md<cutoff`
replace-routing wrecks precision because aircraft/birds naturally have low micro-Doppler). What
works: a measured augmentation-intensity frontier (severe-masking ×0.3 recall recoverable from
12.3% up to 91.8% at a precision cost) plus **escalation routing** — the deployable default —
which recovers ×0.3 recall 12.3%→48.7% and ×0.1 from 0%→48.2% at **near-zero clean cost**
(accuracy 97.59%→97.44%). Both failed combination strategies are kept visible.

**Adversarial evasion — empirical training plateaus, but a certificate closes it**
(`ADVERSARIAL_DEFENSE_COMPLETE.md`). Iterative adversarial training drops evasion 100%→~71% and
then oscillates in a 71–82% band — the honest ceiling of the technique — while raising required
perturbation ~40%. The real win is **certified robustness via randomized smoothing**: the
hardened model's mean certified radius rises 0.53→1.94 std-units, and **83.3% of drones are
provably un-evadable within the attack's own median perturbation** (up from 19.2%). A physics
plausibility gate independently catches 41% of the evasions that get through.

**Cross-sensor transfer — a menu of measured recoveries** (`CROSS_SENSOR_DOMAIN_ADAPT.md`).
CORAL domain adaptation recovers +13 drone-recall points with **zero target labels** (unlabeled
target features only); leave-one-sensor-out pooling reaches 64% mean accuracy (+15pt over
single-source zero-shot); few-shot climbs to 82% at 500 labels. Two honest negatives kept:
diagonal moment-matching underperforms full CORAL, and CORAL+few-shot underperforms plain
few-shot (labels dominate).

**Model extraction — prevention *and* attribution both fail; only operational cost survives**
(`EXTRACTION_DEFENSE_COMPLETE.md`). Fingerprint-based theft detection was tested three ways and
does not separate a clone from an honest competitor — the model is too generic to carry a
copyable signature. Boundary label-noise joins probability-rounding as a useless response-side
defense. The anomaly filter is quantified as **defeated by a realistic query profile** (1.1%
caught vs 95.4% for naive uniform). The one lever with no demonstrated bypass is per-key query
budgeting; the honest posture is layered/operational, not a technical guarantee.

## Explicitly NOT tested — future work, not silently assumed safe

- **Training-data poisoning.** No evaluation of what happens if an adversary could
  influence the training data (not applicable to the simulated dataset, but
  relevant if this pipeline were retrained on operationally-collected data in
  deployment).
- **Multi-target / clutter confusion attacks.** Deliberately operating a drone near
  a bird flock or in radar clutter to induce misclassification via legitimate
  target confusion rather than signal countermeasures — related to but distinct
  from the `anomaly` class already in the schema; not stress-tested as a deliberate
  evasion tactic.

## Mitigation direction for the known vulnerabilities

The fusion/fallback direction named here was implemented and measured in full — see the
COMPLETED section above and `MASKING_DEFENSE_COMPLETE.md`. The honest outcome differed from the
prediction: naive fusion (probability-blending the kinematic fallback into the primary)
*underperforms*, and the deployable win turned out to be escalation routing plus a measured
augmentation-intensity frontier. The direction was right; the specific mechanism that works was
found empirically, not assumed.

The adversarial "joint, correlated shift" monitor anticipated here is now realized two ways in
`ADVERSARIAL_DEFENSE_COMPLETE.md`: a physics plausibility gate (an IsolationForest on the joint
real-drone feature distribution, catching 41% of evasions) and, more fundamentally, randomized-
smoothing certification that provably bounds *any* small joint perturbation rather than watching
for specific ones.

For model extraction: query-pattern anomaly detection and rate limiting were implemented and
measured (`EXTRACTION_DEFENSE.md`), and the follow-up (`EXTRACTION_DEFENSE_COMPLETE.md`)
established the honest ceiling — fingerprint attribution and every response-side defense fail,
the anomaly filter is defeated by realistic queries, and per-key budgeting is the surviving
operational lever. Extraction is treated as a cost-raising and monitoring problem, not a solved
one — stated plainly rather than papered over.
