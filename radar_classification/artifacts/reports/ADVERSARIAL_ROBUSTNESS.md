# Black-Box Adversarial Spoofing Search

Fills the two gaps `THREAT_MODEL.md` explicitly named as "not attempted here": (1) an
*optimized* adversarial search, rather than the physically-motivated single-countermeasure
tests already in `robustness.py`, and (2) evasion via a crafted feature vector, rather than
pure signature suppression. Full methodology in
[`radarsense/adversarial_robustness.py`](../../radarsense/adversarial_robustness.py)'s
docstring.

## Setup

- **Target**: the deployed 11-feature frozen model, at its real deployed threshold (0.54).
- **Attack**: black-box random search / basin-hopping (CatBoost is a tree ensemble, not
  differentiable the way a neural net is — gradient-free search is the appropriate method,
  as `THREAT_MODEL.md` itself names). For each of 179 real drones the model already correctly
  classifies, 8 restarts × 150 iterations searching for the smallest perturbation (measured in
  per-feature standard-deviation units, so features on very different scales — m/s vs. a
  [0,1] jitter ratio — are compared fairly) that drops p(drone) below the deployed threshold.
- **Attack surface**: 6 of 11 features an adversary could plausibly influence via flight
  behavior or physical countermeasures — `doppler_velocity`, `rcs_estimate`,
  `micro_doppler_bandwidth`, `micro_doppler_jitter`, `velocity_variance`, `acceleration`.
  Excludes `range_m`/`azimuth_deg`/`altitude_m` (changes mission profile, not signature) and
  `snr_db` (sensor/environment-determined, not attacker-controlled). `doppler_freq_shift` is
  re-derived from the perturbed `doppler_velocity`, never perturbed independently.
- **Bounds**: every perturbation is clipped to the same physical sensor-reading ranges
  `simulator.py` itself enforces — this never searches outside physically valid readings.

## Result: 100% evasion, at a small perturbation

All 179 attacked drones were successfully pushed below the drone threshold:

| | Value |
|---|---|
| Attack success rate | **179/179 (100%)** |
| Median minimal perturbation norm | 1.04 std-units |
| 25th/75th percentile | 0.67 / 1.45 std-units |

For scale: 1 std-unit total (summed in quadrature across 6 features) is a *small* joint
change — far smaller than what any single physically-motivated countermeasure in
`robustness.py` needed to move the needle. This is a materially more severe finding than
anything reported there.

Features most exploited by the successful attacks (mean |perturbation| / feature std):

| Feature | Mean exploitation |
|---|---|
| `rcs_estimate` | **0.656** |
| `doppler_velocity` | 0.338 |
| `micro_doppler_jitter` | 0.289 |
| `micro_doppler_bandwidth` | 0.263 |
| `acceleration` | 0.179 |
| `velocity_variance` | 0.121 |

## The genuinely new finding: an interaction effect single-feature testing couldn't see

`robustness.py`'s existing single-feature RCS suppression test (−3 to −9 dBsm) found RCS
suppression **alone does not evade detection** (drone recall stayed at 91–94%). This new
result finds `rcs_estimate` is nonetheless the *single most exploited feature* once combined
with small nudges to velocity and micro-Doppler jitter. Neither test contradicts the other —
they answer different questions. RCS reduction alone isn't enough because the model still has
kinematic and micro-Doppler signal to fall back on; but RCS reduction *combined with* small,
individually-unremarkable changes to a few other features is enough, because it erodes
several weak signals at once rather than one strong one. This is exactly the kind of
interaction effect a joint optimizer finds and a one-feature-at-a-time sweep structurally
cannot.

## Honest interpretation

- This is a real, demonstrated vulnerability under the black-box search's specific threat
  model: full read/write control over sensor-reading feature values, within physically valid
  ranges. It should be read as an **upper bound on vulnerability under an idealized
  attacker**, not a demonstrated real-world attack.
- **What this does NOT show**: that a real drone operator can *achieve* this exact combined
  perturbation. Simultaneously and precisely modulating RCS, velocity, and micro-Doppler
  jitter by the specific found amounts would require a real physical mechanism (materials +
  flight control + rotor modulation, jointly tuned) that isn't modeled or verified here — the
  same limitation `robustness.py`'s own physically-motivated tests avoid by testing named,
  achievable countermeasures instead of a free-form search. Closing this gap (a per-feature
  *cost* model — e.g. how much RCS reduction a given RAM coating thickness actually buys —
  and re-running the search under that cost constraint) is the natural next step.
- Combined with `THREAT_MODEL.md`'s existing micro-Doppler-masking finding, the honest
  picture is: this classifier's decision boundary has real, exploitable slack across several
  features jointly, not just its single most-important one. A defense that hardens against
  micro-Doppler masking alone (the mitigation direction already proposed in
  `THREAT_MODEL.md`) would not close this broader vulnerability.

## Caveats

- Single search configuration (8×150 random search) — a more exhaustive black-box method
  (CMA-ES, genetic algorithm with larger population) might find even smaller successful
  perturbations; this result is an upper bound on required perturbation, not a proof of the
  true minimum.
- Attacked only the 11-feature frozen model — not yet repeated against the 31-feature,
  42-feature, or quantum-kernel backends, which may have different (larger or smaller)
  attack surfaces.
- Physical-bound clipping is necessary but not sufficient for real-world achievability, as
  discussed above.

## Artifacts

- `radarsense/adversarial_robustness.py`
- `artifacts/reports/adversarial_robustness_results.json` — per-sample attack results
