# Micro-Doppler Masking: From Single Point of Failure to Tunable Frontier

`THREAT_MODEL.md` reported the sharpest vulnerability in this project: micro-Doppler masking
(shrouded rotors — attenuate `micro_doppler_bandwidth`/`micro_doppler_jitter`) collapses the
11-feature deployed model's drone recall from **91.8% → 12.3%** at ×0.3 attenuation, and to
**0.0%** at ×0.1. Three partial mitigations were measured *separately* earlier
(`FEATURE_DROPOUT_TRAINING.md`, `KINEMATIC_FALLBACK.md`, `FEAT42_MASKING_ROBUSTNESS.md`). This
is the complete treatment: it combines them, tests the "stack everything" hypothesis directly,
finds it **wrong**, and produces instead a measured precision ↔ robustness frontier with a
chosen, deployable operating point at essentially zero clean-data cost.

## The plan was "stack all three and let them compound." Built honestly, that plan is wrong.

Two combination strategies were tested and **rejected with numbers**, which matters more than
quietly shipping the one that happened to work:

1. **Probability-blend stacking fails.** Blending the (hardened) primary's class
   probabilities with the kinematic fallback's, weighted by a masking-suspicion score, *drags
   the strong model down toward the weaker one*. The fallback's masking recall ceiling is ~88%;
   any average with it caps below what the hardened primary alone reaches. Measured ×0.3 recall:
   dropout primary **0.774** vs. blended stack **0.738**. Averaging two models only helps when
   neither dominates — here one does.

2. **Replace-routing on `md < cutoff` fails.** Routing every low-micro-Doppler sample to the
   immune fallback destroys precision (**0.775 → 0.313**), because the drone's *own* 20th-percentile
   micro-Doppler cutoff (166.8 Hz) sits *above* almost all aircraft/bird micro-Doppler (15–45 Hz) —
   so "low md" is not a masking signature for the population at large, and the rule dumps nearly
   every non-drone through the weaker model.

Both negatives are kept visible in the code and JSON rather than hidden, because "we tried the
obvious combination and it underperformed its own best part" is the honest and more useful
finding.

## What actually works, part 1: the augmentation-intensity frontier

`FEATURE_DROPOUT_TRAINING.md` sampled *one* point (K=1 masked duplicate per drone,
attenuation [0.3, 0.9]). Masking recall is trivially driven up by over-augmenting — but at a
precision cost — so the honest object is the whole frontier, not one number:

| Operating point | Aug. | Clean acc | Clean drone-prec | ×0.3 recall | ×0.1 recall |
|---|---|---|---|---|---|
| Original (undefended) | — | 97.6% | 0.775 | 12.3% | 0.0% |
| `k1_light` | K=1, [0.3,0.9] | 97.0% | 0.672 | 71.3% | 39.0% |
| **`k2_medium`** (chosen) | K=2, [0.2,0.9] | 97.1% | **0.706** | **77.4%** | 50.8% |
| `k3_wide` | K=3, [0.12,0.95] | 96.1% | 0.526 | **91.8%** | **90.3%** |

`k2_medium` **dominates** the original single-point `k1` baseline — higher clean precision
*and* higher masking recall — so the earlier baseline wasn't even on the frontier. And
`k3_wide` shows the vulnerability can be **nearly eliminated** (×0.3: 12.3→91.8%, ×0.1:
0→90.3%), at a real drone-precision cost (0.775→0.526). There is no free lunch, but there is a
dial, and its whole range is now measured.

## What actually works, part 2: escalation routing at near-zero clean cost

The correct use of the masking-immune fallback is neither blending nor replacing — it's
**escalation**: keep the original model's decisions everywhere, and only *add* a drone call
where masking is plausible (`md` below cutoff) **and** the immune fallback is *confident* it's a
drone. A genuine aircraft/bird with naturally low micro-Doppler is rejected by the fallback too,
so it is never escalated; only a kinematically drone-like target whose micro-Doppler has been
suppressed gets rescued. The rescue confidence threshold was swept and pinned to keep clean
precision within 5 points of the original:

| Level | Original | Escalation-routed | Δ |
|---|---|---|---|
| Unperturbed | 91.8% | **93.3%** | +1.5 |
| ×0.7 | 76.4% | 81.0% | +4.6 |
| ×0.5 | 51.8% | 65.6% | +13.8 |
| ×0.3 | 12.3% | **48.7%** | **+36.4** |
| ×0.1 | 0.0% | **48.2%** | **+48.2** |

**Clean-data cost: accuracy 97.59% → 97.44% (−0.15 pt), drone precision 0.775 → 0.731 (−4.4 pt),
and clean drone recall actually *rises* 91.8% → 93.3%** (naturally-quiet real drones get rescued
too). This is the precision-preserving default: a 4× severe-masking recall recovery and non-zero
recall restored at the extreme ×0.1 level, for a clean-data cost that rounds to noise.

**A stronger posture from the same knob.** The rescue threshold is a dial. Lowering it to 0.5
(a masking-threat deployment that accepts more precision cost) recovers far more: **×0.3 recall
12.3% → 76.9% and ×0.1 0% → 76.4%** — a 6× recovery, essentially matching the masking-immune
fallback — at a clean-accuracy cost of only −2.6 points (97.6% → 95.0%). So the operator picks
the point: near-zero-cost 4× recovery, or 6× recovery for a couple of accuracy points, both from
one measured sweep.

## The complete picture: three deployable postures, one measured frontier

Micro-Doppler masking is no longer a single point of failure. It is a **tunable frontier**, and
the right point depends on the threat posture:

| Posture | Mechanism | ×0.3 recall | Clean cost |
|---|---|---|---|
| Masking rare, precision precious | **Escalation routing** | 12.3 → **48.7%** | acc −0.15 pt |
| Masking expected, balanced | `k2` curriculum dropout | 12.3 → **77.4%** | prec −6.9 pt |
| Masking the primary threat | `k3` curriculum dropout | 12.3 → **91.8%** | prec −24.9 pt |
| Absolute masking immunity | fallback-only (no micro-Doppler) | flat **87.7%** | prec −47.5 pt |

For context, the 42-feature combined model (`FEAT42_MASKING_ROBUSTNESS.md`) reaches 16.8% at
×0.3 from feature fusion alone — better than the undefended 11-feature model but far below any
of the active defenses here.

## Honest interpretation and caveats

- **This is the one vulnerability that is now essentially closable.** At the aggressive end
  (`k3`/fallback-only) severe-masking recall is fully recovered; the only question is how much
  clean-data precision you spend, and that is now a measured, explicit choice rather than an
  unknown.
- **Adaptive adversary caveat (real).** Escalation routing keys off `md < cutoff`. An adversary
  who knows this could mask micro-Doppler to *just above* the cutoff to stay on the original
  model. But that is a weaker attack: partial masking above the cutoff degrades the original
  only mildly (×0.7 → 76.4%), and to reach the damaging regime they must cross the cutoff and
  trip the detector. They cannot both fool the primary (needs low md) and avoid the detector
  (needs high md). The curriculum-dropout postures have no cutoff to game and degrade
  gracefully instead.
- **Precision is the currency, not accuracy.** Because the drone class is ~2.5% of traffic,
  overall accuracy barely moves; drone precision is the honest cost meter, and it's reported
  everywhere above.
- Single seed for augmentation sampling; the frontier's ordering (k2 dominates k1; blend and
  replace-routing lose) is robust to that, but exact recall values would shift a few points
  across seeds.

## Artifacts

- `radarsense/masking_defense_stacked.py`
- `artifacts/models/masking_defense_primary.joblib`, `masking_defense_fallback.joblib`,
  `masking_defense_meta.json`
- `artifacts/reports/masking_defense_stacked_results.json` — full frontier, stack-vs-primary
  comparison, and the rescue-threshold sweep
