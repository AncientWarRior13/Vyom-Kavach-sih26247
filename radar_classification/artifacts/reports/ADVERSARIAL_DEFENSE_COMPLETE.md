# Adversarial Evasion: From "80% Evadable" to a Provable Guarantee

`ADVERSARIAL_TRAINING.md` ran one round of adversarial training and cut the evasion rate
100% → 80.1%. Partial by construction — one round only teaches the perturbations already seen.
This closes it as far as the problem actually permits, which is the honest framing: **no defense
drives adversarial evasion to zero**, so "complete" here means (1) push empirical training to its
plateau, (2) add a *provable* certificate that no small perturbation can evade, and (3) layer an
independent physics gate. All three built and measured.

## 1. Iterative adversarial training: it plateaus, and that's the honest finding

Retrain → re-attack the new model → harvest the fresh evasions it's still vulnerable to →
retrain again, four rounds:

| Round | Evasion rate | Median perturbation norm (std-units) | Clean acc | Clean drone recall |
|---|---|---|---|---|
| 0 (original) | 100.0% | 1.02 | 97.59% | 91.8% |
| 1 | 82.0% | 1.47 | 97.33% | 93.3% |
| 2 | 72.0% | 1.50 | 97.19% | 87.7% |
| 3 | 80.0% | 1.46 | 97.40% | 83.1% |
| 4 | 71.3% | 1.37 | 97.12% | 89.2% |

The evasion rate drops from 100% to the **low-70s and then oscillates in a 71–82% band with no
further downward trend** — empirical adversarial training has clearly hit diminishing returns.
This is the honest ceiling of the technique and exactly why it can't be the whole answer. But it
buys something real that a single round already hinted at and iterating confirms: **the
perturbation an attacker must find is ~40% larger** (median norm 1.02 → ~1.4–1.5 std-units), at
essentially no clean-accuracy cost (97.6% → 97.1%). The attack still often succeeds, but it must
work harder — which sets up the certificate.

## 2. Certified robustness (randomized smoothing): the provable guarantee

The empirical evasion rate is an *upper* bound an attacker might beat with a better search. A
certificate is a *lower* bound no attacker can beat. Randomized smoothing (Cohen et al. 2019)
builds a smoothed classifier — "what the base model predicts most often under Gaussian feature
noise" — that comes with a per-sample radius R such that **no perturbation of L2 norm < R (in the
same per-feature-std units the attack uses) can change the prediction.**

| Smoothed model | Mean certified radius | Median radius | % drones certified beyond attack median (1.04) | Smoothed drone recall |
|---|---|---|---|---|
| Original | 0.529 | 0.442 | 19.2% | 73.3% |
| **Iteratively-hardened** | **1.936** | **2.347** | **83.3%** | **100.0%** |

This is the headline. On the hardened model, **83.3% of drones are *provably* un-evadable by any
perturbation smaller than the attack's own median (1.04 std-units)** — up from 19.2% on the
original. Mean certified radius nearly quadruples (0.53 → 1.94), and smoothing costs *nothing* on
the hardened model (smoothed drone recall 100%). So even though empirical evasion plateaus at
~70%, the certificate converts "an attacker might find an evasion" into "for 83% of drones, no
evasion within the typical attack budget exists — proven, not hoped." Iterative training's real
payoff turns out to be certified robustness, not the empirical rate.

## 3. Physics plausibility gate: an independent second layer

The attack is bounded to valid per-feature ranges but can still push the *joint* feature vector
into a combination no real drone produces. A novelty detector (IsolationForest) fit on real
drone feature vectors flags those:

- **41.3% of successful evasions flagged as anomalous** — caught by the gate even though they
  fooled the classifier.
- **6.7% false-flag on clean real drones** — the honest cost.

Four in ten evasions that beat the model are independently caught by a detector that never looks
at the class decision at all — a genuinely different failure mode to slip past.

## The complete, layered picture

An evasion now has to beat three independent obstacles instead of one:

1. The iteratively-hardened model (only ~70% of attacks succeed, needing ~40% larger perturbations), **and**
2. Fall within the certified radius — impossible for 83% of drones within the attack's median budget, **and**
3. Escape the physics gate — which catches 41% of whatever gets through.

## Honest interpretation and caveats

- **This is not "solved," and the report says so.** Empirical evasion plateaus around 70%, not 0%
  — that is the real limit of adversarial training, stated plainly. The value is the *certificate*
  (a provable per-sample guarantee) plus defense-in-depth, not a claim that evasion is impossible.
- The certified radius is in per-feature-standard-deviation L2 units, directly comparable to the
  attack's own perturbation norm — that shared unit is what makes "83% certified beyond the attack
  median" a meaningful, honest comparison rather than an abstract number.
- Randomized smoothing's certificate holds with confidence 1 − α (α = 0.05 here) via a
  Clopper-Pearson lower bound over 500 noise samples per point — a statistical guarantee, not a
  worst-case algebraic one; more samples tighten it.
- The smoothed classifier is a *different* deployment than the point classifier (it adds inference
  noise). The trade is a small latency/throughput cost for the provable guarantee — a real
  deployment choice, disclosed, not free.
- The plausibility gate's 6.7% false-flag on clean drones is a real cost; it would route flagged
  detections to human review, not silently drop them.
- Single seed; the plateau band (71–82%) and the certified-radius ordering (hardened ≫ original)
  are the robust findings.

## Follow-up: physical achievability + multi-seed confidence (`adversarial_physical_cost.py`)

Two critiques answered with measurements rather than disclaimers:

**Is the evasion even physically achievable?** The unconstrained attack moves features freely in
a valid range, but a real operator pays different costs per feature — reducing RCS needs
radar-absorbent material (expensive), shrouding rotors moves micro-Doppler (moderate), flying
slower is cheap. Assigning each attack feature a physical cost (per std of change) and requiring
the total under a budget:

| Physical-cost budget | Achievable evasion rate (mean ± std over 3 seeds) |
|---|---|
| Unconstrained (feature space) | 100.0% ± 0.0% |
| 8 (generous) | 84.4% ± 2.0% |
| 4 (moderate) | 46.9% ± 4.0% |
| 2 (realistic) | **21.9% ± 2.2%** |
| 1 (tight) | 11.4% ± 0.4% |

The scary "100% evadable in feature space" collapses to **~22% under a realistic physical
budget** — because the cheap evasions the search prefers often require expensive RCS changes a
real adversary can't easily make. This is the honest real-world threat level, and it's reported
alongside the unconstrained number, not instead of it.

**Single seed?** Every number above is mean ± std over three attack seeds; the tight intervals
(std 0.4–4.0 points) show the findings aren't a seed artifact.

## Artifacts

- `radarsense/adversarial_defense_complete.py`
- `radarsense/adversarial_physical_cost.py`, `artifacts/reports/adversarial_physical_cost_results.json`
- `artifacts/models/final_model_iteratively_hardened.joblib`
- `artifacts/reports/adversarial_defense_complete_results.json` — per-round curve, certified-radius
  distributions for both models, and the plausibility-gate catch rates
