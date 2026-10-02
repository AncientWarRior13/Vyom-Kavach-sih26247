# Adversarial Training: Does Retraining on Found Evasions Help?

Follow-up to `ADVERSARIAL_ROBUSTNESS.md`'s 100% evasion finding — the standard defense against a
demonstrated evasion attack: augment training data with adversarial examples, correctly
labeled, so the model learns a decision boundary the attack doesn't find gaps in.

## Method

1. Ran the exact same black-box search as `adversarial_robustness.py`, this time against 300
   **training** drones (test set stays untouched for honest re-evaluation) — harvested 300/300
   real, successful evasion perturbations.
2. Applied each found perturbation direction to 3 additional real drone training rows (still
   physically drones, correctly labeled "drone" — only the measured feature vector shifted),
   generating 900 adversarially-augmented training examples.
3. Retrained CatBoost on original + augmented data, same config as the deployed model
   (`train_baseline.py`'s catboost params, `optimize.py`'s balanced class-weighting).
4. Re-ran the exact same attack (same 200 held-out test drones, same search config) against the
   retrained model.

## Result: a real, partial improvement — not a fix

| | Before | After adversarial training |
|---|---|---|
| Evasion success rate | **100%** (179/179) | **80.1%** (145/181) |
| Median perturbation needed | 1.04 std-units | **1.55 std-units** (+50%) |

Both numbers moved in the right direction: fewer drones can be evaded at all, and the ones that
still can require a meaningfully larger perturbation to pull off. This is a genuine, measured
improvement, not a complete defense — 4 in 5 attacked drones are still evadable.

**Sanity check — normal performance held up, even improved slightly**:

| | Accuracy | Drone recall |
|---|---|---|
| Original model | 97.59% | 91.79% |
| Retrained model | 97.09% | **92.82%** |

A small accuracy trade (−0.5pt) for a small recall gain (+1.0pt) — the adversarial examples,
still correctly labeled drones, function partly as useful data augmentation near the decision
boundary, not purely as a robustness tax.

## Honest interpretation

Adversarial training measurably raises the cost of the specific attack already characterized —
20 points fewer evadable, ~50% more perturbation required for the ones that still work — without
meaningfully hurting the model's real job. That's a legitimate, worthwhile improvement.

It is **not** a solved problem. 80% evasion is still a real, exploitable vulnerability, and this
is the well-known limitation of adversarial training generally: it hardens the model against
the *specific* perturbations it was shown, not against the full space an adaptive attacker could
still search. An attacker re-running the same black-box search against *this* retrained model
would very plausibly find a new set of successful (if now somewhat larger) perturbations — that
loop (attack, retrain, re-attack) is the standard adversarial-training arms race, and one cycle
of it was run here, not the full cycle to convergence.

## Caveats

- One round of adversarial training, not iterative (attack → retrain → attack → retrain...).
  Each additional round would likely raise the cost further, with diminishing returns.
  Not run here due to time cost (harvesting + retraining + re-attacking is ~5 minutes per round).
- Augmentation used 300 harvested directions × 3 target rows each (900 total) — a modest
  augmentation budget; more harvested directions or more targets per direction might close the
  gap further.
- Tested only against the 11-feature model. Not yet repeated for the 31-feature, 42-feature, or
  quantum-kernel backends.
- Retrained model saved separately (`final_model_adversarially_trained.joblib`), not
  auto-deployed in place of the current `final_model.joblib` — that's a decision to make
  deliberately.

## Artifacts

- `radarsense/adversarial_training.py`
- `artifacts/reports/adversarial_training_results.json`
- `artifacts/models/final_model_adversarially_trained.joblib`
