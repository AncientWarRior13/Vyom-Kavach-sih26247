# Feature-Dropout Training: A Direct Fix for Micro-Doppler Masking

`THREAT_MODEL.md`'s one clean 11-feature-model vulnerability: masking collapses drone recall
from 91.8% to 12.3% because `micro_doppler_bandwidth` is the dominant SHAP feature — the model
leans on it almost as a sufficient statistic. Direct fix, not a workaround: teach the model
during training that a low micro-Doppler reading doesn't always mean "not drone."

## Method

CatBoost has no built-in feature dropout the way a neural net does; the practical equivalent
for tree ensembles is data augmentation. Took 50% of drone training rows (455 of 910), created
masked duplicates with `micro_doppler_bandwidth`/`micro_doppler_jitter` attenuated by a random
factor in the same 0.3–0.9 range the attack itself uses, **kept the label as "drone"** (still a
real target, just a degraded measurement), mixed into training, retrained with the same
CatBoost config as the deployed model.

Re-tested against the exact same masking stress test as `robustness.py` (×0.7/×0.5/×0.3,
directly comparable to the reported numbers).

## Result: the vulnerability is substantially, not just marginally, reduced

| Attenuation | Original model | Feature-dropout-trained |
|---|---|---|
| Unperturbed | 91.8% | 88.2% |
| ×0.7 | 76.4% | **84.6%** |
| ×0.5 | 51.8% | **77.4%** |
| **×0.3** | **12.3%** | **64.1%** |

At the severe ×0.3 case — the one `THREAT_MODEL.md` called a "known, documented vulnerability"
— recall goes from 12.3% to **64.1%, a more than 5× improvement.** This is a substantially
larger effect than the 42-feature combined model's masking robustness (16.8% at ×0.3, see
`FEAT42_MASKING_ROBUSTNESS.md`) — directly retraining to reduce reliance on the vulnerable
feature works better here than the redundancy-from-fusion approach, at least for this specific
attack.

**Real cost, disclosed**: unperturbed drone recall drops 3.6 points (91.8%→88.2%). The
standard robustness-accuracy trade-off — the model is deliberately less "peaked" on its single
strongest signal, giving up a little normal-case performance for much better worst-case
performance under this specific attack.

## Honest interpretation

This is the most effective single mitigation tested against micro-Doppler masking so far —
more effective than feature fusion (42-feature model), and directly targets the mechanism
`THREAT_MODEL.md` already diagnosed (over-reliance on one dominant feature) rather than working
around it. Combining this WITH the 42-feature model's fusion-based robustness (not tested here)
is a natural next step that could plausibly do even better, since the two mechanisms are
different (regularizing away over-reliance vs. providing redundant signal).

Not a complete fix: 64.1% recall at severe masking is still a real degradation from the 88.2%
baseline, and this is one specific attack (uniform multiplicative attenuation) — a differently-
shaped countermeasure might still exploit the residual reliance on micro-Doppler that remains.

## Caveats

- One dropout fraction (50%) and one attenuation range (0.3–0.9×) tested — not swept. A
  higher dropout fraction might trade more accuracy for more robustness, or vice versa.
- Single run, not repeated across seeds.
- Retrained model saved separately (`final_model_feature_dropout_trained.joblib`), not
  auto-deployed in place of the current `final_model.joblib`.
- Not yet combined with adversarial training (`ADVERSARIAL_TRAINING.md`) or tested against the
  joint-feature adversarial search (`ADVERSARIAL_ROBUSTNESS.md`) — this specifically targets
  the single-feature masking attack, not the broader joint-perturbation vulnerability.

## Artifacts

- `radarsense/feature_dropout_training.py`
- `artifacts/reports/feature_dropout_training_results.json`
- `artifacts/models/final_model_feature_dropout_trained.joblib`
