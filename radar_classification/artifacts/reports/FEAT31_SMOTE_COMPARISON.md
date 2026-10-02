# SMOTE vs. Class-Weighting vs. Combined: 31-Feature Model

The 31-feature model has only ever used class-weighting for its minority-class (drone)
imbalance. The incumbent 11-feature model went through a full 7-way comparison
(`baseline` / `class_weight` / `smote` / `borderline_smote` / `adasyn` / `smote_tomek` /
`smote+class_weight`, [`optimize.py`](../../radarsense/optimize.py)) that the 31-feature
model never had. [`feat31_optimize.py`](../../radarsense/feat31_optimize.py) runs the
identical comparison — reusing `optimize.py`'s `fit_variant`/`predict`/`threshold_predict`/
`tune_threshold`/`SAMPLERS` functions unmodified, since they operate on arrays and a model
family name, not hardcoded 11-feature column names. Same protocol, not a re-implementation
that could silently diverge.

Evaluated on two slices: the overall test set (matching the incumbent's own comparison
scope) and the SNR ≤ -23dB deep-low-SNR slice specifically — the 31-feature model's entire
reason for existing.

## Result

| Variant | Overall recall | Overall precision | Overall PR-AUC | -23dB recall | -23dB precision | -23dB PR-AUC |
|---|---|---|---|---|---|---|
| baseline | 0.728 | 0.861 | **0.834** | 0.615 | 0.706 | 0.699 |
| **class_weight** (current) | 0.805 | 0.657 | 0.831 | 0.718 | 0.475 | **0.702** |
| smote | 0.785 | 0.668 | 0.819 | 0.718 | 0.509 | 0.674 |
| borderline_smote | 0.774 | 0.683 | 0.819 | 0.718 | 0.528 | 0.680 |
| adasyn | 0.795 | 0.701 | 0.816 | **0.769** | 0.556 | 0.682 |
| smote_tomek | 0.795 | 0.692 | 0.820 | 0.744 | 0.547 | 0.671 |
| smote+class_weight | 0.785 | 0.668 | 0.819 | 0.718 | 0.509 | 0.674 |

Numbers shown are raw-argmax predictions (pre-threshold-tuning), the same way the incumbent's
own 7-way comparison scores its variants — PR-AUC is threshold-independent and reflects
ranking quality across all thresholds, not the recall/precision at one specific operating
point.

**Two different variants win depending on scope**, which is itself the finding:

- **Overall test set: `baseline` wins** (highest PR-AUC, 0.834) — despite class_weight
  having *higher raw recall* (0.805 vs 0.728), baseline's higher precision (0.861 vs 0.657)
  gives it better ranking quality across all thresholds. After threshold-tuning specifically
  for recall (F-beta2, threshold=0.08), baseline reaches 85.6% drone recall at 96.8% overall
  accuracy — both slightly *better* than the currently-deployed class_weight variant's own
  tuned numbers (83.1% recall, 96.6% accuracy).
- **SNR ≤ -23dB: `class_weight` wins** (highest PR-AUC at this slice, 0.702) — even though
  `adasyn` has higher raw recall (0.769 vs 0.718) at this slice, class_weight's PR-AUC edge
  wins by the same ranking-quality criterion.

Neither imbalance strategy dominates on both axes. This wasn't visible in either model's
individual reporting until they were run against the same 7-way comparison and sliced by
SNR.

## What this means for deployment

Following the incumbent's own protocol (tune the threshold only for the variant that wins
*overall*, since that's the deployment scope), `baseline` — i.e., **no synthetic
oversampling and no class-weighting at all** — is the candidate that would replace the
currently-deployed `feat31_model.joblib` (class_weight only) if this recommendation were
adopted. It is **not** auto-swapped in; saved separately as
`feat31_model_optimized_baseline.joblib` for a deliberate decision, matching how
`feat31_compare.py` already treats the deployed 31-feature model as something not silently
overwritten by a new experiment.

If deployment instead prioritizes the deep-low-SNR regime specifically (the model's stated
purpose), the current class_weight choice is already the right one — it isn't wrong, it's
scope-dependent.

## Caveats

- Same SMOTE risk as the incumbent's own disclosure: SMOTE interpolates synthetic minority
  samples on **already-synthetic** data, which can amplify simulator artifacts rather than
  model anything about real low-SNR drone returns. Both variants' numbers here are real
  held-out results, not cherry-picked, but the risk applies equally to this comparison as to
  the original one.
- Single run, not repeated across seeds.
- "Baseline wins overall" depends on the PR-AUC-first, then-recall winner-selection
  criterion (matching the incumbent's exact protocol) — a criterion that explicitly favors
  ranking quality over raw-recall at the default threshold. A team that weights raw recall
  more heavily would read this table differently; the criterion is disclosed here rather
  than silently chosen to produce one preferred answer.

## Artifacts

- `radarsense/feat31_optimize.py` — comparison script
- `artifacts/reports/feat31_optimization_results.json` — full results (both slices, all
  variants, final tuned metrics)
- `artifacts/models/feat31_model_optimized_baseline.joblib` — candidate model (not deployed)
