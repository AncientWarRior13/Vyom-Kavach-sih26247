# Classical+Quantum Ensemble: Does Combining Beat Picking a Winner?

Every classical-vs-quantum comparison elsewhere in this project picks a winner between the
two kernels. Nobody had tried combining their predictions. Tested here on RAD-DAR CNN
embeddings (`raddar_cnn.keras`, already trained — no retraining), reusing the exact
embedding-extraction/subsampling/kernel-fitting machinery from `raddar_cnn.py`.

## Method

Three ensemble strategies, evaluated on the SAME held-out half of each seed's test pool as
both individual methods (a 300-sample pool per seed, split into a 150-sample stack-fit half
used only to train the stacking meta-model, and a 150-sample final-eval half every method is
scored on — apples-to-apples):

1. **Simple average** of P(drone) from both kernels.
2. **Weighted average**, swept over a weight grid — lets an asymmetric split win if one exists.
3. **Stacking**: a logistic-regression meta-model trained on `[quantum_p_drone,
   classical_p_drone] → label`.

5 seeds × 3 train sizes (200/400/800), matching `raddar_cnn.py`'s own multi-seed discipline.

## Result: no PR-AUC win, but a real, consistent recall trade-off in the ensemble's favor

**By PR-AUC (ranking quality, threshold-independent), no ensemble method beats classical RBF
alone** at any train size:

| n_train | Best individual | Best ensemble | Ensemble wins? |
|---|---|---|---|
| 200 | classical_rbf (0.9616) | weighted-avg (0.9618) | Marginal — within noise (std ≈ 0.025) |
| 400 | classical_rbf (0.9573) | weighted-avg (0.9557) | No |
| 800 | classical_rbf (0.9547) | weighted-avg (0.9512) | No |

A first single-seed smoke test suggested a clear ensemble win (PR-AUC 0.986 vs. 0.967) — the
full 5-seed run shows that was largely a lucky draw, not a real effect. Worth stating plainly
since it's exactly the kind of result a less careful pass would have reported.

**But drone recall tells a different, real story.** Paired seed-by-seed (same train/test
split, same seed, every method scored on it), stacking's recall vs. classical alone:

| n_train | Stacking wins | Ties | Stacking loses |
|---|---|---|---|
| 200 | 4/5 | 1/5 | 0/5 |
| 400 | 4/5 | 1/5 | 0/5 |
| 800 | 2/5 | 2/5 | 1/5 |

10 wins, 4 ties, 1 loss across 15 seed-pairs — a consistent, real lean toward higher recall,
strongest at smaller train sizes (where classical alone is comparatively weaker) and weakening
at n=800 (where classical alone is already strong, leaving less room to add value). Mean
recall: stacking 96.9–97.8% vs. classical alone 95.6–96.0% across the three sizes — a genuine
1.5–2.2 point gain, at a smaller precision cost (stacking 92.8–93.4% vs. classical's
93.8–94.4%, roughly 1 point lower).

**This is exactly the trade this project's own cost function favors.** The deployed cost
matrix elsewhere (`config.py`: `COST_FN_DRONE=25, COST_FP_DRONE=1`) treats a missed drone as
25× worse than a false alarm — trading ~1 point of precision for ~2 points of recall is a
favorable trade under that asymmetry, not a neutral one.

**The stacking meta-model genuinely uses both signals**, not just picking one: learned
coefficients across seeds are comparable in magnitude for quantum (2.5–3.2) and classical
(2.4–3.5) — consistent with the feat42 SHAP/LIME finding elsewhere in this project that
combining complementary signals, not picking a winner, is where real gains show up.

## Replicated on DIAT-uSAT (quantum's WINNING case) — the pattern holds, even more cleanly

The caveat below originally flagged this as untested: does the same recall-favoring pattern
hold on DIAT-uSAT, where quantum actually beats classical on its own (unlike RAD-DAR, its
weak case)? Tested with the identical method (`classical_quantum_ensemble_diat.py`, same 3
strategies, 5 seeds × 3 train sizes, reusing the already-trained `diat_usat_cnn.keras`, no
retraining) — same drone-vs-rest framing extracted from the 4-class kernel output.

**PR-AUC**: same story as RAD-DAR — no ensemble consistently beats classical alone (wins
narrowly at n=200, loses at n=400 and n=800).

**Recall**: **even cleaner than RAD-DAR** — stacking beats classical alone in **15/15 paired
seed comparisons** (5/5 at every one of the three train sizes), zero ties, zero losses. Mean
recall: stacking 65.0–68.9% vs. classical alone 58.6–62.7% across the three sizes — a 4–6
point gain, at a small precision cost (stacking 70.7–71.9% vs. classical's 72.1–72.7%, roughly
0.4–1.4 points lower). The recall-favoring pattern isn't an artifact of RAD-DAR specifically —
it replicated on a second, independent real sensor, with an even stronger, more consistent win
rate.

One nuance worth noting from `ensemble_decision_explainer.py`'s aggregate analysis (below):
even on DIAT-uSAT, where quantum wins standalone, the stacking meta-model still weights
classical *more* heavily than quantum at every seed (coefficients roughly 2–6× larger for
classical) — the ensemble isn't simply "trusting whichever model is individually stronger,"
it's learning a specific combination that happens to favor classical's vote while still using
quantum's.

## Does calibrating the probabilities first help further? Yes, measurably

A natural follow-up question: the stacking meta-model trusts each kernel's *raw* probability
as a genuine confidence signal. If a model's confidence is miscalibrated, the ensemble could be
misled by loud-but-hollow confidence rather than real reliability.

Checked directly (`ensemble_calibration_check.py`, pooling predictions across all 15 RAD-DAR
seed/train-size runs, 8,976 pooled predictions): **both kernels are already reasonably well-
calibrated in aggregate.** When quantum says ≥80% confident, it's actually right 94.7% of the
time (n=1,640); classical, 96.7% (n=1,881) — both close to or above their claimed confidence,
not meaningfully overconfident on average. So the specific failure case documented below
(quantum confidently wrong on one detection) is a real individual miss, not evidence of
systematic miscalibration.

**But isotonic-calibrating each kernel's probabilities before stacking still measurably
improves the ensemble**, even though both were already decently calibrated:

| | Recall | Precision | PR-AUC |
|---|---|---|---|
| Stacking on raw probabilities | 93.8% | 94.2% | 0.951 |
| Stacking on calibrated probabilities | **96.0%** | 92.5% | **0.955** |

+2.2 recall points at ~1.7 precision points cost — the same favorable direction as the base
finding, stacking on top of it. Calibrating first is cheap (one isotonic fit per kernel on a
held-out split) and a real, not merely theoretical, improvement.

## Is the ensemble's decision auditable, or a black box?

Built `ensemble_decision_explainer.py` specifically so the ensemble isn't just an opaque
probability: for each detection, it logs what each individual kernel said, what the ensemble
said, and WHY —

- **The meta-model's contribution is EXACT, not approximated**: since stacking is a 2-input
  logistic regression, `coefficient × probability` for each kernel is the literal, exact
  reason the ensemble leaned where it did — no SHAP/LIME estimation needed for that part,
  because nothing about it is opaque to begin with.
- **What's genuinely opaque — why each kernel made ITS OWN call — is explained via LIME** on
  the 8-dim CNN embedding, run separately per kernel. Classical and quantum turn out to lean on
  the *same* four embedding dimensions (just reordered by relative weight), not different ones
  — meaning they're reading the same underlying signal and mostly agreeing (only 6/150 eval
  detections had them split on the hard label), with the ensemble's value coming from how they
  weight that shared signal differently in the disagreement cases, not from looking at
  unrelated information.
- On a 20-detection log (oversampling the rare disagreement cases on purpose, since that's
  where "why" matters most), the ensemble was correct 18/20 — including one honestly-reported
  failure case: a real `not_drone` detection where classical was confidently right (p=0.06) and
  quantum was confidently wrong (p=0.80), and the ensemble followed quantum into the wrong
  answer (p=0.54). Not every disagreement resolves correctly; this is shown, not hidden.

Full per-detection log: `artifacts/reports/ensemble_decision_log.json`.

## Honest interpretation

- The headline metric that gets reported everywhere in this project's other comparisons
  (PR-AUC) says ensembling doesn't help. A different, also-legitimate metric (drone recall,
  the project's own stated minority-class priority) says it does, consistently, in a
  real paired comparison. Neither framing is wrong — they're answering different questions,
  and reporting only one would have been misleading either way.
- The effect is real but modest (1.5–2.2 recall points) and shrinks as training data grows —
  this is not a dramatic finding, but it's a genuine, replicated one, not noise.
- Simple averaging and weighted averaging show the same directional recall lean as stacking
  but slightly weaker — stacking's ability to learn a nonlinear-ish combination (via the
  logistic link) captures a bit more of the available signal than a fixed linear blend.

## Caveats

- Stack-fit half is only 150 samples — a small set to fit even a 2-parameter logistic
  regression on; a larger stacking-fit budget might sharpen the effect further.
- PR-AUC and recall-at-default-threshold are reported; no threshold retuning was done for the
  ensemble outputs specifically (unlike the deployed models elsewhere, which all get a tuned
  drone threshold) — an ensemble-specific threshold tune might change the precision/recall
  trade shown here.
- The calibration finding (raw vs. calibrated stacking) was checked on RAD-DAR only, not yet
  repeated on DIAT-uSAT.
- The decision-log explainability check was run on RAD-DAR only (20 detections, one
  n_train/seed configuration) — not yet repeated on DIAT-uSAT, and not exhaustive (full
  150-detection eval pool was not logged, only a sample oversampling disagreement cases).

## Artifacts

- `radarsense/classical_quantum_ensemble.py` (RAD-DAR), `classical_quantum_ensemble_diat.py`
  (DIAT-uSAT replication), `ensemble_calibration_check.py`, `ensemble_decision_explainer.py`
- `artifacts/reports/classical_quantum_ensemble_raddar_results.json`,
  `classical_quantum_ensemble_diat_results.json`, `ensemble_calibration_check_results.json`,
  `ensemble_decision_log.json`
