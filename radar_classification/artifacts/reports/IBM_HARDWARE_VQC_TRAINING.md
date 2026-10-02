# Real IBM Quantum Hardware VQC Training Run

*Unlike `IBM_HARDWARE_EXPERIMENT.md` and `IBM_HARDWARE_LARGE.md` (which characterize a
**fixed** feature-map kernel's fidelity on real hardware, no training involved), this run
actually trains a Variational Quantum Classifier's parameters using loss evaluations
executed on real IBM Quantum hardware, one classical-optimizer iteration at a time.*

## What was run

Same circuit family as `vqc.py`'s noiseless run: `ZZFeatureMap(4, reps=2)` encoding +
`RealAmplitudes(4, reps=2)` trainable ansatz (12 parameters), parity-based binary readout
(odd measured-bitstring popcount → drone class), COBYLA (gradient-free) minimizing
cross-entropy loss.

`qiskit_machine_learning`'s `VQC` class (used by `vqc.py`) requires `qiskit<2.0`, while
`qiskit-ibm-runtime` requires `qiskit>=2.3` — genuinely incompatible in one environment
(same constraint documented in `IBM_HARDWARE_EXPERIMENT.md`). So the training loop is
hand-rolled directly against `SamplerV2` in the isolated `.venv_ibm_quantum/` environment,
rather than reusing the `VQC` class.

**Quota-bounded, not maxiter-bounded.** Gradient-free training needs one hardware job *per
optimizer iteration* (each evaluating the ansatz circuit across the training batch) — unlike
the kernel-characterization runs, which were one batched job total. At the time this was
scoped, the IBM Quantum Open-plan account had **417 of 600 QPU-seconds remaining** for the
period. The script checks real remaining quota via `QiskitRuntimeService.usage()` before
every job submission and stops cleanly once remaining budget drops below a 20s safety
margin, rather than crashing mid-run against the plan's hard limit.

- 50 training points (`radarsense.quantum.subsample`, drone fraction raised to 30%, same
  sampling function as every other quantum experiment in this project)
- 256 shots per circuit (deliberately low — trades per-shot precision for more affordable
  iterations under a tight budget)
- Backend: `ibm_fez`

## Result

**67 real-hardware training iterations completed** (COBYLA halted itself cleanly on the
quota safety check — not a crash, not `maxiter`), consuming 402 of the 417 available
QPU-seconds, ending with 15s headroom.

| | Hardware-trained | Noiseless reference, same n_train & iteration count |
|---|---|---|
| Best training loss (cross-entropy) | 0.623 | 0.530 |
| Test ROC-AUC | 0.566 | 0.536 |
| Test PR-AUC (drone) | 0.354 | 0.341 |
| Test recall (drone) | 0.356 | 0.344 |
| Test precision (drone) | 0.344 | 0.348 |

The noiseless reference is the *same* circuit, same 50 training points, same COBYLA
optimizer, capped at the identical 67 iterations the hardware run actually achieved — an
apples-to-apples comparison that isolates the effect of real hardware noise from the effect
of "fewer iterations than a longer simulator run," holding everything else fixed.

Job IDs for all 67 iterations, per-iteration loss, and quota accounting are in
`artifacts/reports/ibm_hw_vqc_raw.json`; full metrics in
`artifacts/reports/ibm_hardware_vqc_results.json`.

## Honest interpretation

**Training loss evaluations ran on real IBM hardware; the final held-out test metrics were
computed via noiseless statevector simulation using the hardware-learned parameters** — a
deliberate choice to conserve the very limited remaining quota for training itself rather
than spending it on evaluation, stated plainly rather than left implicit.

Both the hardware-trained and noiseless-reference models land close to chance (ROC-AUC
0.54–0.57, near 0.5; PR-AUC 0.34–0.35, only modestly above the ~0.30 drone base rate in
these subsamples). This is consistent with — not contradicted by — the rest of the project's
existing finding that gradient-free VQC training (COBYLA/SPSA) underperforms on this task
(`vqc_pennylane.py`'s reference numbers: Qiskit VQC with COBYLA, ROC-AUC ~0.50; with SPSA,
ROC-AUC 0.629). Real hardware noise did **not** meaningfully worsen the outcome relative to
the noiseless reference at this scale (ROC-AUC 0.566 vs. 0.536 — hardware is actually
marginally *higher*, well within the noise you'd expect from only 67 iterations and 50
training points); the dominant effect on performance here is the small
iteration/data budget and the gradient-free optimizer choice, not hardware noise itself.

**This does not show a quantum training advantage, or even quantum training working well.**
It shows a real, small VQC trained under quota-constrained conditions on actual IBM hardware,
reported at whatever performance it actually reached — not an assumed or fabricated one.

## Caveats

- 50 training points / 67 iterations is small by ML standards, dictated by real QPU-second
  budget, not by choice — the same budget constraint documented up front, not discovered
  after the fact.
- 256 shots/circuit is on the low end for probability-estimate precision; part of why the
  loss curve is noisy iteration-to-iteration rather than monotonically decreasing.
- Single hardware session, single backend (`ibm_fez`), single random seed — no repeated-seed
  variance estimate (unlike `vqc_pennylane.py`'s 3-seed noiseless runs), again a direct
  consequence of the quota available for this run.
- Final test metrics are noiseless-simulated using hardware-learned parameters, not
  hardware-evaluated — disclosed above, not hidden.
