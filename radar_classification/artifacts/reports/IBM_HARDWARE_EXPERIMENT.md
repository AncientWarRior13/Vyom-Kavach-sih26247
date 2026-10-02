# Real IBM Quantum Hardware Run

*Every quantum result elsewhere in this project (`quantum.py`, `vqc.py`, `diat_usat_cnn.py`,
`raddar_cnn.py`, `track_fusion.py`) uses noiseless exact statevector simulation — always
honestly disclosed as such. This is the one real-hardware measurement in the project.*

## What was run

The exact same `ZZFeatureMap(feature_dimension=4, reps=2)` circuit used throughout the
tabular quantum-kernel work, executed on **`ibm_fez`** (156-qubit IBM Quantum processor),
via the IBM Cloud channel (`qiskit-ibm-runtime`).

Fidelity kernel estimation on real hardware can't read a statevector directly (that's a
simulator-only capability) — it uses the standard inversion test instead: for each pair of
points `(x_i, x_j)`, build a circuit that applies the feature map for `x_i` then the
*inverse* feature map for `x_j`, measure all qubits, and take the fraction of shots landing
on the all-zeros outcome as the fidelity estimate `|⟨ψ(x_j)|ψ(x_i)⟩|²`.

- 6 real data points (`radarsense.quantum.subsample`, same sampling function as every other
  quantum experiment), 15 unique pairwise fidelities
- 2048 shots per circuit, all 15 circuits submitted as one batched job (not 15 separate
  jobs — minimizes queue/scheduling overhead)
- Job ID `d9f6pskjeosc73fji2j0`, completed in 172 seconds end-to-end

**Scale note, stated plainly:** this is deliberately small — 6 points, not the n=800 used in
the noiseless scaling study. Free-tier IBM Quantum access has real, limited monthly quota;
reproducing the full study on hardware would require far more shots and queue time than is
sensible to spend proving a point that a smaller, honest sample already answers.

## Result

| | Value |
|---|---|
| Mean absolute deviation (real hardware vs. noiseless simulation) | **0.020** |
| Pearson correlation (does hardware preserve fidelity *ordering*) | **0.920** |

Real hardware fidelities track the noiseless-simulated ones closely in relative terms
(r=0.92) — pairs the simulator calls "more similar" are still generally called "more
similar" by real hardware — while absolute magnitudes shift by a small but real amount
(mean deviation 0.02, on fidelity values ranging roughly 0.005–0.19 in this sample). This is
consistent with expected NISQ-era behavior: gate errors and readout noise perturb the exact
fidelity value without completely destroying the kernel's relative geometry.

## Honest interpretation

This is a **noise characterization**, not a retrained hardware classifier — with only 15
pairwise samples, it doesn't support claims like "the quantum kernel classifier performs at
X% on real hardware." What it *does* support: the specific 4-qubit, reps=2 ZZFeatureMap
circuit already used throughout this project's noiseless experiments produces fidelity
estimates on real IBM hardware that correlate strongly (r=0.92) with the noiseless values —
real evidence that the noiseless simulation results reported elsewhere aren't wildly
detached from what actual hardware would produce, even though they're not identical.

## Caveats

- One backend (`ibm_fez`), one calibration snapshot in time — real hardware noise profiles
  drift day to day; this is a single measurement, not a stable characterization.
- 15 pairs is enough to estimate a correlation, not enough to bound it tightly — a wider
  study (more points, repeated across backends/days) would be needed for a stronger claim.
- No error mitigation applied (no zero-noise extrapolation, no readout-error correction) —
  a straightforward measurement, which is itself the honest baseline to report.
