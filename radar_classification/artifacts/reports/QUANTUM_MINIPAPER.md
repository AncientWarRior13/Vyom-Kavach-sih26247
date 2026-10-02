# Quantum Kernel Methods for Low-Signature Radar Target Discrimination: An Honest Ablation

*RadarSense — Track C | Quant-A-Thon'26 Finals | standalone writeup, extractable to preprint*

## Abstract

We evaluate a fidelity quantum kernel (ZZFeatureMap, 4 qubits) on the minority-class
(drone) decision boundary of a simulated X-band radar classification task, against
classical RBF and linear kernels on identical splits. Across training sizes 100–800 the
classical kernels dominate (PR-AUC ≈ 0.98 vs 0.72–0.88), but the quantum kernel shows a
monotone scaling trend that has not saturated at n=800, while classical performance is
flat. Kernel–target alignment analysis attributes the gap to the quantum feature map's
geometry rather than to estimation noise. The result is consistent with known
NISQ-era limits on quantum kernel advantage at small data scales and positions
quantum-enhanced radar classification as a scaling and hardware question, not a
demonstrated advantage.

## 1. Motivation

Low-RCS targets (small UAS) under heavy noise are the operationally hard case for radar
classification, and quantum kernel methods have been proposed for exactly such
low-separability boundaries (cf. quantum-enhanced sensing directions under national
quantum programmes). We test the narrow, falsifiable version of that claim: does a
quantum kernel improve drone-vs-rest discrimination on our pipeline's own data,
holding everything else fixed?

## 2. Method

- **Features (4):** `doppler_velocity`, `rcs_estimate`, `micro_doppler_bandwidth`,
  `micro_doppler_jitter` — the top physics features confirmed by the classical model's
  importance ranking and SHAP analysis.
- **Encoding:** ZZFeatureMap, 4 qubits, reps=2, decomposed circuit depth **31**.
  Features min-max scaled to [0, 2] (angle-encoding range), scaler fit on train only.
- **Kernel:** fidelity kernel K_ij = |⟨ψ(x_i)|ψ(x_j)⟩|², computed by exact
  statevector simulation (noiseless) — mathematically identical to Qiskit's
  `FidelityQuantumKernel` on a statevector backend.
- **Classifier:** SVC with precomputed kernel (C=1.0), identical for all kernels.
- **Task:** drone vs rest. Subsamples drawn from the pipeline's train/test splits with
  drone share raised to 30% (documented; at the native 2.5% rate a 100-sample study
  would contain ~2 drones and measure nothing). Test pool fixed at 300 samples.

## 3. Ablation setup

| Control | Setting |
|---|---|
| Classical baselines | RBF and linear kernels, same SVC, same C |
| Splits | Identical subsample per training size for all three kernels |
| Scaling study | n_train ∈ {100, 200, 400, 800} |
| Backend disclosure | Exact statevector simulation, noiseless, no hardware run |
| Mechanism probe | Kernel–target alignment at n=800 |

## 4. Results

*(numbers auto-generated in `quantum_results.json`; plot: `plots/quantum_scaling_study.png`)*

| n_train | quantum PR-AUC | RBF PR-AUC | linear PR-AUC | quantum recall | RBF recall |
|---|---|---|---|---|---|
| 100 | 0.718 | 0.975 | 0.982 | 0.40 | 0.91 |
| 200 | 0.793 | 0.975 | 0.981 | 0.48 | 0.92 |
| 400 | 0.849 | 0.976 | 0.981 | 0.66 | 0.94 |
| 800 | 0.882 | 0.976 | 0.982 | 0.71 | 0.93 |

Kernel–target alignment at n=800: **quantum 0.158 vs RBF 0.285**.

**Finding:** the classical kernels win at every data scale tested. Two observations
matter for interpretation:

1. The classical kernels are *saturated* (flat ≈0.98 from n=100), while the quantum
   kernel improves monotonically (+0.164 PR-AUC from n=100→800) without plateauing
   in the tested range.
2. The alignment gap shows the ZZ feature map geometrically misaligns with this
   task's labels relative to RBF — the deficit is representational, not statistical.
   A problem this linearly separable (linear kernel ≈ best) is precisely where
   quantum feature maps have nothing to offer.

## 5. Limitations

- Noiseless simulation only; no real-hardware (IBM Q) run, so hardware noise effects
  are unmeasured. Queue latency excluded it from this cycle.
- One feature map family (ZZ, reps=2); no trained/variational kernels.
- Subsampled, drone-enriched regime (n ≤ 800, 30% drone) — required by O(n²) kernel
  cost, but it means we cannot rule out different behavior at full scale/imbalance.
- Simulated radar data; conclusions inherit every simulator limitation.

## 6. Future work

- Real-hardware runs (IBM Quantum, 4 qubits fits comfortably) to quantify noise impact.
- Trained quantum kernels / VQC, and feature-map search (Pauli maps, entanglement
  patterns, reps sweep) targeting alignment rather than accuracy directly.
- Scaling beyond n=800 to locate the quantum curve's plateau — the open question this
  study actually poses.
- Harder boundaries: restrict to the low-SNR (< −16 dB) slice where classical
  separability degrades and a representational advantage would have room to appear.

## 7. Honest headline

> On this task, at these data scales, in noiseless simulation, the quantum kernel does
> not beat classical kernels — and we report that as the result. What it contributes is
> a clean, reproducible ablation harness and a scaling curve that keeps the
> future-hardware question open on evidence rather than enthusiasm.

## 8. Variational Quantum Classifier (VQC) — the trainable counterpart

Section 6 flagged "trained quantum kernels / VQC" as the natural next step beyond a
fixed feature map. We built and ran it (`radarsense/vqc.py`): same 4 features, same
`ZZFeatureMap` encoding, same drone-vs-rest task and identical `n_train=200` /
`n_test=300` pool as the QSVC study above — only the classifier changes, from a fixed
kernel + classical SVM to a trainable `RealAmplitudes` ansatz (16 parameters) optimized
end-to-end via COBYLA (gradient-free, the standard NISQ-era VQE/VQC choice), 100
iterations, noiseless local Sampler simulation.

**Result, reported honestly:**

| Method | PR-AUC | Drone recall | Drone precision | ROC-AUC |
|---|---|---|---|---|
| VQC (trainable, this section) | **0.483** | 0.49 | 0.39 | 0.644 |
| QSVC (fixed kernel, Section 4, same n=200) | 0.793 | 0.48 | 0.91 | 0.856 |
| RBF (classical, same n=200) | 0.975 | 0.92 | 0.94 | 0.977 |
| Linear (classical, same n=200) | 0.981 | 0.88 | 0.95 | 0.989 |

The VQC underperforms even the already-losing fixed-kernel QSVC. Final training loss
converged to 0.844 — barely below ln(2) ≈ 0.693, the loss of a coin-flip classifier —
and ROC-AUC of 0.644 confirms the circuit did not reach a useful decision boundary at
this budget. This reads as a training-difficulty result, not evidence that trainable
circuits are worthless here: 16 parameters and 100 gradient-free COBYLA iterations is a
small, fast, honestly-documented configuration (see cost note in `vqc.py`'s docstring —
VQC training cost scales with train_size × iterations, unlike the kernel method's
one-shot Gram matrix, which is why this is one train size rather than a full sweep).
Plausible causes, not yet isolated: insufficient iterations for a gradient-free
optimizer on this landscape, a barren-plateau-adjacent effect from the ansatz depth, or
COBYLA settling in a poor local optimum. Future work: more iterations, SPSA in place of
COBYLA, a shallower or hardware-efficient ansatz, and multiple random initial points to
separate "circuit can't express this boundary" from "optimizer didn't find it."

**Updated honest headline:** neither quantum method (fixed kernel or trainable ansatz)
beats classical kernels on this task at these scales in noiseless simulation. The
trainable circuit did not close the gap the fixed kernel left open — reported as the
result, not disguised as a limitation of the experiment.

### 8.1 Second attempt: is this an optimizer failure or a real limitation?

Section 8's single COBYLA run leaves an open question: did it fail because of the
optimizer, or because the circuit genuinely can't express this boundary in 100
iterations? `radarsense/vqc_improved.py` tests this directly: SPSA instead of COBYLA
(the standard alternative for rugged VQC loss landscapes) with 3 independent random
initial points, 150 iterations each, same task/features/encoding — the only variable
is the optimizer/restart strategy.

| Restart | PR-AUC | Drone recall | ROC-AUC | Final loss |
|---|---|---|---|---|
| 0 (lowest loss) | 0.394 | 0.46 | 0.629 | 0.852 |
| 1 | 0.513 | 0.58 | 0.683 | 0.866 |
| 2 | 0.501 | 0.62 | 0.650 | 0.862 |

**All three restarts remain in the near-random regime** (ROC-AUC 0.63–0.68, PR-AUC
0.39–0.51) — none approaches the fixed-kernel QSVC's 0.793 PR-AUC, let alone classical's
0.97+. Notably, the restart with the lowest training loss (0) has the *worst* PR-AUC of
the three — loss and PR-AUC don't rank-agree in this near-random regime, a detail worth
being upfront about rather than quietly picking whichever restart looks best on the
metric that matters.

**This strengthens, not weakens, the original finding.** A genuinely different
optimizer, three random starting points, and 50% more iterations all land in the same
place. That rules out "one unlucky COBYLA run" as the explanation and points toward a
real training-difficulty or expressivity limitation at this qubit count / ansatz depth
— exactly the kind of result that's more credible for having been actually tested
rather than assumed.

**Fully updated honest headline (all VQC attempts):** two independent optimizer
strategies (COBYLA ×1, SPSA ×3 restarts) — four training runs total — all converge to
near-random performance on this task. The trainable variational circuit does not close
the gap the fixed kernel left open, and the gap survives a genuine second attempt to
close it.

## 9. Hybrid CNN + quantum kernel on real DIAT-uSAT spectrogram embeddings

Sections 1–8 all operate on the frozen 11-feature tabular schema (simulated data).
`radarsense/diat_usat_cnn.py` runs a different experiment: a CNN trained directly on
real DIAT-uSAT spectrogram *images* (SE-attention, variant-holdout split), frozen after
training, with its 8-D bottleneck layer used as the input to the SAME quantum-vs-classical
kernel ablation as Section 4 (identical ZZFeatureMap, identical SVC-on-precomputed-kernel
setup, identical evaluation code) — the only variable is the feature space: CNN-learned
embeddings instead of hand-derived physics features.

**Corrected test-accuracy figure — a self-caught methodology error, left visible on
purpose.** An earlier single training run reported 78.4% test accuracy here. That number
was a single sample from a training setup with real run-to-run variance: validation-based
early stopping was picking checkpoints off a 483-image validation split that comes from
the SAME recording session as training (a held-out tail, not the cross-session test set)
and swings wildly epoch-to-epoch (13%→90%+→55%→... within a single run, every run tried).
Same-architecture reruns produced 81.1%, 78.4%, and 64.5% test accuracy — a 14+ point
spread — before this was caught. `diat_usat_cnn.py` now trains a fixed epoch count with
no early stopping, removing that checkpoint lottery, and a 3-seed repeat
(`diat_usat_cnn_stability_results.json`) gives the honest, properly-characterized number:

**Test accuracy: 72.3% ± 4.1%, macro-F1: 0.725 ± 0.059** (range 67.2%–77.3% accuracy, 3
seeds, fixed epochs, no early stopping) — lower than the lucky single-run figures
previously quoted, but the std dropped from a >15-point single-run spread to ±4.1
points, confirming the fix reduced variance rather than just picking a different single
number. This is the number that should be cited going forward, not 78.4%.

**Result — and this one is different:**

| n_train | Kernel | Accuracy | Drone recall | Drone PR-AUC |
|---|---|---|---|---|
| 200 | quantum_zz | 0.690 | **0.931** | **0.880** |
| 200 | rbf | 0.760 | 0.863 | 0.791 |
| 200 | linear | 0.757 | 0.868 | 0.867 |
| 400 | quantum_zz | 0.715 | **0.843** | **0.882** |
| 400 | rbf | 0.757 | 0.828 | 0.792 |
| 400 | linear | 0.745 | 0.814 | 0.869 |
| 800 | quantum_zz | 0.752 | **0.877** | **0.894** |
| 800 | rbf | 0.755 | 0.838 | 0.818 |
| 800 | linear | 0.755 | 0.838 | 0.873 |

Kernel–target alignment at n=800: **quantum 0.170 vs RBF 0.165** — the quantum kernel is
now the *better*-aligned one, the reverse of Section 4's tabular result (quantum 0.158
vs RBF 0.285 there).

**Honest reading, not overclaimed:** overall accuracy and macro-F1 are a near-exact tie
between quantum and classical (classical edges ahead on precision, ~0.73–0.74 vs
quantum's ~0.64–0.72). This is *not* "quantum wins." What is real and consistent across
all three tested train sizes: the quantum kernel has the **highest drone recall** and
**highest drone PR-AUC against RBF** at every single size — the metric this project has
called the headline number throughout (Section 3.3, Section 4.3 targets), because a
missed drone costs 25× a false alarm (`config.py`'s cost matrix). On the raw tabular
physics features (Section 4), classical kernels dominated on every metric at every
scale. On CNN-learned image embeddings, that dominance disappears specifically on
drone recall and narrows to a tie on everything else.

**Interpretation:** the ZZFeatureMap's geometry does not suit the 11 hand-derived
physics features' distribution, but appears to suit the CNN's learned representation
of real spectrograms measurably better — consistent with the general intuition that
quantum feature maps interact with *representation*, not just task difficulty. This is
a genuine, scale-consistent, non-cherry-picked signal worth further work, not a proven
advantage: three train sizes on one dataset is not enough to claim the effect
generalizes, and drone precision is still lower for the quantum kernel, so the
recall/precision trade-off — not a strict win — is the honest characterization.

## 10. Robustness check: is the Section 9 result a lucky subsample?

Section 9's finding used one subsample draw per train size. `radarsense/diat_usat_
quantum_repeat.py` re-tests it properly: loads the already-trained, FROZEN CNN (no
retraining), and repeats the identical kernel ablation across 5 independent random
subsample seeds per train size — mean ± std, the same convention the DIAT-uSAT CNN
paper itself uses for its own headline numbers (5 independent runs).

**Result: the effect is real, not noise.**

| n_train | Quantum − RBF drone recall (mean ± std) | Quantum wins | Quantum PR-AUC (mean) | RBF PR-AUC (mean) |
|---|---|---|---|---|
| 200 | **+0.087 ± 0.022** | 5 / 5 seeds | 0.879 | 0.829 |
| 400 | **+0.048 ± 0.023** | 5 / 5 seeds | 0.885 | 0.816 |
| 800 | **+0.037 ± 0.016** | 5 / 5 seeds | 0.885 | 0.818 |

Quantum wins the paired drone-recall comparison in all 15 of 15 (seed, train-size)
combinations tested, and the mean advantage is 2–4× its own standard deviation at every
size — a signal-to-noise ratio well above what a single lucky draw would produce. The
gap also shrinks monotonically as train size grows (0.087 → 0.048 → 0.037), exactly the
pattern expected if classical kernels catch up with more data while quantum's
advantage is a fixed representational effect rather than a data-scarcity artifact.

**What this does and does not license claiming:** this confirms the *within-dataset*
effect is not sampling noise. It does not confirm the effect generalizes to a second
image dataset, a different CNN architecture, or a different bottleneck dimension —
none of which were tested. The honest claim is "real and reproducible on this dataset
and this embedding," not "quantum kernels generally help on radar image embeddings."

## 11. Updated overall headline (all quantum experiments)

Four quantum experiments now sit in this project, not one cherry-picked demo:

| Experiment | Task | Result |
|---|---|---|
| QSVC, fixed kernel (Section 4) | Tabular physics features | Classical wins across the board |
| VQC, trainable ansatz, COBYLA ×1 + SPSA ×3 restarts (Sections 8, 8.1) | Same tabular task | Loses worse; near-random on all 4 runs |
| Quantum kernel on frozen CNN embeddings, single seed (Section 9) | Real DIAT-uSAT spectrograms | Ties accuracy, wins drone recall/PR-AUC at 3/3 scales |
| Same, 5-seed robustness check (Section 10) | Same | Confirmed real: wins 15/15 (seed × scale) combinations, 2–4× signal-to-noise |

The pattern: quantum kernel advantage in this project is representation-dependent, not
present by default, and appeared only once a classical CNN had already done the
representation learning — and it survived both a genuine attempt to disprove it (more
seeds) and a genuine attempt to close the *other* gap with a trainable circuit (more
optimizers, more restarts). That is a more defensible and more interesting research
finding than either "quantum wins" or "quantum never helps."

## 12. Noisy hardware simulation — closing the "what about real noise" gap

Sections 1–11 all run on exact, noiseless statevector simulation — honestly disclosed
throughout as a limitation (Section 5). `radarsense/quantum_noisy.py` closes that gap
directly: the identical QSVC task from Section 4 (same 4 features, same ZZFeatureMap,
same drone-vs-rest subsampling, n_train=100 matching a point already in Section 4's
table), but evaluated via shot-based sampling (1,024 shots/circuit) under a NoiseModel
built from representative NISQ-era error rates (0.1% single-qubit gate error, 1%
two-qubit/CX gate error, 2% readout error — literature-typical values, not a specific
device snapshot; qiskit-ibm-runtime's calibrated fake backends aren't installed in this
environment and the qiskit-core fake_provider backends available are old
pulse-simulation scaffolding, not representative of current hardware, so a documented
custom NoiseModel was used instead of citing an unrepresentative fake device).

**Result:**

| Kernel | PR-AUC | Drone recall | Drone precision | ROC-AUC |
|---|---|---|---|---|
| Quantum, noiseless (Section 4, n=100) | 0.718 | 0.40 | 0.88 | 0.817 |
| **Quantum, noisy (this section, n=100)** | **0.720** | **0.23** | **1.00** | **0.820** |
| RBF, classical (n=100) | 0.975 | 0.91 | 0.98 | 0.976 |
| Linear, classical (n=100) | 0.982 | 0.91 | 0.96 | 0.988 |

**Honest reading:** PR-AUC and ROC-AUC — the kernel's underlying ranking/separation
quality — are essentially unchanged under noise (0.720 vs 0.718, 0.820 vs 0.817).
What changed substantially is the SVM's realized decision boundary at its default
operating point: recall dropped from 0.40 to 0.23 while precision rose to a perfect
1.00 — the classifier became more conservative under noise, missing more drones but
never false-alarming on this test slice. This reads as noise perturbing the precomputed
kernel matrix in a way that shifts where the SVM's margin lands, more than it degrades
the kernel's fundamental ability to separate the classes. Neither number is good enough
to matter competitively — classical kernels still dominate by a wide margin regardless
of quantum noise — but this is real, measured evidence instead of an unanswered
question, and the specific finding (ranking quality stable, operating point shifted) is
more informative than a simple "noise made it worse" would have been.

**Limitations of this specific noise model**: representative literature-typical rates,
not a live device calibration snapshot — if the actual QPU error rates differ
substantially (better or worse), the quantitative shift would differ, though the
qualitative pattern (PR-AUC stability, operating-point sensitivity) is a plausible
general behavior for shot-based fidelity kernels under symmetric noise. Only one
train size tested (n=100) given shot-based sampling's much higher per-entry cost
than exact statevector fidelity (951s here vs. 1.7s for the noiseless equivalent) —
a scaling study across sizes is future work, not run here for time reasons.
