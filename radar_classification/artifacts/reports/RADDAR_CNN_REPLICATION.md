# RAD-DAR CNN + Quantum-Kernel Replication Attempt

*This is a replication attempt of the `diat_usat_cnn.py` finding (quantum fidelity kernel
beats classical RBF/linear kernels on frozen CNN bottleneck embeddings, specifically on
drone recall) on a second, independent real sensor. All quantum results below use noiseless
exact statevector simulation — same as every other quantum-kernel experiment in this project
except [`IBM_HARDWARE_EXPERIMENT.md`](IBM_HARDWARE_EXPERIMENT.md).*

## Why a second sensor

One real-sensor result showing "quantum kernel wins on CNN embeddings" is a single data
point — it could be an artifact of DIAT-uSAT's specific physics (X-band CW spectrograms) or
its specific class structure (4-way: aircraft/bird/drone/anomaly). RAD-DAR is genuinely
different on both axes: 8.75GHz FMCW instead of X-band CW, raw 11×61 range-Doppler
**matrices** instead of rendered spectrogram **images**, and a binary task (drone vs.
anomaly — RAD-DAR's own class map has no separate bird/aircraft classes). If the same
pattern holds here too, that's real evidence of generality. If it doesn't, that's an equally
legitimate finding — arguably the more important one, since it bounds how far the DIAT-uSAT
result can be trusted to generalize.

## Setup

- **Data**: 17,485 real RAD-DAR range-Doppler matrices (12,420 anomaly / 5,065 drone),
  session-aware `GroupShuffleSplit` (no session spans train/val/test — same discipline as
  `raddar_experiments.py`) → train 11,886 / val 2,543 / test 3,056, across 56/12/12 sessions.
- **CNN**: small Conv2D(16)→SE→pool→Conv2D(32)→SE→pool→GAP→Dense(64)→bottleneck(8)→Dense(2),
  class-weighted, 30 epochs, manual `train_on_batch`/`test_on_batch` loop (see implementation
  note below).
- **Quantum kernel**: `ZZFeatureMap(feature_dimension=8, reps=2)` on the frozen 8-dim
  bottleneck embeddings — same circuit family as `diat_usat_cnn.py`, sized to this model's
  bottleneck width. Exact statevector fidelity kernel, `SVC(kernel="precomputed")`.
- **Comparison**: quantum kernel vs. classical RBF vs. classical linear, at train sizes
  n∈{200, 400, 800}, 5 seeds each, stratified subsampling, fixed 300-point test set per seed.

### Implementation note: `.predict()` hangs on this machine, worked around

`model.fit()` and `model.predict()` both hang indefinitely on this machine via TensorFlow's
`tf.function` graph-tracing path — confirmed by isolated timeout tests. Training was already
routed around this with a manual `train_on_batch`/`test_on_batch` loop under
`tf.config.run_functions_eagerly(True)`. `.predict()` hung too, for a related but distinct
reason: TF's own runtime warning confirms `run_functions_eagerly` does **not** apply to the
`tf.data` pipeline that `.predict()` builds internally. Fixed by replacing every `.predict()`
call with direct batched `model(X_batch, training=False)` calls, which stay on the eager path
already proven to work. Same underlying model, same weights, same math — this is purely a
call-path change to avoid a TF-internals deadlock on this machine, not a change to what's
being measured.

## Result

**The CNN itself is excellent on RAD-DAR**: 95.4% test accuracy, 97.8% drone recall, 0.986
PR-AUC — this is a real, strong result on its own, independent of anything quantum.

**The quantum-kernel-wins pattern does NOT replicate.** At every training size, the quantum
kernel's drone recall is at or below both classical kernels, and by a wide margin at the
smallest size:

| n_train | kernel | accuracy (mean±std, 5 seeds) | drone recall (mean±std) | PR-AUC (mean±std) |
|---|---|---|---|---|
| 200 | quantum_zz | 0.883 ± 0.050 | **0.797 ± 0.112** | 0.953 ± 0.022 |
| 200 | rbf | 0.953 ± 0.008 | **0.946 ± 0.012** | 0.962 ± 0.017 |
| 200 | linear | 0.950 ± 0.009 | 0.937 ± 0.011 | 0.984 ± 0.011 |
| 400 | quantum_zz | 0.931 ± 0.024 | **0.913 ± 0.050** | 0.959 ± 0.015 |
| 400 | rbf | 0.953 ± 0.006 | **0.949 ± 0.012** | 0.962 ± 0.017 |
| 400 | linear | 0.952 ± 0.005 | 0.937 ± 0.004 | 0.984 ± 0.011 |
| 800 | quantum_zz | 0.945 ± 0.004 | 0.939 ± 0.014 | 0.957 ± 0.014 |
| 800 | rbf | 0.951 ± 0.007 | 0.942 ± 0.006 | 0.962 ± 0.013 |
| 800 | linear | 0.951 ± 0.006 | 0.934 ± 0.006 | 0.983 ± 0.012 |

Paired quantum-vs-RBF drone recall, per-seed (5 seeds each):

| n_train | quantum wins (of 5 seeds) | mean(quantum − RBF) |
|---|---|---|
| 200 | 0 / 5 | **−0.149** |
| 400 | 2 / 5 | −0.036 |
| 800 | 2 / 5 | −0.003 |

At n=200, quantum loses every single seed, by 15 points of drone recall on average — a large,
consistent gap, not noise. The gap narrows as training size grows and is within one std of
zero by n=800, but never flips to a quantum advantage on this sensor at any size tested.

## Honest interpretation

The DIAT-uSAT finding does **not generalize** to RAD-DAR as tested here. Two sensors, same
circuit family (`ZZFeatureMap`, reps=2) applied to frozen CNN bottleneck embeddings, opposite
outcomes on which kernel wins. Plausible reasons, none confirmed:

- **Embedding dimensionality differs**: DIAT-uSAT's CNN bottleneck and RAD-DAR's are not
  necessarily encoding equally kernel-friendly geometries — an 8-qubit ZZFeatureMap embeds an
  8-dim vector into an exponentially larger Hilbert space, which can help or hurt depending on
  how the classical embedding is already structured for this specific decision boundary.
- **Task shape differs**: RAD-DAR is a comparatively "easy" binary problem (RBF and linear
  both clear 93%+ drone recall even at n=200) where classical kernels are already close to a
  ceiling, leaving little room for a quantum kernel to show an advantage even if one exists in
  principle. DIAT-uSAT's 4-way task with class imbalance may have had more headroom.
  This is a hypothesis, not a demonstrated cause.
- **Small-sample quantum kernels are noisier**: the quantum kernel's own std (0.11 at n=200)
  is far larger than RBF's (0.012) at the same size — whatever quantum is doing here, it's
  also less stable across seeds at small training sizes, independent of the mean gap.

**What this means for the project's overall claim**: the correct, honest framing is "a
quantum-kernel advantage on CNN embeddings was observed on one real sensor (DIAT-uSAT) and
did not replicate on a second (RAD-DAR)" — not "quantum kernels beat classical kernels on CNN
embeddings" as a general statement. This is a genuine boundary condition, found by actually
running the second experiment rather than assuming the first result would hold.

## Caveats

- One architecture, one circuit family, one bottleneck width (8) — not a sweep over encoding
  choices that might behave differently on RAD-DAR.
- 5 seeds per train size is enough to see a real, large effect at n=200 (0/5 wins, consistent
  −0.15 gap) but not enough to tightly bound the near-zero gap at n=800.
- Noiseless simulation throughout — see [`IBM_HARDWARE_EXPERIMENT.md`](IBM_HARDWARE_EXPERIMENT.md)
  for the one real-hardware measurement in this project (a different experiment, same feature
  map family, not this specific comparison).

## Artifacts

- `artifacts/reports/raddar_cnn_quantum_results.json` — full per-seed results
- `artifacts/plots/raddar_cnn_quantum_multiseed.png` — accuracy/recall vs. train size, both kernels
- `radarsense/raddar_cnn.py` — full script
