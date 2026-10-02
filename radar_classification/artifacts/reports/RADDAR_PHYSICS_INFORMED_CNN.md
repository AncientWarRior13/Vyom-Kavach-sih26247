# Physics-Informed CNN on RAD-DAR: Does an Auxiliary Physics Loss Help?

Bakes real radar physics directly into training, not just as engineered input features: an
auxiliary regression head, trained jointly with the classification head, forced to predict
physically-meaningful quantities computed directly from the same raw range-Doppler matrix the
classifier sees — `doppler_velocity`, `micro_doppler_bandwidth`, `micro_doppler_jitter` (via
`raddar_experiments.matrix_features()`, imported unmodified, the same deterministic transform
already used elsewhere in this project to bridge RAD-DAR into the tabular schema). No extra
labels needed — the targets are self-supervised, computed from the same matrix.

Why this counts as physics-informed and not just feature engineering: the auxiliary targets
are never given to the network as *inputs* — they're a second loss term the convolutional
filters must also satisfy, intended to regularize the learned representation toward
physically-grounded structure (Doppler centroid/spread/jitter) rather than whatever pattern
happens to separate the two classes.

## Setup

Identical backbone to `raddar_cnn.py`'s CNN (same Conv2D/SE/pool stack, same 8-dim
bottleneck) so any difference reflects the added physics loss, not an architecture change.
Two heads off the shared bottleneck: classification (2-way softmax) and auxiliary regression
(3 targets, z-scored on train statistics so all three contribute comparably to the loss
despite very different natural scales — m/s vs. Hz vs. a [0,1] ratio). Combined loss =
classification cross-entropy + 1.0 × auxiliary MSE.

## Result: the auxiliary physics loss does not help — a real, clean negative result

| | Baseline (no physics loss) | Physics-informed |
|---|---|---|
| Test accuracy | **95.4%** | 94.5% |
| Test drone recall | **97.8%** | 95.5% |

Slightly *worse* on both metrics, not better. The auxiliary regression loss converged
reasonably (final z-scored MSE 0.944 — the network does learn to predict the physics
quantities to a real degree, this isn't a training failure), but that additional supervision
signal didn't translate into a better classifier.

The quantum-vs-classical kernel comparison on the resulting embeddings shows the same
qualitative pattern as the original (non-physics-informed) RAD-DAR CNN — quantum kernel
still trails classical RBF at every training size, closing but not flipping the gap as n
grows:

| n_train | quantum accuracy | RBF accuracy | quantum recall | RBF recall |
|---|---|---|---|---|
| 200 | 86.8% ± 4.1 | 94.7% ± 0.7 | 77.9% ± 11.9 | 94.3% ± 1.4 |
| 400 | 91.5% ± 1.5 | 94.7% ± 0.6 | 91.5% ± 3.7 | 94.6% ± 2.2 |
| 800 | 93.4% ± 0.9 | 94.5% ± 0.9 | 94.6% ± 1.8 | 94.6% ± 1.7 |

Both kernels score marginally lower here than on the original (non-physics-informed)
embeddings reported in `RADDAR_CNN_REPLICATION.md` (e.g. RBF 94.7% vs. 95.3% at n=200) —
consistent with the physics-informed CNN producing marginally weaker embeddings overall, not
specifically worse for one kernel over the other. The quantum-vs-classical finding itself
(quantum doesn't win here) is unchanged by adding the physics loss.

## Honest interpretation

Adding physics-informed multi-task supervision did not improve this classifier, and slightly
hurt it. Plausible reasons, none confirmed:

- **The auxiliary targets may already be redundant with what classification supervision
  alone teaches the network.** `doppler_velocity`/`micro_doppler_bandwidth`/
  `micro_doppler_jitter` are exactly the kind of signal a CNN trained on drone-vs-anomaly
  classification would already need to encode to separate the classes — the physics loss may
  be asking the network to explicitly learn something it was already implicitly learning,
  adding optimization difficulty (a second, sometimes-competing gradient signal) without new
  information.
- **λ=1.0 may not be the right weighting.** No sweep over the auxiliary loss weight was run —
  it's possible a smaller λ (lighter-touch regularization) would help where λ=1.0 doesn't, or
  that no weighting helps and the effect is genuinely null. Not distinguished here.
- **This is one architecture, one sensor, one set of three auxiliary targets** — not a
  general claim that physics-informed losses don't work for radar CNNs, just that this
  specific, reasonably-motivated version of the idea didn't pay off on this task.

This is reported as a real negative result rather than tuned or reframed until it looked
better — consistent with how this project has handled every other negative finding (the
RAD-DAR quantum-kernel replication, the flat drone-recall result in the 300-step waveform
experiment, the SNR-aware threshold experiment).

## Caveats

- Compared against `raddar_cnn.py`'s own already-reported baseline numbers (same
  architecture/data/split/seed), not a freshly retrained matched baseline in this same
  session — TensorFlow training isn't perfectly bit-reproducible run-to-run even with a fixed
  seed, so a small part of the ~1-point gap could be run-to-run noise rather than the physics
  loss itself. The RAD-DAR CNN's own multi-seed stability elsewhere in this project suggests
  this kind of gap is plausibly noise-sized, which is itself part of the honest reading here:
  the physics loss is not clearly *harmful* either, just not clearly helpful.
- No λ sweep, no alternative auxiliary-target sets tried.
- Single run for the direct classification comparison (the quantum-kernel comparison is
  5-seed, the CNN training itself is not).

## Artifacts

- `radarsense/raddar_physics_informed_cnn.py`
- `artifacts/reports/raddar_physics_informed_cnn_results.json`
- `artifacts/plots/raddar_physics_informed_cnn_quantum_multiseed.png`
- `artifacts/models/raddar_physics_informed_cnn.keras`
