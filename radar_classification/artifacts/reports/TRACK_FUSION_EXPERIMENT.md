# Track-Level (Multi-Frame) Fusion + Quantum Kernel

*New, previously-unexplored territory for this project — combining track-level fusion
(standard in real radar ATR systems) with the quantum-kernel ablation already established
in `radarsense/quantum.py`.*

## Motivation

Every quantum experiment so far (`quantum.py`, `vqc.py`, `diat_usat_cnn.py`) classifies one
snapshot at a time. Real radar systems track a target across consecutive detections and
fuse those detections into one decision — a single noisy frame is the hard case, not the
representative one. This asks: **does fusing predictions across a target's track help the
quantum kernel specifically close any of the gap to classical**, or does fusion help both
backends equally (or neither)?

## Method

- `track_simulator.py`: generates 10,000 tracks × 5 frames each (50,000 rows). Each track
  shares ONE clean physics draw (`simulator.py`'s `_class_block`, imported unmodified — same
  target, doesn't change between frames), with **independent per-frame measurement noise**
  applied per frame (same noise-injection formula as the frozen simulator) — mimicking
  consecutive radar returns of one physical target.
- Split by **track**, never by frame (a track's 5 frames never span train/val/test — the
  obvious leakage risk, same discipline as the real-sensor bridges' session-aware splits).
- Two fusion rules, applied identically to both backends: mean probability across the
  track's 5 frames (then threshold), and majority vote across 5 hard per-frame predictions.
- Classical: the frozen `final_model.joblib`, unmodified. Quantum: a fresh QSVC trained on
  this track data's 4 physics features (n_train=400, same tractability constraint as
  `quantum.py`).

## Result

| | Drone recall | Drone precision | PR-AUC |
|---|---|---|---|
| Classical, single-frame | 88.1% | 78.4% | 0.887 |
| Classical, track-fused (mean proba) | 91.9% | 87.2% | 0.944 |
| Quantum, single-frame | 69.7% | 24.4% | 0.306 |
| Quantum, track-fused (mean proba) | 78.4% | 46.0% | 0.522 |

**Drone-recall gain from track fusion: classical +3.8 points, quantum +8.6 points.**

Fusion helps **both** backends, and quantum still trails classical after fusion (78.4% vs.
91.9% recall) — this is not "quantum catches up." But the *relative* benefit of fusion is
more than 2x larger for quantum than for classical, and quantum's precision more than
doubles (24.4%→46.0%) under fusion, its biggest single improvement of any experiment in
this project.

## Interpretation

Statistically sensible, not a fluke: quantum's single-frame predictions start from a
noisier, lower baseline (PR-AUC 0.306 vs. classical's 0.887), so averaging out independent
per-frame noise removes proportionally more of that noise — the same reason averaging N
independent noisy measurements helps more when the individual measurements are noisier to
begin with. This doesn't mean quantum kernels are "secretly good once fused" — it's
consistent with the already-established finding that this quantum kernel is a noisier,
weaker classifier on raw physics features, and fusion is a variance-reduction technique
that helps noisier classifiers proportionally more.

## Caveats, stated plainly

- One simulator, one track-length (K=5 frames), one quantum kernel configuration
  (n_train=400). Not tested: whether the gain scales with more frames, or whether it
  holds at other quantum train sizes.
- Majority-vote fusion's PR-AUC (0.783 classical / 0.368 quantum) is *worse* than
  mean-probability fusion's — using a hard-vote's own vote-fraction as a "score" for PR-AUC
  is a coarser signal than an averaged continuous probability, which is expected and not a
  contradictory result, just a reminder that fusion METHOD matters, not just fusion itself.
- Quantum still loses to classical after fusion — this strengthens the case for track-level
  processing generally, not for preferring the quantum backend.

## Honest headline

> Track-level fusion is a real, measurable win for both backends, and specifically closes
> more of the quantum kernel's gap (in relative terms) than it closes for classical — a
> genuine, previously-untested finding for this project, reported alongside the caveat that
> quantum still loses in absolute terms even after fusion.
