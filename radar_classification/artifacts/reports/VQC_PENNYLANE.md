# PennyLane Differentiable VQC — the Second SDK, and It Fixes the Qiskit VQC

The Quant-A-thon primer teaches **two** SDKs — Qiskit (circuit/hardware-first) and PennyLane
(differentiable / ML-native) — and points to PennyLane for quantum ML because its circuits are
functions you can **differentiate** (slide 12: "gradient descent, on a qubit"). The project was
100% Qiskit, and its Qiskit VQC (trained with the gradient-*free* optimizers COBYLA/SPSA)
underperformed badly. This adds the PennyLane half and trains the same kind of ansatz with true
**gradient descent** — which turns the failed VQC into a competent one.

## Result: gradient descent rescues the VQC

Same setup as every other quantum experiment here (the 4 `QUANTUM_FEATURES`, drone-vs-not,
stratified subsample, 3 seeds). Circuit: `AngleEmbedding` → `StronglyEntanglingLayers` →
`expval(PauliZ(0))`, a QNode differentiated end-to-end with Adam.

| VQC | Optimizer | ROC-AUC | PR-AUC |
|---|---|---|---|
| Qiskit `vqc.py` | COBYLA (gradient-free) | ~0.50 | 0.483 |
| Qiskit `vqc_v2.py` | SPSA (gradient-free) | 0.629 | 0.394 |
| **PennyLane (this)** | **Adam (gradient descent)** | **0.979 ± 0.004** | **0.924 ± 0.022** |
| Classical LogisticRegression | — | 0.979 | 0.939 |

**The VQC went from near-random (0.50–0.63 ROC-AUC) to 0.979 — competitive with classical —
purely by switching to gradient-based training.** This is a direct, measured confirmation of the
primer's slide-12 thesis: the earlier VQC failure was an **optimizer problem** (gradient-free
COBYLA/SPSA get stuck), not a fundamental one. PennyLane differentiates through the circuit, so
Adam can actually descend the loss.

## Honest interpretation (consistent with the project's overall finding)

- **This does NOT overturn the no-quantum-advantage conclusion.** The classical baseline still
  edges the VQC (PR-AUC 0.939 vs 0.924); the VQC *matches* classical, it does not beat it. So the
  project's central, well-established finding stands — quantum provides no advantage on this data.
- **What changed is honesty of the VQC claim.** Before, the VQC was a *broken* model (near
  random) — an easy target for "your VQC doesn't work." Now it's a *working* model trained the way
  the field (and the hackathon primer) actually recommends. "Our VQC reaches 0.979 ROC-AUC,
  competitive with classical, once trained with gradient descent" is a far stronger, and true,
  position than the earlier failure.
- **Both SDKs are now used.** Qiskit for the kernels + real hardware + error mitigation;
  PennyLane for the differentiable VQC — exactly the primer's "pick by goal," using each for what
  it's best at.

## Device-agnostic (primer slide 14), verified

The same QNode runs unchanged on different backends. Confirmed: identical output on
`default.qubit` and `lightning.qubit` (max difference **0.0**). Swapping to `qiskit.aer` or a real
IBM backend is the same one-line device change (via `pennylane-qiskit`) — the primer's "prototype
and train in PennyLane, then deploy the same circuit on Qiskit/IBM" workflow.

## Caveats

- 4 qubits, statevector-exact (noiseless) — same NISQ-scale caveat as all quantum work here; a
  4-qubit circuit is trivially classically simulable, so no advantage is possible in principle.
- ROC-AUC is used for the cross-method comparison (prevalence-robust); PR-AUC reported alongside.
- 3 seeds; the ordering (PennyLane-Adam ≫ Qiskit-COBYLA/SPSA, and PennyLane ≈ classical) is the
  robust finding.

## Artifacts

- `radarsense/vqc_pennylane.py`
- `artifacts/reports/vqc_pennylane_results.json`
