# Classical (ONNX) vs. Quantum (QSVC) Backend Comparison

*All numbers below are pulled directly from `artifacts/reports/onnx_benchmark.json`,
`baseline_results.json`, `quantum_results.json`, and `latency_snr_sweep_results.json`
— nothing here is estimated or illustrative.*

## Read this before the table: the two backends are not the same benchmark

This is not an apples-to-apples comparison and presenting it as one would be
misleading. The two models differ in task scope, not just implementation:

| | Classical (ONNX) | Quantum (QSVC) |
|---|---|---|
| Task | Full 4-class (aircraft/bird/drone/anomaly) | Binary (drone vs. not-drone) |
| Features | All 11 (frozen schema) | 4 physics features only |
| Evaluation set | Full held-out test, n=7,800 | Subsampled pool, n_train=800 / n_test=300 |

Any table that puts a single "accuracy" number from each side by side is
implicitly comparing different tasks on different data. The table below is
scoped honestly instead: each model's own real, already-published number, on
its own real evaluation protocol.

## Accuracy / recall (each on its own protocol)

| Metric | Classical (4-class, full test) | QSVC (binary, n_train=800) |
|---|---|---|
| Drone recall | 91.8% | 71.1% |
| Drone precision | 77.5% | 84.2% |
| PR-AUC (drone) | 0.938 | 0.882 |
| ROC-AUC | — (not the primary metric used) | 0.934 |

QSVC's own reported result already loses to classical kernels at every
tested scale on this exact task (`QUANTUM_MINIPAPER.md`) — this is not new
information, and this comparison doesn't change that conclusion. Classical
RBF/linear kernels on the same n_train=800 subsample reach 0.976–0.982
PR-AUC, well above both the classical deployed model's 0.938 (different task
scope, full 4-class) and QSVC's 0.882.

## Latency (real measurements, both backends, this session)

| | Classical (ONNX) | Quantum (QSVC) |
|---|---|---|
| Median single-call latency | ~0.03–0.06 ms | ~8–17 ms |
| Correlation with input SNR | r = 0.051 (none) | r = -0.030 (none) |
| Startup cost | none (model loads pre-trained) | ~3–9s (trains QSVC once at process start) |

Both are measured across the full test set's SNR range (`latency_snr_sweep.py`,
`artifacts/reports/latency_snr_sweep_results.json`) — neither backend's
latency depends on how noisy the input is, which is the expected, correct
behavior for a fixed-depth model (tree ensemble or fixed quantum circuit):
**noise degrades accuracy, not inference speed**, for either backend.
Quantum is ~150–500x slower per call than classical — a real, direct
consequence of simulating a live quantum circuit per request instead of a
tensor forward pass.

## Model size / persistence

| | Classical (ONNX) | Quantum (QSVC) |
|---|---|---|
| Model artifact | `radarsense.onnx`, 2,734.5 KiB (~2.7 MB) | none — no model file exists; kernel + SVC state is computed fresh at every process startup (~3–9s) |
| Reproducibility | Deterministic (fixed weights) | Deterministic given fixed seed + subsample, but requires re-running the training step, not just loading a file |

## Deployment surface

| | Classical (ONNX) | Quantum (QSVC) |
|---|---|---|
| Browser / client-side | Yes (`onnxruntime-web`, verified in `frontend/`) | No — needs a live Qiskit runtime, cannot run client-side |
| Server / API | Yes (`api/server.py`, port 8080) | Yes (`api/quantum_server.py`, port 8081) |
| Edge / mobile | Yes (2.7MB, 0.03ms — trivially edge-deployable) | No — server/workstation-class CPU only in practice |

## Explainability

| | Classical (ONNX) | Quantum (QSVC) |
|---|---|---|
| Method | SHAP (summary + per-detection waterfall, `explain.py`) | Kernel-target alignment (`quantum.py`'s alignment metric — a global geometry diagnostic, not a per-prediction explanation) |
| Per-prediction explanation | Yes | No — QSVC has no per-sample feature-attribution method implemented in this project |

## Honest summary

> The classical ONNX model is the deployed, production-facing system:
> full 4-class, sub-millisecond, edge-deployable, per-prediction explainable.
> The QSVC backend is a real, independently-running quantum inference
> service — not a toggle or a simulation of one — but it is narrower in
> scope (binary, 4 features), ~150–500x slower per call, cannot run
> client-side, and its own accuracy already loses to classical kernels on
> the same task. It exists to demonstrate genuine quantum inference and to
> support the project's honest quantum ablation story, not to outperform
> the deployed model. Both backends' latency is confirmed independent of
> input SNR (Pearson r ≈ 0 for both) — noise affects what each model
> predicts, not how fast it predicts.
