# 42-Feature Combined Model: Deployment

Production stack for the combined model reported in
[`FEAT42_COMBINED_EXPERIMENT.md`](FEAT42_COMBINED_EXPERIMENT.md), matching the existing
11-feature/31-feature backends' conventions exactly.

## What was built

- **ONNX export** (`radarsense/feat42_export_onnx.py`): CatBoost's native ONNX export,
  parity-checked against the original model. **100% label agreement, max probability
  deviation 3.2e-7** — the ONNX model is not an approximation. 0.038ms median single-sample
  latency, ~53,000 samples/sec batched throughput, CPU only (no GPU dependency), 3.4MB model
  file.
- **Inference client** (`api/feat42_predict.py`): same pattern as `api/feat31_predict.py`.
- **API server** (`api/feat42_server.py`): FastAPI, CORS-enabled, `/health` `/predict`
  `/batch` — a fourth independent backend alongside the classical (8080), quantum (8081), and
  31-feature (8082) services.
- **`start_all.sh`**: wired in as step 5/6, port **8083**.

## Live end-to-end verification (not just "the training script ran")

Started the server for real, sent actual HTTP requests, and checked the response against the
offline model directly — not inferred from unit-level correctness:

- A real -25dB drone from the held-out test set, submitted via `curl POST /predict`:
  correctly classified `drone` at p=0.998, threshold gate fired correctly. The live HTTP
  response and a direct call to `predict_one()` in-process matched **bit-for-bit**.
- `/batch` with one real sample of each class (aircraft/bird/anomaly/drone): all 4 correctly
  classified with high confidence (0.99+ each) via the live endpoint.
- `/health` confirms the correct 41-feature contract, class order, and deployed threshold
  (0.365) are being served.

This is the same live-request-against-a-running-process bar the incumbent 11-feature and
31-feature backends were already held to — the earlier `feat42_results.json` numbers were
real held-out metrics, but this is what actually proves the *serving path* works, which is a
different claim.

## Scope: backend only

Per explicit instruction, no frontend files were touched. The frontend's existing ONNX
inference plumbing (`frontend/app/components/OnnxInference.ts`) is not wired to this new
backend — that would require a frontend change, which was out of scope here.

## Branch

Built and pushed on a new standalone branch, `feat42`, branched from `feature31-standalone`
(which already includes the teammate's backend/frontend integration work, merged in first so
this builds on top of it rather than diverging from it).
