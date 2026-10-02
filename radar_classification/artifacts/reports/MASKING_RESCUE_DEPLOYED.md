# Deployed: Masking-Rescue Routing (clean recall/precision preserved)

The finals-hardening masking defense is now **served** — not by swapping the ONNX model (which
would move clean recall/precision), but as a routing layer on top of the unchanged clean ONNX.
This was a deliberate choice after measuring the alternative.

## Why routing, not a hardened model

Retraining a single model on masking+evasion augmentation and exporting it (measured, then
reverted) hardened masking but moved clean metrics materially:

| | Clean acc | Clean recall | Clean precision | Masking ×0.3 |
|---|---|---|---|---|
| Combined hardened model | 96.7% | 85.1% | 60.6% | 82.1% |
| **Deployed: clean ONNX + routing** | **97.6% (=baseline)** | **preserved** | **preserved** | 32.8% |

Retraining shifts the decision boundary, so clean recall/precision necessarily change. Routing
keeps the clean model's decision everywhere and only *escalates* on a suspected-masking sample,
so clean performance is preserved.

## What's deployed (`api/masking_rescue.py`, wired into `api/server.py`)

For each request: the clean ONNX model predicts as before. A non-drone prediction is escalated to
drone **only if** `micro_doppler_bandwidth` is below the drone-class cutoff (166.8 Hz — a masking
signature) **and** the kinematic fallback model — which never sees micro-Doppler, so masking
cannot fool it — is ≥ **0.98** confident the target is a drone. Gate is tunable (`RESCUE_GATE`).

## Verified end-to-end (2,500-sample clean set + masked drone set)

| Metric | Baseline | Deployed (routed) |
|---|---|---|
| Clean drone recall | 0.9322 | **0.9322** (identical) |
| Clean drone precision | 0.7237 | **0.7237** (identical) |
| Masked ×0.3 drone recall | 12.3% | **32.8%** (2.7×) |

Clean recall and precision are **bit-identical** (no clean sample tripped the conservative gate),
and severe-masking recall improves 2.7×. Live-server checks confirmed: a clean drone predicts
drone (no rescue); the API `/health` advertises the rescue layer and gate.

## Honest limits

- **Gate 0.98 is conservative by design** — it recovers only the ~33% of masked drones whose
  *kinematics* clearly indicate a drone (the fallback is near-certain). The other ~67% have
  ambiguous kinematics and are not rescued. Recovering more requires lowering the gate, which
  spends clean precision (dial documented in `MASKING_DEFENSE_COMPLETE.md`: gate 0.90 → ~4×
  masking at −4pt precision; gate 0.50 → 6.3× at −37pt precision).
- This defends **micro-Doppler masking** only, not black-box evasion or extraction (separate
  layers). It is not a certified guarantee.

## Artifacts

- `api/masking_rescue.py`, `api/server.py` (endpoints `/predict`, `/batch` apply the rescue)
- `artifacts/models/masking_defense_fallback.joblib`, `masking_defense_meta.json`
