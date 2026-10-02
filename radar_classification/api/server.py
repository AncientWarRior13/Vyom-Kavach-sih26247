"""FastAPI wrapper around the ONNX model — production-shaped inference service.

Run:  uvicorn api.server:app --port 8080
Then: curl -X POST localhost:8080/predict -H 'content-type: application/json' -d @sample.json

Endpoints:
    GET  /health   liveness + model metadata
    POST /predict  single sample -> label + calibrated-ish probabilities
    POST /batch    list of samples

Extraction defenses (see EXTRACTION_DEFENSE.md for what was measured and
why -- response-hardening was tried first and found to do NOTHING against
the attack in MODEL_EXTRACTION_ATTACK.md, since it only ever needed the
hard label, not the probability):

  1. Query-pattern anomaly filter: an IsolationForest fit on the real
     training distribution (radarsense/extraction_defense.py), rejecting
     queries that don't resemble real physical detections. Catches 95.2%
     of the extraction attack's i.i.d.-uniform query sampling while
     wrongly rejecting only ~1% of real held-out detections.
  2. Rate limiting on total requests per client (slowapi) -- caps how
     many attempts (rejected or not) a client gets per window.

Honest limit, stated plainly: this raises the cost of extraction
substantially (~25x fewer usable queries at the same attempt budget), it
does not eliminate it -- the queries that DO pass the filter are already
informative enough for a workable surrogate. Not a complete fix; the best
defense measured so far, not a guarantee.
"""

import os

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from api.predict import predict_one, FEATURES, CLASSES, DRONE_T, META
from api.masking_rescue import apply_masking_rescue, RESCUE_GATE
from radarsense import config as C


def _predict_with_rescue(sample_dict: dict) -> dict:
    """Clean ONNX prediction, then the masking-rescue routing layer (keeps
    clean recall/precision; recovers micro-Doppler-masked drones)."""
    result = predict_one(sample_dict)
    r = apply_masking_rescue(sample_dict, result["label"])
    result["label"] = r["label"]
    result["masking_rescue"] = r["masking_rescue"]
    if "fallback_drone_prob" in r:
        result["fallback_drone_prob"] = r["fallback_drone_prob"]
    return result

app = FastAPI(title="RadarSense Inference API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("RADARSENSE_CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
RATE_LIMIT = "100/minute"

_anomaly_filter = joblib.load(C.MODELS_DIR / "extraction_anomaly_filter.joblib")
import json as _json
_anomaly_meta = _json.loads((C.MODELS_DIR / "extraction_anomaly_filter_meta.json").read_text())
_REJECT_THRESHOLD = _anomaly_meta["reject_threshold"]


def _looks_like_real_detection(sample: dict) -> bool:
    x = np.array([[float(sample[f]) for f in FEATURES]])
    score = _anomaly_filter.score_samples(x)[0]
    return score >= _REJECT_THRESHOLD


class RadarSample(BaseModel):
    doppler_velocity: float = Field(ge=-50, le=50)
    doppler_freq_shift: float
    rcs_estimate: float = Field(ge=-20, le=20)
    micro_doppler_bandwidth: float = Field(ge=0, le=500)
    micro_doppler_jitter: float = Field(ge=0, le=1)
    range_m: float = Field(ge=50, le=5000)
    azimuth_deg: float = Field(ge=-180, le=180)
    altitude_m: float = Field(ge=0, le=3000)
    velocity_variance: float = Field(ge=0)
    acceleration: float = Field(ge=-10, le=10)
    snr_db: float = Field(ge=-20, le=-10)


@app.get("/health")
def health():
    return {"status": "ok", "model": META["model_family"],
            "variant": META["variant"], "classes": CLASSES,
            "feature_order": FEATURES, "drone_threshold": DRONE_T,
            "masking_rescue": {"enabled": True, "gate": RESCUE_GATE,
                               "note": "clean ONNX decisions preserved; non-drone predictions with "
                                       "anomalously low micro-Doppler are escalated to drone only when "
                                       "the micro-Doppler-immune fallback is >= gate confident"}}


@app.post("/predict")
@limiter.limit(RATE_LIMIT)
def predict(request: Request, sample: RadarSample):
    sample_dict = sample.model_dump()
    if not _looks_like_real_detection(sample_dict):
        raise HTTPException(400, "sample does not resemble a real radar detection "
                                 "(failed the query-pattern anomaly check)")
    try:
        return _predict_with_rescue(sample_dict)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/batch")
@limiter.limit(RATE_LIMIT)
def batch(request: Request, samples: list[RadarSample]):
    out = []
    for s in samples:
        sample_dict = s.model_dump()
        if not _looks_like_real_detection(sample_dict):
            out.append({"error": "sample does not resemble a real radar detection "
                                 "(failed the query-pattern anomaly check)"})
            continue
        out.append(_predict_with_rescue(sample_dict))
    return out
