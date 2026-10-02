"""FastAPI wrapper around the 31-feature model's ONNX backend -- a third,
independent service alongside api/server.py (classical, 11-feature) and
api/quantum_server.py (QSVC, 4-feature).

Run:  uvicorn api.feat31_server:app --port 8082

Endpoints:
    GET  /health   liveness + model metadata (feature order, threshold, SNR range)
    POST /predict  single sample (31 raw statistical features) -> label + probabilities
    POST /batch    list of samples

Input contract: 31 pre-extracted statistical features (see
radarsense/feat31_features.py) -- NOT raw waveforms. A caller with raw
100-sample amplitude/velocity/energy waveforms must run
feat31_features.extract_features() first; that step isn't done inside this
server since it operates on batches of waveform arrays, not a single JSON
request body cleanly.
"""

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, create_model

from api.feat31_predict import predict_one, FEATURES, CLASSES, DRONE_T

app = FastAPI(title="RadarSense 31-Feature Model Inference API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("RADARSENSE_CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

Feat31Sample: type[BaseModel] = create_model(
    "Feat31Sample", **{f: (float, ...) for f in FEATURES})


@app.get("/health")
def health():
    return {"status": "ok", "model": "CatBoost (31-feature)", "classes": CLASSES,
            "feature_order": FEATURES, "drone_threshold": DRONE_T,
            "note": "31 pre-extracted statistical features required, not raw waveforms"}


@app.post("/predict")
def predict(sample: Feat31Sample):
    try:
        return predict_one(sample.model_dump())
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/batch")
def batch(samples: list[Feat31Sample]):
    return [predict_one(s.model_dump()) for s in samples]
