"""FastAPI wrapper around the 42-feature combined model's ONNX backend --
a fourth, independent service alongside api/server.py (classical,
11-feature), api/quantum_server.py (QSVC, 4-feature), and
api/feat31_server.py (31-feature).

Run:  uvicorn api.feat42_server:app --port 8083

Endpoints:
    GET  /health   liveness + model metadata (feature order, threshold, SNR range)
    POST /predict  single sample (41 combined features) -> label + probabilities
    POST /batch    list of samples

Input contract: 41 pre-computed features -- the 11 physics features AND
the 31 waveform-statistic features for the SAME detection (see
radarsense/feat42_simulator.py). Not raw waveforms; same documented
limitation as api/feat31_server.py.
"""

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, create_model

from api.feat42_predict import predict_one, FEATURES, CLASSES, DRONE_T

app = FastAPI(title="RadarSense 42-Feature Combined Model Inference API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("RADARSENSE_CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

Feat42Sample: type[BaseModel] = create_model(
    "Feat42Sample", **{f: (float, ...) for f in FEATURES})


@app.get("/health")
def health():
    return {"status": "ok", "model": "CatBoost (42-feature combined: 11 physics + 31 waveform-stat)",
            "classes": CLASSES, "feature_order": FEATURES, "drone_threshold": DRONE_T,
            "note": ("41 pre-computed features required (both feature sets for the same "
                     "detection), not raw waveforms")}


@app.post("/predict")
def predict(sample: Feat42Sample):
    try:
        return predict_one(sample.model_dump())
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/batch")
def batch(samples: list[Feat42Sample]):
    return [predict_one(s.model_dump()) for s in samples]
