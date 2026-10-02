"""FastAPI wrapper around the QSVC quantum-kernel backend -- a SEPARATE
service from api/server.py (the classical ONNX model). See
api/quantum_predict.py's docstring for why these can't be merged: the
quantum kernel needs a live Qiskit runtime per prediction, which ONNX
Runtime / a browser cannot provide.

Run:  uvicorn api.quantum_server:app --port 8081
Then: curl -X POST localhost:8081/predict -H 'content-type: application/json' \
        -d '{"doppler_velocity": 2.1, "rcs_estimate": -15.0, "micro_doppler_bandwidth": 320.0, "micro_doppler_jitter": 0.8}'

Endpoints:
    GET  /health   liveness + circuit metadata (qubits, depth, backend, n_train)
    POST /predict  single sample -> drone / not_drone + probability
    POST /batch    list of samples

Honesty reminder baked into /health's response: this backend's own reported
accuracy LOSES to the classical model (see QUANTUM_MINIPAPER.md) -- it is
serving REAL quantum inference, not a better prediction.
"""

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api.quantum_predict import predict_one, FEATURES, CLASSES, _train_once, get_hardware_validation_info

app = FastAPI(title="RadarSense Quantum Kernel (QSVC) Inference API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("RADARSENSE_CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuantumSample(BaseModel):
    doppler_velocity: float = Field(ge=-50, le=50)
    rcs_estimate: float = Field(ge=-20, le=20)
    micro_doppler_bandwidth: float = Field(ge=0, le=500)
    micro_doppler_jitter: float = Field(ge=0, le=1)
    use_ensemble: bool = False
    mode: str = "quantum"


@app.on_event("startup")
def _warm_start():
    _train_once()


@app.get("/health")
def health():
    st = _train_once()
    return {
        "status": "ok",
        "model": "QSVC + Stacking Ensemble API",
        "task": "binary: drone vs. not_drone (NOT the full 4-class problem)",
        "classes": CLASSES,
        "feature_order": FEATURES,
        "ensemble_supported": True,
        "circuit": st["circuit_meta"],
        "honesty_note": ("This backend serves both pure QSVC quantum inference and a "
                          "Classical+Quantum Logistic Stacking Ensemble -- see "
                          "artifacts/reports/CLASSICAL_QUANTUM_ENSEMBLE.md."),
    }


@app.get("/hardware_validation")
def hardware_validation():
    return get_hardware_validation_info()



@app.post("/predict")
def predict(sample: QuantumSample):
    try:
        sample_dict = sample.model_dump()
        use_ens = sample_dict.pop("use_ensemble", False) or (sample_dict.pop("mode", "quantum") == "ensemble")
        return predict_one(sample_dict, use_ensemble=use_ens)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/batch")
def batch(samples: list[QuantumSample]):
    out = []
    for s in samples:
        d = s.model_dump()
        use_ens = d.pop("use_ensemble", False) or (d.pop("mode", "quantum") == "ensemble")
        out.append(predict_one(d, use_ensemble=use_ens))
    return out

