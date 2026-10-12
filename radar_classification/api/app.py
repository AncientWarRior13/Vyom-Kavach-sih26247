"""RadarSense Unified FastAPI Server.
Single-entry application providing endpoints for all models:
  1. Classic 11-Feature Model (/predict/classic)
  2. 31-Feature Model (/predict/31)
  3. 41/42-Feature Combined Model (/predict/42)
  4. Quantum (QSVC / Ensemble) Model (/predict/quantum, /predict)

Run:
  uvicorn api.app:app --port 8081 --reload
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, create_model

from api.predict import (
    predict_one as predict_classic,
    FEATURES as FEATURES_CLASSIC,
    CLASSES as CLASSES_CLASSIC,
    DRONE_T as DRONE_T_CLASSIC
)
from api.quantum_predict import (
    predict_one as predict_quantum,
    FEATURES as FEATURES_QUANTUM,
    CLASSES as CLASSES_QUANTUM,
    _train_once,
    get_hardware_validation_info
)
from api.feat31_predict import (
    predict_one as predict_31,
    FEATURES as FEATURES_31,
    CLASSES as CLASSES_31,
    DRONE_T as DRONE_T_31
)
from api.feat42_predict import (
    predict_one as predict_42,
    FEATURES as FEATURES_42,
    CLASSES as CLASSES_42,
    DRONE_T as DRONE_T_42
)
from api.masking_rescue import apply_masking_rescue, RESCUE_GATE
from api.fleet import router as fleet_router
from api.explain import router as explain_router
from api.alerting import router as alert_router

app = FastAPI(
    title="DroneThreatTrainer Radar Analytics API",
    version="2.0",
    description="Unified radar classification API for DroneThreatTrainer: Classic, Quantum, 31-Feature, and 42-Feature models."
)

cors_origins = os.getenv("RADARSENSE_CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if "*" in cors_origins else cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(fleet_router)
app.include_router(fleet_router, prefix="/api")
app.include_router(explain_router)
app.include_router(alert_router)

class QuantumSample(BaseModel):
    doppler_velocity: float = Field(ge=-50, le=50)
    rcs_estimate: float = Field(ge=-20, le=20)
    micro_doppler_bandwidth: float = Field(ge=0, le=500)
    micro_doppler_jitter: float = Field(ge=0, le=1)
    use_ensemble: bool = False
    mode: str = "quantum"

ClassicSample: type[BaseModel] = create_model(
    "ClassicSample", **{f: (float, ...) for f in FEATURES_CLASSIC}
)

Feat31Sample: type[BaseModel] = create_model(
    "Feat31Sample", **{f: (float, ...) for f in FEATURES_31}
)

Feat42Sample: type[BaseModel] = create_model(
    "Feat42Sample", **{f: (float, ...) for f in FEATURES_42}
)

@app.on_event("startup")
def startup_event():
    import threading
    def _bg_train():
        try:
            print("Quantum model warm-up started in background...")
            _train_once()
            print("Quantum model warm-up complete.")
        except Exception as e:
            print(f"Warning: Quantum model warm start failed: {e}")
    threading.Thread(target=_bg_train, daemon=True).start()

@app.get("/")
@app.get("/health")
def root_health():
    try:
        st = _train_once()
        q_meta = st["circuit_meta"]
    except Exception as e:
        q_meta = f"Initializing or unavailable: {e}"
    return {
        "status": "ok",
        "service": "DroneThreatTrainer Radar Analytics API",
        "endpoints": {
            "quantum": "/predict/quantum (or /predict)",
            "classic": "/predict/classic",
            "feat31": "/predict/31",
            "feat42": "/predict/42",
            "hardware_validation": "/api/v1/quantum/hardware_validation",
        },
        "quantum_circuit": q_meta,
    }

@app.get("/health/quantum")
@app.get("/api/v1/quantum/health")
def quantum_health():
    st = _train_once()
    return {
        "status": "ok",
        "model": "QSVC + Stacking Ensemble API",
        "classes": CLASSES_QUANTUM,
        "feature_order": FEATURES_QUANTUM,
        "circuit": st["circuit_meta"]
    }

@app.get("/health/quantum/hardware")
@app.get("/api/v1/quantum/hardware_validation")
def quantum_hardware_endpoint():
    return get_hardware_validation_info()

@app.post("/predict/quantum")
@app.post("/predict")
@app.post("/api/v1/quantum/predict")
def predict_quantum_endpoint(sample: QuantumSample):
    try:
        sample_dict = sample.model_dump()
        use_ens = sample_dict.pop("use_ensemble", False) or (sample_dict.pop("mode", "quantum") == "ensemble")
        return predict_quantum(sample_dict, use_ensemble=use_ens)
    except Exception as e:
        raise HTTPException(500, f"Quantum inference error: {str(e)}")


@app.get("/health/classic")
@app.get("/api/v1/classic/health")
def classic_health():
    return {
        "status": "ok",
        "model": "Classic 11-Feature ONNX Model",
        "classes": CLASSES_CLASSIC,
        "feature_order": FEATURES_CLASSIC,
        "drone_threshold": DRONE_T_CLASSIC,
        "masking_rescue": {"enabled": True, "gate": RESCUE_GATE}
    }

@app.post("/predict/classic")
@app.post("/api/v1/classic/predict")
def predict_classic_endpoint(sample: ClassicSample):
    try:
        d = sample.model_dump()
        result = predict_classic(d)
        r = apply_masking_rescue(d, result["label"])
        result["label"] = r["label"]
        result["masking_rescue"] = r["masking_rescue"]
        if "fallback_drone_prob" in r:
            result["fallback_drone_prob"] = r["fallback_drone_prob"]
        return result
    except Exception as e:
        raise HTTPException(500, f"Classic model inference error: {str(e)}")

@app.get("/health/31")
@app.get("/api/v1/feat31/health")
def feat31_health():
    return {
        "status": "ok",
        "model": "CatBoost 31-Feature Model",
        "classes": CLASSES_31,
        "feature_order": FEATURES_31,
        "drone_threshold": DRONE_T_31
    }

@app.post("/predict/31")
@app.post("/api/v1/feat31/predict")
def predict_feat31_endpoint(sample: Feat31Sample):
    try:
        return predict_31(sample.model_dump())
    except Exception as e:
        raise HTTPException(500, f"31-Feature model inference error: {str(e)}")

@app.get("/health/42")
@app.get("/api/v1/feat42/health")
def feat42_health():
    return {
        "status": "ok",
        "model": "CatBoost 42-Feature Combined Model",
        "classes": CLASSES_42,
        "feature_order": FEATURES_42,
        "drone_threshold": DRONE_T_42
    }

@app.post("/predict/42")
@app.post("/api/v1/feat42/predict")
def predict_feat42_endpoint(sample: Feat42Sample):
    try:
        return predict_42(sample.model_dump())
    except Exception as e:
        raise HTTPException(500, f"42-Feature model inference error: {str(e)}")
