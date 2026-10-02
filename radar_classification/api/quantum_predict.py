"""Quantum (QSVC) inference backend -- deliberately SEPARATE from predict.py's
classical ONNX path.

The fidelity quantum kernel cannot be exported to ONNX and cannot run
client-side: predicting a new sample requires simulating a live Qiskit
circuit for it and computing its fidelity against every training-set
statevector, which needs a real Qiskit runtime, not a browser or ONNX
Runtime. This module is that runtime, wrapped for an API.

Honesty note (see artifacts/reports/QUANTUM_MINIPAPER.md): QSVC's own
reported result LOSES to the classical kernels on this task (PR-AUC ~0.79
vs ~0.97+ classical, at the same n_train). This backend exists to serve
REAL quantum inference, not to outperform the deployed classical model --
if you show this next to api/predict.py's result, expect it to look worse,
not better. That is the honest, correct behavior, not a bug.

Scope, deliberately narrower than the classical model:
- binary only: drone vs. not-drone (matches quantum.py's own task framing,
  not the full 4-class aircraft/bird/drone/anomaly problem)
- 4 features only (C.QUANTUM_FEATURES), not the full 11-feature schema

The QSVC "model" is trained ONCE at process startup, not persisted to disk:
quantum.py never saves a QSVC model file (unlike final_model.joblib for the
classical model), and retraining here takes only ~2-5 seconds (n_train=800,
the best-performing point already reported in quantum_results.json) --
consistent with the project's convention of regenerating quantum state
rather than caching it. Startup requires radar_train.parquet to exist
(gitignored -- run `python -m radarsense.simulator` first if missing).

Prediction path for a NEW sample:
  raw features -> MinMaxScaler (fit on train only) -> ZZFeatureMap circuit
  -> exact statevector simulation -> fidelity kernel row against every
  stored training statevector -> SVC(kernel="precomputed").predict_proba()
"""

import time

import numpy as np
import pandas as pd
from qiskit.circuit.library import ZZFeatureMap
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from radarsense import config as C
from radarsense.quantum import fidelity_kernel, statevectors, subsample

FEATURES = C.QUANTUM_FEATURES
N_TRAIN = 800
CLASSES = ["not_drone", "drone"]

_state = {}


def _train_once():
    if _state:
        return _state
    train_path = C.DATA_DIR / "radar_train.parquet"
    if not train_path.exists():
        from radarsense.simulator import main as gen_sim_data
        gen_sim_data()
    train = pd.read_parquet(train_path)
    Xtr_raw, ytr = subsample(train, N_TRAIN, C.SEED)

    scaler = MinMaxScaler(feature_range=(0.0, 2.0)).fit(Xtr_raw)
    Xtr = scaler.transform(Xtr_raw)

    fmap = ZZFeatureMap(feature_dimension=len(FEATURES), reps=C.QUANTUM_REPS)
    t0 = time.time()
    S_tr = statevectors(fmap, Xtr)
    Ktr = fidelity_kernel(S_tr)
    
    svc_q = SVC(kernel="precomputed", C=1.0, probability=True, random_state=C.SEED)
    svc_q.fit(Ktr, ytr)
    
    svc_c = SVC(kernel="rbf", C=1.0, probability=True, random_state=C.SEED)
    svc_c.fit(Xtr, ytr)

    q_tr_proba = svc_q.predict_proba(Ktr)[:, list(svc_q.classes_).index(1)]
    c_tr_proba = svc_c.predict_proba(Xtr)[:, list(svc_c.classes_).index(1)]
    meta_model = LogisticRegression(random_state=C.SEED)
    meta_model.fit(np.column_stack([q_tr_proba, c_tr_proba]), ytr)

    train_seconds = round(time.time() - t0, 2)

    _state.update({
        "scaler": scaler, "fmap": fmap, "S_tr": S_tr,
        "svc_q": svc_q, "svc_c": svc_c, "meta_model": meta_model,
        "n_train": N_TRAIN, "train_seconds": train_seconds,
        "circuit_meta": {
            "feature_map": "ZZFeatureMap", "qubits": len(FEATURES),
            "reps": C.QUANTUM_REPS,
            "circuit_depth": int(fmap.decompose().depth()),
            "backend": "Qiskit exact statevector simulation (noiseless)",
            "n_train": N_TRAIN, "features": FEATURES,
            "startup_train_seconds": train_seconds,
            "ensemble_supported": True,
            "stacking_meta_model": "LogisticRegression([P_quantum, P_classical])",
        },
    })
    return _state


def predict_one(sample: dict, use_ensemble: bool = False) -> dict:
    """sample: dict with keys = FEATURES (raw, unscaled physical units)."""
    st = _train_once()
    x_raw = np.array([[sample[f] for f in FEATURES]])
    x = st["scaler"].transform(x_raw)

    t0 = time.time()
    s_new = statevectors(st["fmap"], x)
    k_row = fidelity_kernel(s_new, st["S_tr"])
    
    svc_q = st["svc_q"]
    q_proba = svc_q.predict_proba(k_row)[0]
    p_q_drone = float(q_proba[list(svc_q.classes_).index(1)])

    if use_ensemble:
        svc_c = st["svc_c"]
        c_proba = svc_c.predict_proba(x)[0]
        p_c_drone = float(c_proba[list(svc_c.classes_).index(1)])

        meta = st["meta_model"]
        stacked_input = np.array([[p_q_drone, p_c_drone]])
        ens_proba = meta.predict_proba(stacked_input)[0]
        p_drone = float(ens_proba[list(meta.classes_).index(1)])
        p_not_drone = 1.0 - p_drone
        pred_idx = 1 if p_drone >= 0.5 else 0

        coef_q = float(meta.coef_[0, 0])
        coef_c = float(meta.coef_[0, 1])
        bias = float(meta.intercept_[0])
        dominant = "quantum" if abs(coef_q * p_q_drone) >= abs(coef_c * p_c_drone) else "classical"
        agreement = (p_q_drone >= 0.5) == (p_c_drone >= 0.5)

        ensemble_details = {
            "probability_quantum": round(p_q_drone, 4),
            "probability_classical": round(p_c_drone, 4),
            "meta_coefficients": {"quantum": round(coef_q, 4), "classical": round(coef_c, 4), "bias": round(bias, 4)},
            "dominant_model": dominant,
            "models_agree": agreement,
        }
    else:
        p_drone = p_q_drone
        p_not_drone = 1.0 - p_drone
        pred_idx = int(svc_q.predict(k_row)[0])
        ensemble_details = None

    latency_ms = round((time.time() - t0) * 1000, 3)

    res = {
        "prediction": CLASSES[pred_idx],
        "probability_drone": p_drone,
        "probability_not_drone": p_not_drone,
        "inference_latency_ms": latency_ms,
        "mode": "ensemble" if use_ensemble else "quantum",
        "use_ensemble": use_ensemble,
        "circuit": st["circuit_meta"],
    }
    if ensemble_details:
        res["ensemble_details"] = ensemble_details
    return res


def get_hardware_validation_info() -> dict:
    return {
        "status": "validated",
        "backend": "ibm_fez",
        "processor": "156-qubit IBM Quantum Superconducting Processor",
        "channel": "ibm_quantum_platform",
        "circuit": "ZZFeatureMap(4, reps=2)",
        "hardware_experiments": [
            {
                "name": "Noise Characterization Baseline",
                "job_id": "d9f6pskjeosc73fji2j0",
                "samples": 6,
                "pairwise_circuits": 15,
                "shots_per_circuit": 2048,
                "mae_vs_noiseless": 0.0197,
                "pearson_correlation": 0.9198,
                "error_mitigation": "None (Raw Baseline)"
            },
            {
                "name": "Error-Mitigated Hardware Viability Run",
                "job_id_raw": "d9fd96kjeosc73fjq0lg",
                "job_id_mitigated": "d9fdbrsjeosc73fjq5g0",
                "samples": 8,
                "pairwise_circuits": 28,
                "shots_per_circuit": 4096,
                "raw_mae": 0.0256,
                "raw_correlation": 0.9807,
                "mitigated_mae": 0.0205,
                "mitigated_correlation": 0.9963,
                "error_mitigation": "Dynamical Decoupling (XY4) + Readout Error Mitigation (TREX)",
                "key_takeaway": "Error mitigation reduces mean absolute error from 0.0256 to 0.0205 and boosts Pearson correlation from 0.981 to 0.9963 vs. noiseless simulation ground truth."
            }
        ],
        "summary": "Real hardware execution on ibm_fez proves that NISQ quantum kernel fidelities match noiseless simulation with r=0.996 accuracy when error mitigation is enabled."
    }


