"""Live, single-detection SHAP + LIME explanation -- the deployment
counterpart to explain.py (which runs offline, on a random test-set
instance, writing static PNGs). This module answers "why did the model call
*this specific* live detection a drone/bird/aircraft/anomaly", for any
feature vector the API is handed, not a pre-picked example.

Both explainers wrap the same artifacts/models/final_model.joblib CatBoost
model already validated by explain.py's physics-sanity check (micro-Doppler
features must rank top-3 for the drone class) -- this module doesn't retrain
or re-derive anything, it explains individual predictions from the existing
model on demand.

- SHAP: shap.TreeExplainer, exact (not sampled) -- cheap and exact for tree
  models, same explainer type explain.py uses.
- LIME: LimeTabularExplainer over a background sample of the real training
  distribution (radar_train.parquet), same num_samples=500 convention as
  ensemble_decision_explainer.py / feat42_explain.py.

api/explain.py calls into this and renders the result to PDF.
"""

import json

import joblib
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

from radarsense import config as C

LIME_NUM_SAMPLES = 500
LIME_NUM_FEATURES = 11

_model = joblib.load(C.MODELS_DIR / "final_model.joblib")
_meta = json.loads((C.MODELS_DIR / "final_model_meta.json").read_text())
FEATURES = _meta["feature_order"]
CLASSES = _meta["classes"]
DRONE_T = _meta["drone_threshold_deployed"]

_shap_explainer = shap.TreeExplainer(_model)

_train_path = C.DATA_DIR / "radar_train.parquet"
if not _train_path.exists():
    from radarsense.simulator import main as _gen_data
    _gen_data()
_train_full = pd.read_parquet(_train_path)
_train_bg = _train_full[FEATURES].sample(min(1000, len(_train_full)), random_state=C.SEED)
_lime_explainer = LimeTabularExplainer(
    _train_bg.values,
    feature_names=FEATURES,
    class_names=CLASSES,
    mode="classification",
    random_state=C.SEED,
)


def _predict_proba_df(x: np.ndarray) -> np.ndarray:
    return _model.predict_proba(pd.DataFrame(x, columns=FEATURES))


def predict(sample: dict) -> dict:
    """Same drone-gate decision rule as api/predict.py (kept independent so
    this module has no import-time dependency on the ONNX runtime)."""
    missing = [f for f in FEATURES if f not in sample]
    if missing:
        raise ValueError(f"missing features: {missing}")
    row = pd.DataFrame([[float(sample[f]) for f in FEATURES]], columns=FEATURES)
    proba = _model.predict_proba(row)[0]
    p = {c: float(proba[i]) for i, c in enumerate(CLASSES)}
    drone_idx = CLASSES.index("drone")
    if proba[drone_idx] >= DRONE_T:
        label = "drone"
    else:
        others = [i for i in range(len(CLASSES)) if i != drone_idx]
        label = CLASSES[others[int(np.argmax(proba[others]))]]
    return {"label": label, "probabilities": p, "drone_threshold": DRONE_T,
            "drone_gate_fired": bool(proba[drone_idx] >= DRONE_T)}


def shap_explanation(sample: dict, target_class: str) -> dict:
    """Exact SHAP contributions for `target_class` on this one sample."""
    row = pd.DataFrame([[float(sample[f]) for f in FEATURES]], columns=FEATURES)
    sv = _shap_explainer.shap_values(row)
    if isinstance(sv, np.ndarray) and sv.ndim == 3:
        sv = [sv[:, :, k] for k in range(sv.shape[2])]
    class_idx = CLASSES.index(target_class)
    values = sv[class_idx][0]
    ev = _shap_explainer.expected_value
    base_value = float(ev[class_idx] if np.ndim(ev) else ev)

    contributions = sorted(
        [{"feature": f, "value": float(row.iloc[0][f]), "shap_value": float(v)}
         for f, v in zip(FEATURES, values)],
        key=lambda c: -abs(c["shap_value"]),
    )
    return {"target_class": target_class, "base_value": base_value,
            "contributions": contributions}


def lime_explanation(sample: dict, target_class: str) -> dict:
    """LIME feature weights for `target_class` on this one sample."""
    x = np.array([float(sample[f]) for f in FEATURES])
    class_idx = CLASSES.index(target_class)
    exp = _lime_explainer.explain_instance(
        x, _predict_proba_df, labels=(class_idx,),
        num_features=LIME_NUM_FEATURES, num_samples=LIME_NUM_SAMPLES,
    )
    weights = [{"feature": f, "weight": float(w)} for f, w in exp.as_list(label=class_idx)]
    weights.sort(key=lambda w: -abs(w["weight"]))
    return {"target_class": target_class, "weights": weights,
            "local_pred_proba": float(exp.local_pred[0]) if exp.local_pred is not None else None}


def explain(sample: dict) -> dict:
    """Full bundle: prediction + SHAP + LIME, both computed for the
    predicted class (and for drone specifically too, if that's not already
    the predicted class -- the operationally important number either way)."""
    pred = predict(sample)
    predicted_class = pred["label"]
    result = {
        "sample": {f: float(sample[f]) for f in FEATURES},
        "prediction": pred,
        "shap": {predicted_class: shap_explanation(sample, predicted_class)},
        "lime": {predicted_class: lime_explanation(sample, predicted_class)},
    }
    if predicted_class != "drone":
        result["shap"]["drone"] = shap_explanation(sample, "drone")
        result["lime"]["drone"] = lime_explanation(sample, "drone")
    return result
