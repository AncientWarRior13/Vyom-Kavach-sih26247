"""Reference ONNX inference client for the 42-feature combined model (11
physics + 31 waveform-statistic, 41 unique columns -- see
radarsense/feat42_simulator.py). Same pattern as api/feat31_predict.py,
pointed at feat42_model.onnx / feat42_model_meta.json.

Input contract: 41 pre-computed features (both the 11 physics features
AND the 31 waveform-statistic features for the SAME detection). A caller
with only raw waveforms/physics measurements must compute both feature
sets first -- this server does not do that extraction itself, same
reasoning as api/feat31_server.py's own documented limitation.

Usage:
    python api/feat42_predict.py                # demo on 5 random test rows
    python api/feat42_predict.py sample.json     # single JSON dict of 41 features
"""

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
META = json.loads((ROOT / "artifacts/models/feat42_model_meta.json").read_text())
FEATURES = META["feature_order"]
CLASSES = META["classes"]
DRONE_T = META["drone_threshold_deployed"]

_sess = ort.InferenceSession(str(ROOT / "artifacts/onnx/feat42_model.onnx"),
                             providers=["CPUExecutionProvider"])
_in = _sess.get_inputs()[0].name


def predict_one(sample: dict) -> dict:
    missing = [f for f in FEATURES if f not in sample]
    if missing:
        raise ValueError(f"missing features: {missing}")
    x = np.array([[float(sample[f]) for f in FEATURES]], dtype=np.float32)
    out = _sess.run(None, {_in: x})[-1]
    if isinstance(out, list) and out and isinstance(out[0], dict):
        proba = np.array([out[0][c] for c in CLASSES])
    else:
        proba = np.asarray(out)[0]
    p = {c: float(proba[i]) for i, c in enumerate(CLASSES)}
    drone_idx = CLASSES.index("drone")
    if proba[drone_idx] >= DRONE_T:
        label = "drone"
    else:
        others = [i for i in range(len(CLASSES)) if i != drone_idx]
        label = CLASSES[others[int(np.argmax(proba[others]))]]
    return {"label": label, "probabilities": p,
            "drone_threshold": DRONE_T, "drone_gate_fired": bool(proba[drone_idx] >= DRONE_T)}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(json.dumps(predict_one(json.loads(Path(sys.argv[1]).read_text())), indent=2))
    else:
        import pandas as pd
        test = pd.read_parquet(ROOT / "artifacts/data/feat42_test.parquet")
        for _, row in test.sample(5, random_state=0).iterrows():
            out = predict_one(row[FEATURES].to_dict())
            print(f"true={row['label']:9s} pred={out['label']:9s} "
                  f"p_drone={out['probabilities']['drone']:.3f}")
