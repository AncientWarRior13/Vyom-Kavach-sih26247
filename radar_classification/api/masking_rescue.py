"""Masking-rescue routing layer (deployment option 3).

Keeps the clean ONNX model's decision EVERYWHERE and only ESCALATES a
non-drone prediction to drone when micro_doppler_bandwidth reads anomalously
low (a masking signature) AND the kinematic fallback model -- which never
sees micro-Doppler, so masking cannot fool it -- is >= RESCUE_GATE confident
the target is a drone.

Why this preserves clean performance: a clean sample only gets touched if its
micro-Doppler is already below the drone-class cutoff AND the fallback is
near-certain it is a drone. Measured effect at gate=0.98
(artifacts/reports/masking_defense_stacked_results.json): clean accuracy
identical to baseline (97.59%), clean drone recall +1pt (91.8->92.8%), clean
precision within 0.5pt (77.5->77.0%), severe-masking (x0.3) drone recall
12.3% -> 32.8% (2.7x). The gate is a dial -- lower it (e.g. 0.90) for more
masking recovery (up to ~4x) at a precision cost.
"""

import json

import joblib
import numpy as np

from radarsense import config as C

_META = json.loads((C.MODELS_DIR / "masking_defense_meta.json").read_text())
_FALLBACK = joblib.load(C.MODELS_DIR / "masking_defense_fallback.joblib")
_NON_MD = _META["non_md_features"]
_CUTOFF = float(_META["suspicion_cutoff"])
_DRONE_IDX = list(_FALLBACK.classes_).index("drone")

RESCUE_GATE = 0.98


def apply_masking_rescue(sample: dict, primary_label: str) -> dict:
    """Escalate a non-drone prediction to drone only when masking is suspected
    (low micro-Doppler) and the micro-Doppler-immune fallback is near-certain.
    Returns {label, masking_rescue, fallback_drone_prob?}."""
    if primary_label == "drone" or float(sample["micro_doppler_bandwidth"]) >= _CUTOFF:
        return {"label": primary_label, "masking_rescue": False}
    x = np.array([[float(sample[f]) for f in _NON_MD]])
    fb_p = float(_FALLBACK.predict_proba(x)[0, _DRONE_IDX])
    rescued = fb_p >= RESCUE_GATE
    return {"label": "drone" if rescued else primary_label,
            "masking_rescue": rescued, "fallback_drone_prob": round(fb_p, 4)}
