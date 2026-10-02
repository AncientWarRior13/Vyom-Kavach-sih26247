"""Bridges scenario_engine contacts to RadarSense's real trained CatBoost
ONNX classifier (`radar_classification/artifacts/onnx/radarsense.onnx`).

The scenario engine knows ground truth (what a contact actually is). A real
sensor doesn't. There's no physical radar in this trainer, so there's no
real captured I/Q either — but rather than skip straight to formula-estimated
feature values, `iq_synthesizer.py` embeds a contact's true kinematics (radial
velocity, RCS) and a type-conditioned rotor/wing micro-motion signature as
actual frequency-domain content in a synthesized I/Q segment, then runs it
through `radar_classification/radarsense/radar_dsp.py`'s genuine range-Doppler
FFT + CA-CFAR detector — the same real signal-processing chain RadarSense
validated on actual 77GHz data. doppler_velocity/snr_db/micro_doppler_* are
therefore *measured* by that real DSP (and can come out noisier, bin-
quantized, or even undetected at low SNR — a real missed detection), not
assigned from a lookup table. altitude/range/azimuth remain direct geometry
(correctly — CFAR doesn't need to "recover" those), and the resulting
11-feature vector runs through the real trained model. The trainee sees this
model's output, not the ground truth label — exactly like a real operator
would see a classifier's call on their scope, which can be wrong, especially
for decoys built to mimic a drone signature.
"""
from __future__ import annotations

import json
import math
import random
from functools import lru_cache
from pathlib import Path

from . import iq_synthesizer as iq

RADARSENSE_ROOT = Path(__file__).resolve().parent.parent / "radar_classification"
ONNX_PATH = RADARSENSE_ROOT / "artifacts" / "onnx" / "radarsense.onnx"
META_PATH = RADARSENSE_ROOT / "artifacts" / "models" / "final_model_meta.json"

C_LIGHT = 3e8
F_CARRIER = 10e9  # 10 GHz, matches RadarSense's feature derivation

# type-conditioned physical signature priors: (rcs_dbsm, micro_doppler_bw_hz, micro_doppler_jitter)
TYPE_PROFILE = {
    "hostile_drone": (-6.0, 220.0, 0.55),
    "hostile_swarm_member": (-8.0, 240.0, 0.6),
    "friendly_drone": (-6.0, 220.0, 0.55),
    "decoy": (-7.0, 180.0, 0.45),       # built to mimic a drone signature, imperfectly
    "bird": (-14.0, 90.0, 0.7),
    "aircraft": (8.0, 20.0, 0.1),
}


@lru_cache(maxsize=1)
def _session():
    try:
        import onnxruntime as ort
        meta = json.loads(META_PATH.read_text())
        sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
        return sess, sess.get_inputs()[0].name, meta["feature_order"], meta["classes"], meta["drone_threshold_deployed"]
    except Exception:
        return None  # onnxruntime not installed, or artifacts missing — fall back to a heuristic


def _velocity_at(path: list[dict], t: float) -> tuple[float, float, float]:
    pts = path
    i = 0
    while i < len(pts) - 1 and pts[i + 1]["t"] < t:
        i += 1
    a, b = pts[i], pts[min(i + 1, len(pts) - 1)]
    dt = max(1e-6, b["t"] - a["t"])
    return ((b["x"] - a["x"]) / dt, (b["y"] - a["y"]) / dt, (b["z"] - a["z"]) / dt)


def _position_at(path: list[dict], t: float) -> tuple[float, float, float]:
    pts = path
    if t <= pts[0]["t"]:
        p = pts[0]
        return p["x"], p["y"], p["z"]
    if t >= pts[-1]["t"]:
        p = pts[-1]
        return p["x"], p["y"], p["z"]
    for a, b in zip(pts, pts[1:]):
        if a["t"] <= t <= b["t"]:
            f = 0 if b["t"] == a["t"] else (t - a["t"]) / (b["t"] - a["t"])
            return (a["x"] + (b["x"] - a["x"]) * f, a["y"] + (b["y"] - a["y"]) * f, a["z"] + (b["z"] - a["z"]) * f)
    p = pts[-1]
    return p["x"], p["y"], p["z"]


def _swarm_density_at(contact: dict, t: float, siblings: list[dict] | None) -> int:
    """Count of other same-swarm members within ~150m and active at time t.
    Dense swarm formations produce overlapping micro-Doppler returns in a
    real radar — the sparse literature on swarm (vs. single-drone) micro-
    Doppler classification (see research/PROTOCOLS.md) flags this as an
    unmodeled confound; we fold it into the bandwidth/jitter features here
    rather than ignoring it.
    """
    if not siblings or not contact.get("swarm_id"):
        return 0
    x, y, _ = _position_at(contact["path"], t)
    count = 0
    for s in siblings:
        if s["contact_id"] == contact["contact_id"] or s.get("swarm_id") != contact["swarm_id"]:
            continue
        if not (s["spawn_t"] <= t <= s["despawn_t"]):
            continue
        sx, sy, _ = _position_at(s["path"], t)
        if math.hypot(x - sx, y - sy) <= 150.0:
            count += 1
    return count


def derive_features(contact: dict, t: float, degradation: float, siblings: list[dict] | None = None) -> dict:
    path = contact["path"]
    x, y, z = _position_at(path, t)
    vx, vy, vz = _velocity_at(path, t)
    speed = math.sqrt(vx * vx + vy * vy + vz * vz)
    range_m = max(1.0, math.sqrt(x * x + y * y))
    azimuth_deg = math.degrees(math.atan2(y, x))

    radial_unit = (x / range_m, y / range_m) if range_m else (0, 0)
    true_doppler_velocity = max(-50.0, min(50.0, vx * radial_unit[0] + vy * radial_unit[1]))

    # roadmap feature 13 (spoofed-signature scenarios): a contact crafted to
    # evade classification (RadarSense's adversarial-robustness framing)
    # presents whatever signature it's spoofing, not its true type's —
    # exactly the real sensor's blind spot a spoofed target exploits.
    signature_type = contact.get("spoofed_as") or contact["contact_type"]
    rcs_base, mdb_base, jit_base = TYPE_PROFILE.get(signature_type, TYPE_PROFILE["hostile_drone"])
    noise_scale = 1.0 + degradation * 2.0

    swarm_density = _swarm_density_at(contact, t, siblings)
    # overlapping returns from nearby swarm members widen/roughen the
    # apparent micro-Doppler signature of any one member
    mdb_base = mdb_base + swarm_density * 15.0
    jit_base = min(1.0, jit_base + swarm_density * 0.03)

    rcs_estimate = max(-20.0, min(20.0, rcs_base + random.gauss(0, 2.0 * noise_scale)))
    true_snr_db = max(-20.0, min(-10.0, -10.0 - (range_m / 500.0) - degradation * 5.0 + random.gauss(0, 1.0)))

    # real signal-processing path: synthesize I/Q embedding the true
    # kinematics + a type-conditioned rotor/wing micro-motion signature as
    # actual frequency-domain content, then run RadarSense's genuine
    # range-Doppler FFT + CA-CFAR chain on it. doppler_velocity/snr_db/
    # micro_doppler_* below are MEASURED by that real DSP, not assigned.
    micro_motion_hz = [
        (mdb_base * 0.3, jit_base * 0.6),
        (mdb_base * 0.6, jit_base * 0.4),
        (mdb_base * 0.9, jit_base * 0.25),
    ]
    dsp = iq.dsp_measure(true_doppler_velocity, rcs_estimate, true_snr_db, micro_motion_hz)

    doppler_velocity = max(-50.0, min(50.0, dsp["doppler_velocity"]))
    doppler_freq_shift = 2 * doppler_velocity * F_CARRIER / C_LIGHT
    snr_db = max(-20.0, min(-10.0, dsp["snr_db"]))
    micro_doppler_bandwidth = max(0.0, min(500.0, dsp["micro_doppler_bandwidth"]))
    micro_doppler_jitter = max(0.0, min(1.0, dsp["micro_doppler_jitter"]))

    altitude_m = max(0.0, min(3000.0, z))
    velocity_variance = max(0.0, speed * 0.05 * noise_scale)
    acceleration = random.gauss(0, 0.5 * noise_scale)

    return {
        "doppler_velocity": doppler_velocity,
        "doppler_freq_shift": doppler_freq_shift,
        "rcs_estimate": rcs_estimate,
        "micro_doppler_bandwidth": micro_doppler_bandwidth,
        "micro_doppler_jitter": micro_doppler_jitter,
        "range_m": max(50.0, min(5000.0, range_m)),
        "azimuth_deg": azimuth_deg,
        "altitude_m": altitude_m,
        "velocity_variance": velocity_variance,
        "acceleration": acceleration,
        "snr_db": snr_db,
    }


def _heuristic_classify(features: dict) -> dict:
    """Fallback used only if onnxruntime / the trained artifacts aren't
    available in this environment — keeps the trainer runnable everywhere,
    but the real path above is the one that's actually "the model"."""
    mdb = features["micro_doppler_bandwidth"]
    rcs = features["rcs_estimate"]
    if rcs > 4:
        label = "aircraft"
    elif mdb < 120 and rcs < -10:
        label = "bird"
    elif mdb >= 150:
        label = "drone"
    else:
        label = "anomaly"
    return {"label": label, "probabilities": {label: 0.6}, "source": "heuristic_fallback"}


def classify_contact(contact: dict, t: float, degradation: float = 0.0, siblings: list[dict] | None = None) -> dict:
    """Returns the sensor's classification call for `contact` at time `t`:
    {"label": one of aircraft/anomaly/bird/drone, "probabilities": {...}}.
    This is what the trainer shows the trainee — it is independent of (and
    can disagree with) the scenario's ground truth `contact_type`.

    `siblings` (optional): the scenario's other contacts, used only to
    compute swarm density (see `_swarm_density_at`) — pass the full contact
    list when classifying a swarm member for the density effect to apply.
    """
    features = derive_features(contact, t, degradation, siblings)
    loaded = _session()
    if loaded is None:
        return {**_heuristic_classify(features), "features": features}

    import numpy as np
    sess, input_name, feature_order, classes, drone_threshold = loaded
    x = np.array([[float(features[f]) for f in feature_order]], dtype=np.float32)
    out = sess.run(None, {input_name: x})[-1]
    if isinstance(out, list) and out and isinstance(out[0], dict):
        proba = np.array([out[0][c] for c in classes])
    else:
        proba = np.asarray(out)[0]
    drone_idx = classes.index("drone")
    if proba[drone_idx] >= drone_threshold:
        label = "drone"
    else:
        others = [i for i in range(len(classes)) if i != drone_idx]
        label = classes[others[int(np.argmax(proba[others]))]]
    return {
        "label": label,
        "probabilities": {c: float(proba[i]) for i, c in enumerate(classes)},
        "source": "radarsense_onnx",
        "features": features,
    }
