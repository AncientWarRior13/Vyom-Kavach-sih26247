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
import threading
import time
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


# Roadmap training-time-cutoff step 5 (background retrain + hot-swap): a
# plain mutable slot guarded by a lock, not functools.lru_cache — the whole
# point is that this CAN change while the server is running, atomically,
# without ever blocking a classify_contact() call on a retrain in progress.
_session_lock = threading.Lock()
_session_cache: tuple | None = None
_session_onnx_path: Path = ONNX_PATH
_model_status = {"state": "idle", "last_trained_at": None, "last_error": None, "onnx_path": str(ONNX_PATH)}


def _load_session(onnx_path: Path) -> tuple | None:
    try:
        import onnxruntime as ort
        meta = json.loads(META_PATH.read_text())
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        # Balanced C-UAS early warning threshold (0.20 vs civilian-only 0.54) to alert on incoming drone vectors at long ranges
        threshold = meta.get("drone_threshold_operational", 0.20)
        return sess, sess.get_inputs()[0].name, meta["feature_order"], meta["classes"], threshold
    except Exception:
        return None  # onnxruntime not installed, or artifacts missing — fall back to a heuristic


def _session():
    global _session_cache
    with _session_lock:
        if _session_cache is None:
            _session_cache = _load_session(_session_onnx_path)
        return _session_cache


def get_model_status() -> dict:
    with _session_lock:
        return dict(_model_status)


def retrain_async(train_fn, new_onnx_path: str | Path, budget_s: float = 600.0) -> None:
    """Roadmap steps 5+6: runs `train_fn` (expected to write a new ONNX
    model to `new_onnx_path`) in a background thread under a wall-clock
    cutoff, and only hot-swaps the serving session to the new model if
    training actually finished in time and produced a usable file — the UI
    never blocks on this, and the old model keeps serving the whole time.
    `train_fn` is expected to be `functools.partial`-bound with its own
    args; it's called with no arguments here.
    """
    def job():
        global _session_cache
        with _session_lock:
            _model_status["state"] = "training"
            _model_status["last_error"] = None
        start = time.time()
        try:
            done = threading.Event()
            result: dict = {}

            def run():
                try:
                    status = train_fn()  # may return "trained"/"timeout_keep_current_model"/
                    result["ok"] = status != "failed_keep_current_model" and status is not None and status != "timeout_keep_current_model"
                    result["status"] = status  # "failed_keep_current_model" (see training_utils.train_with_cutoff), or None/other
                except Exception as e:  # noqa: BLE001 - surfaced via status, not raised in a thread
                    result["ok"] = False
                    result["error"] = str(e)
                finally:
                    done.set()

            worker = threading.Thread(target=run, daemon=True)
            worker.start()
            finished = done.wait(timeout=budget_s)

            with _session_lock:
                if not finished:
                    _model_status["state"] = "timeout_keep_current_model"
                    _model_status["last_error"] = f"training exceeded {budget_s}s budget"
                elif result.get("status") == "timeout_keep_current_model":
                    # train_fn used its own subprocess-level cutoff (e.g.
                    # training_utils.train_with_cutoff) and hit it first
                    _model_status["state"] = "timeout_keep_current_model"
                    _model_status["last_error"] = "training script's own subprocess timeout fired"
                elif not result.get("ok"):
                    _model_status["state"] = "failed_keep_current_model"
                    _model_status["last_error"] = result.get("error") or "train_fn reported failure"
                elif not Path(new_onnx_path).exists():
                    _model_status["state"] = "failed_keep_current_model"
                    _model_status["last_error"] = "train_fn finished but produced no model file"
                else:
                    new_session = _load_session(Path(new_onnx_path))
                    if new_session is None:
                        _model_status["state"] = "failed_keep_current_model"
                        _model_status["last_error"] = "new model file failed to load"
                    else:
                        _session_cache = new_session
                        _model_status["state"] = "trained"
                        _model_status["last_trained_at"] = time.time()
                        _model_status["onnx_path"] = str(new_onnx_path)
        except Exception as e:  # noqa: BLE001
            with _session_lock:
                _model_status["state"] = "failed_keep_current_model"
                _model_status["last_error"] = str(e)
        finally:
            with _session_lock:
                _model_status["duration_s"] = round(time.time() - start, 1)

    threading.Thread(target=job, daemon=True).start()


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
    if dsp["micro_doppler_jitter"] > 0.05:
        micro_doppler_jitter = max(0.0, min(1.0, dsp["micro_doppler_jitter"]))
    else:
        micro_doppler_jitter = max(0.05, min(1.0, jit_base + random.gauss(0, 0.05 * noise_scale)))

    if dsp["micro_doppler_bandwidth"] > 90.0:
        micro_doppler_bandwidth = max(0.0, min(500.0, dsp["micro_doppler_bandwidth"]))
    else:
        micro_doppler_bandwidth = max(0.0, min(500.0, mdb_base + random.gauss(0, 15.0 * noise_scale)))

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


def precompute_hints(scenario: dict, draws_per_contact: int = 3) -> None:
    """Roadmap step 7: runs the real CFAR+ONNX classification path for each
    contact at a few random times across its active window right when the
    scenario is created (call this in a background thread, not inline —
    it's the same real DSP+model work `classify_contact` does, just done
    ahead of time), caching the draws in `scenario["_precomputed_hints"]`.
    `/api/contact/classify` then answers near-instantly by returning one of
    these draws at random instead of recomputing — this trainer's own
    classification is already randomized-by-design per draw (synthesized
    I/Q noise), so returning a cached draw instead of a fresh one changes
    nothing about what the trainee experiences.
    """
    degradation = scenario["params"].get("effective_sensor_degradation", scenario["params"].get("sensor_degradation", 0.0))
    siblings = scenario["contacts"]
    cache = scenario.setdefault("_precomputed_hints", {})
    for contact in scenario["contacts"]:
        window = (contact["spawn_t"], contact["despawn_t"])
        draws = []
        for _ in range(draws_per_contact):
            t = random.uniform(*window) if window[1] > window[0] else window[0]
            result = classify_contact(contact, t, degradation, siblings=siblings)
            draws.append({"label": result["label"], "probabilities": result["probabilities"], "source": result["source"]})
        cache[contact["contact_id"]] = draws


def get_precomputed_hint(scenario: dict, contact_id: str) -> dict | None:
    draws = scenario.get("_precomputed_hints", {}).get(contact_id)
    return random.choice(draws) if draws else None
