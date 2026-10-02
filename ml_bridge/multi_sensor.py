"""Roadmap feature 6 (multi-sensor fusion): EO/IR, RF, and acoustic
channels alongside radar (ml_bridge.radar_classifier is the fourth
channel). Each scenario randomly knocks out one channel for its duration
(`scenario["sensor_outage"]`), forcing the trainee to cross-cue with
whichever channels remain — exactly the roadmap's stated goal.

Each reading is a lightweight physically-motivated proxy, not a full
sensor model (there's no real EO/IR, RF or acoustic hardware/dataset in
this project the way there is a real trained radar classifier) — the
point is the cross-cueing *mechanic* (knowing which sensor to trust when),
not another full signal-processing pipeline.
"""
from __future__ import annotations

import math
import random

SENSORS = ["radar", "eo_ir", "rf", "acoustic"]

# EO/IR: a visual/thermal silhouette. Good in daylight, degrades hard at
# night (no IR imager modeled — just visible-light), worse in cluttered
# terrain. Can only distinguish gross shape (bird vs. drone), never
# hostile/friendly intent — that's a real EO/IR limitation, not a shortcut.
_EO_IR_TIME_FACTOR = {"day": 1.0, "dusk": 0.45, "night": 0.1}
_EO_IR_TERRAIN_FACTOR = {"rural": 1.0, "coastal": 0.9, "mountain": 0.75, "urban": 0.6}

# Acoustic: propeller/motor noise, short effective range, masked by urban
# ambient noise and (partially) by wind/terrain.
_ACOUSTIC_BASE_RANGE_M = {"rural": 600.0, "coastal": 500.0, "mountain": 450.0, "urban": 250.0}


def _position_at(path: list[dict], t: float) -> tuple[float, float, float]:
    if not path:
        return (0.0, 0.0, 0.0)
    if t <= path[0]["t"]:
        p = path[0]
        return (p["x"], p["y"], p["z"])
    if t >= path[-1]["t"]:
        p = path[-1]
        return (p["x"], p["y"], p["z"])
    for a, b in zip(path, path[1:]):
        if a["t"] <= t <= b["t"]:
            f = 0.0 if b["t"] == a["t"] else (t - a["t"]) / (b["t"] - a["t"])
            return (a["x"] + (b["x"] - a["x"]) * f, a["y"] + (b["y"] - a["y"]) * f, a["z"] + (b["z"] - a["z"]) * f)
    p = path[-1]
    return (p["x"], p["y"], p["z"])


def eo_ir_reading(contact: dict, t: float, time_of_day: str, terrain: str) -> dict:
    confidence = _EO_IR_TIME_FACTOR.get(time_of_day, 0.5) * _EO_IR_TERRAIN_FACTOR.get(terrain, 0.8)
    confidence = max(0.0, min(1.0, confidence + random.gauss(0, 0.08)))
    detected = confidence >= 0.3
    shape_hint = "bird" if contact["contact_type"] == "bird" else ("drone" if detected else "unknown")
    return {"sensor": "eo_ir", "available": True, "detected": detected,
            "confidence": round(confidence, 2), "shape_hint": shape_hint if detected else "unknown"}


def rf_reading(contact: dict) -> dict:
    """RF-silent contacts (roadmap feature 7's red-team tactic) are
    invisible to this channel regardless of range/terrain — that's the
    whole point of the tactic."""
    if contact.get("rf_silent"):
        return {"sensor": "rf", "available": True, "detected": False, "note": "no RF emission detected"}
    return {"sensor": "rf", "available": True, "detected": True, "note": "control/telemetry RF emission detected"}


def acoustic_reading(contact: dict, t: float, asset_xy: tuple[float, float], terrain: str) -> dict:
    x, y, _ = _position_at(contact["path"], t)
    dist = math.hypot(x - asset_xy[0], y - asset_xy[1])
    effective_range = _ACOUSTIC_BASE_RANGE_M.get(terrain, 500.0)
    detected = dist <= effective_range
    return {"sensor": "acoustic", "available": True, "detected": detected,
            "range_m": round(dist, 1), "effective_range_m": effective_range}


def pick_sensor_outage(difficulty: int) -> str | None:
    """~40% chance at difficulty>=2 that one channel is jammed/unavailable
    for the whole scenario; radar is excluded here since the real ONNX
    classifier hint already has its own independent noise/degradation path."""
    if difficulty < 2 or random.random() > 0.4:
        return None
    return random.choice(["eo_ir", "rf", "acoustic"])


def read_sensor(sensor: str, contact: dict, t: float, time_of_day: str, terrain: str,
                 asset_xy: tuple[float, float], outage: str | None) -> dict:
    if sensor == outage:
        return {"sensor": sensor, "available": False, "reason": "jammed/unavailable this scenario"}
    if sensor == "eo_ir":
        return eo_ir_reading(contact, t, time_of_day, terrain)
    if sensor == "rf":
        return rf_reading(contact)
    if sensor == "acoustic":
        return acoustic_reading(contact, t, asset_xy, terrain)
    raise ValueError(f"unknown sensor: {sensor!r}")
