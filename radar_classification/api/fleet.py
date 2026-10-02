"""Fleet-scale tracking API (deployment layer for radarsense/fleet_tracking.py).

Stateful multi-target tracking across many HTTP calls needs a server-side
session: each call submits one frame's worth of raw detections and gets back
the updated track bank + fleet-aggregate summary, because the Kalman filters'
state has to persist between frames. Sessions are in-process, in-memory
(no auth/DB per this project's finals-hardening scope decision -- see
TODO.md) with a simple cap + LRU eviction so a demo left running doesn't leak
memory unbounded.

Mounted into api/app.py as `/fleet/*`.
"""

import time
import uuid
from collections import OrderedDict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from radarsense.fleet_tracking import FleetTracker
from radarsense import config as C

router = APIRouter(prefix="/fleet", tags=["fleet"])

MAX_SESSIONS = 50
SESSION_TTL_S = 3600

_sessions: "OrderedDict[str, dict]" = OrderedDict()


def _evict_stale():
    now = time.time()
    stale = [sid for sid, s in _sessions.items() if now - s["last_seen"] > SESSION_TTL_S]
    for sid in stale:
        del _sessions[sid]
    while len(_sessions) > MAX_SESSIONS:
        _sessions.popitem(last=False)


class Detection(BaseModel):
    range_bin: float
    doppler_bin: float
    snr_db: float = 0.0


class FrameRequest(BaseModel):
    detections: list[Detection] = Field(default_factory=list)


def _track_view(tracker: FleetTracker):
    return [
        {
            "id": t["id"],
            "range_bin": round(float(t["kf"].x[0]), 2) if t["kf"].x is not None else None,
            "doppler_bin": round(float(t["kf"].x[1]), 2) if t["kf"].x is not None else None,
            "hits": t["hits"],
            "coast": t["coast"],
            "confirmed": t["hits"] >= tracker.confirm_hits,
        }
        for t in tracker.tracks
    ]


@router.get("/health")
@router.get("/api/fleet/health")
def fleet_health():
    return {
        "status": "ok",
        "service": "RadarSense Fleet-Scale Swarm Tracking Engine",
        "active_sessions": len(_sessions),
        "max_sessions": MAX_SESSIONS,
        "session_ttl_sec": SESSION_TTL_S,
    }


@router.post("/session")
def create_session():
    _evict_stale()
    session_id = uuid.uuid4().hex
    _sessions[session_id] = {"tracker": FleetTracker(), "frame": 0, "last_seen": time.time()}
    _sessions.move_to_end(session_id)
    return {"session_id": session_id}


@router.get("/session/{session_id}")
def get_session(session_id: str):
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "unknown or expired fleet session")
    _sessions.move_to_end(session_id)
    tracker: FleetTracker = session["tracker"]
    return {
        "session_id": session_id,
        "frame": session["frame"],
        "tracks": _track_view(tracker),
        "fleet_summary": tracker.fleet_summary(),
    }


@router.post("/session/{session_id}/frame")
def update_frame(session_id: str, body: FrameRequest):
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "unknown or expired fleet session")
    session["last_seen"] = time.time()
    _sessions.move_to_end(session_id)
    tracker: FleetTracker = session["tracker"]

    dets = [(d.range_bin, d.doppler_bin, d.snr_db) for d in body.detections]
    tracker.update(dets)
    session["frame"] += 1

    return {
        "session_id": session_id,
        "frame": session["frame"],
        "tracks": _track_view(tracker),
        "fleet_summary": tracker.fleet_summary(),
    }


@router.delete("/session/{session_id}")
def delete_session(session_id: str):
    _sessions.pop(session_id, None)
    return {"deleted": session_id}


@router.get("/stress_test")
def stress_test_report():
    """Serves the last `python -m radarsense.fleet_tracking` run's real,
    reproduced results (100/500/1000-target cardinality error, completeness,
    association latency) -- not fabricated frontend numbers."""
    path = C.REPORTS_DIR / "fleet_tracking_results.json"
    if not path.exists():
        raise HTTPException(404, "no fleet_tracking_results.json yet -- run "
                                  "`python -m radarsense.fleet_tracking`")
    import json
    return json.loads(path.read_text())
