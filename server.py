"""Top-level app tying the four new pieces together:

scenario_engine  -> generates a scenario (ground truth kept server-side)
trainer_frontend -> trainee plays the scenario, logs detections/classifications/engagements
scoring_engine    -> grades the trainee log against ground truth
aar_dashboard     -> persists the score and renders trends across sessions

Run with: uvicorn server:app --reload --port 8090
"""
from __future__ import annotations

import json
from pathlib import Path
import os

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from scenario_engine import generate_scenario, list_scripted, load_scripted, spawn_live_contact, sign_pack, verify_pack
from scenario_engine.generator import ScenarioParams, sanitize_for_client
from scoring_engine import score_session
from aar_dashboard import (
    init_db, save_session, list_sessions, trainee_trend, unit_trend,
    get_or_assign_arm, experiment_results, weakness_profile,
    get_session_replay, error_taxonomy, missed_positions,
    certification_status, unit_certification_summary,
    verify_audit_chain, get_session_by_scenario,
)
from ml_bridge import (
    classify_contact, generate_swarm_paths, get_model_status, retrain_async,
    precompute_hints, get_precomputed_hint,
)
import threading
from ml_bridge.multi_sensor import read_sensor
from ml_bridge.training_utils import train_with_cutoff
from research import next_difficulty

AAR_ABLATION_ARMS = ["aar_shown", "aar_withheld"]
SEQUENCING_ARMS = ["blocked", "randomized"]

BASE_DIR = Path(__file__).parent

app = FastAPI(title="Vyom Kavach")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security hardening (external review finding #4): this app ships bound to
# 127.0.0.1 only by default (uvicorn's own default when --host isn't
# passed — verified, see README), which already blocks remote network
# access. Two further gaps remain even then, so this middleware checks two
# different headers for two different attacks:
#   - Host: the browser sets this to whatever address/hostname the request
#     is addressed TO. If someone runs with --host 0.0.0.0 and a remote
#     client reaches this server by IP/hostname, Host reflects that, not
#     "localhost" — so this catches unintended remote/LAN exposure.
#   - Origin: the browser sets this (on fetch/XHR, not on simple page
#     navigation) to whatever page's JS actually made the request — this is
#     what stops a malicious page open in another tab from silently
#     fetching this server cross-origin ("localhost CSRF"). Host alone does
#     NOT catch this, because Host always reflects the request's target,
#     never the origin of the page that sent it.
# Either check failing rejects the request, unless explicitly opted out via
# VYOMKAVACH_ALLOW_REMOTE=1 (e.g. for a deliberate multi-machine
# instructor/trainee LAN deployment — which still needs its own auth layer
# in front; this only restores the "only this machine can talk to it"
# property that --host 0.0.0.0 would otherwise remove silently).
_ALLOWED_HOSTS = {"localhost", "127.0.0.1", "[::1]", "testserver"}


def _origin_is_allowed(origin: str) -> bool:
    # origin looks like "http://localhost:8090" or "http://127.0.0.1:8090"
    host_part = origin.split("://", 1)[-1].split(":")[0]
    return host_part in _ALLOWED_HOSTS


@app.middleware("http")
async def _localhost_only_guard(request: Request, call_next):
    if os.environ.get("VYOMKAVACH_ALLOW_REMOTE") == "1":
        return await call_next(request)

    host = (request.headers.get("host") or "").split(":")[0]
    if host not in _ALLOWED_HOSTS:
        return JSONResponse(status_code=403, content={
            "detail": "Rejected: this server only accepts requests addressed to localhost. "
                      "Set VYOMKAVACH_ALLOW_REMOTE=1 if you intentionally need remote/LAN "
                      "access — but add your own authentication in front first.",
        })

    origin = request.headers.get("origin")
    if origin is not None and not _origin_is_allowed(origin):
        return JSONResponse(status_code=403, content={
            "detail": f"Rejected: cross-origin request from {origin} is not allowed against this local server.",
        })

    return await call_next(request)

# full ground-truth scenarios, kept server-side only, keyed by scenario_id
_ACTIVE_SCENARIOS: dict[str, dict] = {}


@app.on_event("startup")
def _startup():
    init_db()


@app.get("/api/research/assignment/{trainee_id}")
def research_assignment(trainee_id: str):
    """Returns this trainee's (stable, persistent) arm in each running
    experiment, plus the difficulty their sequencing arm dictates for their
    *next* attempt. See research/PROTOCOLS.md for the experimental designs.
    """
    aar_arm = get_or_assign_arm("aar_ablation", trainee_id, AAR_ABLATION_ARMS)
    seq_arm = get_or_assign_arm("sequencing", trainee_id, SEQUENCING_ARMS)
    attempt_number = len(list_sessions(trainee_id=trainee_id, limit=10_000)) + 1
    return {
        "trainee_id": trainee_id,
        "aar_ablation_arm": aar_arm,
        "sequencing_arm": seq_arm,
        "attempt_number": attempt_number,
        "recommended_difficulty": next_difficulty(seq_arm, attempt_number),
    }


@app.get("/api/research/results/{experiment}")
def research_results(experiment: str):
    """Raw per-session rows for offline statistical analysis (t-test/ANOVA
    across arms) — not rendered in-app, this is the researcher-facing export."""
    column = {"aar_ablation": "aar_ablation_arm", "sequencing": "sequencing_arm"}.get(experiment)
    if column is None:
        raise HTTPException(404, "unknown experiment; use 'aar_ablation' or 'sequencing'")
    return experiment_results(experiment, column)


class NewScenarioRequest(BaseModel):
    difficulty: int = 3
    time_of_day: str = "day"
    terrain: str = "rural"
    sensor_degradation: float = 0.0
    swarm_attack: bool = False
    duration_s: int = 300
    seed: int | None = None
    red_team_tactics: bool | None = None


@app.post("/api/scenario/new")
def new_scenario(req: NewScenarioRequest):
    params = ScenarioParams(
        difficulty=req.difficulty,
        time_of_day=req.time_of_day,
        terrain=req.terrain,
        sensor_degradation=req.sensor_degradation,
        swarm_attack=req.swarm_attack,
        seed=req.seed,
        red_team_tactics=req.red_team_tactics,
    )
    scenario = generate_scenario(params, duration_s=req.duration_s)
    _ACTIVE_SCENARIOS[scenario["scenario_id"]] = scenario
    _kick_precompute(scenario)
    return sanitize_for_client(scenario)


class AdaptiveScenarioRequest(BaseModel):
    trainee_id: str
    difficulty: int = 3
    time_of_day: str = "day"
    terrain: str = "rural"
    sensor_degradation: float = 0.0
    swarm_attack: bool = False
    duration_s: int = 300
    seed: int | None = None
    red_team_tactics: bool | None = None


@app.post("/api/scenario/adaptive")
def adaptive_scenario(req: AdaptiveScenarioRequest):
    """Roadmap feature 5 (adaptive curriculum): pulls this trainee's
    weakness profile from their AAR history and weights the next
    scenario's contact-type counts toward whatever they're weakest on,
    instead of pure random repetition. Falls back to the normal (1.0x
    every type) distribution for a trainee with no session history yet.
    """
    weights = weakness_profile(req.trainee_id)
    params = ScenarioParams(
        difficulty=req.difficulty,
        time_of_day=req.time_of_day,
        terrain=req.terrain,
        sensor_degradation=req.sensor_degradation,
        swarm_attack=req.swarm_attack,
        seed=req.seed,
        type_weights=weights,
        red_team_tactics=req.red_team_tactics,
    )
    scenario = generate_scenario(params, duration_s=req.duration_s)
    _ACTIVE_SCENARIOS[scenario["scenario_id"]] = scenario
    _kick_precompute(scenario)
    out = sanitize_for_client(scenario)
    out["applied_type_weights"] = weights
    return out


@app.get("/api/scenario/scripted")
def scripted_list():
    return list_scripted()


@app.post("/api/scenario/scripted/{name}")
def scripted_load(name: str):
    try:
        scenario = load_scripted(name)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    _ACTIVE_SCENARIOS[scenario["scenario_id"]] = scenario
    _kick_precompute(scenario)
    return sanitize_for_client(scenario)


@app.get("/api/scenario/scripted/{name}/pack")
def scripted_export_pack(name: str):
    """Roadmap feature 12 (offline mode and audit): a signed scenario pack
    — the scripted scenario's ground truth plus an HMAC signature, so it
    can be distributed to an air-gapped training site and verified as
    unmodified without any network/central-server check."""
    try:
        scenario = load_scripted(name)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    return sign_pack(scenario)


class VerifyPackRequest(BaseModel):
    scenario: dict
    signature: str


@app.post("/api/scenario/pack/verify")
def verify_scenario_pack(req: VerifyPackRequest):
    valid = verify_pack({"scenario": req.scenario, "signature": req.signature})
    return {"valid": valid}


@app.get("/api/audit/verify_chain")
def audit_verify_chain():
    """Roadmap feature 12: confirms the AAR session log's hash chain is
    intact — detects any past session row that was edited or tampered
    with after the fact."""
    return verify_audit_chain()


@app.get("/api/aar/verify_audit_chain")
def aar_verify_audit_chain_alias():
    """Alias for /api/audit/verify_chain, used by the tactical React AAR page."""
    return verify_audit_chain()


@app.get("/api/aar/missed_positions")
def aar_missed_positions_alias(trainee_id: str = "", unit_id: str = ""):
    """Alias for /api/aar/missed_heatmap, used by the tactical React AAR page."""
    return missed_positions(trainee_id or None, unit_id or None)


class ClassifyRequest(BaseModel):
    scenario_id: str
    contact_id: str
    t: float


@app.post("/api/contact/classify")
def classify(req: ClassifyRequest):
    """Runs RadarSense's real trained classifier on a contact's current
    kinematics and returns its sensor-side call (aircraft/anomaly/bird/drone)
    — this is an AI-assist hint the trainee can request, not the ground
    truth; it can be wrong, same as a real classifier would be.

    Roadmap step 7: answers from `scenario["_precomputed_hints"]` (computed
    in the background right when the scenario was created — see
    `_kick_precompute`) when available, falling back to a live compute
    otherwise (e.g. if asked before the background pass finished).
    """
    scenario = _ACTIVE_SCENARIOS.get(req.scenario_id)
    if scenario is None:
        raise HTTPException(404, "unknown or expired scenario_id")
    contact = next((c for c in scenario["contacts"] if c["contact_id"] == req.contact_id), None)
    if contact is None:
        raise HTTPException(404, "unknown contact_id")

    cached = get_precomputed_hint(scenario, req.contact_id)
    if cached is not None:
        return cached

    degradation = scenario["params"].get("effective_sensor_degradation", scenario["params"].get("sensor_degradation", 0.0))
    result = classify_contact(contact, req.t, degradation, siblings=scenario["contacts"])
    return {"label": result["label"], "probabilities": result["probabilities"], "source": result["source"]}


def _kick_precompute(scenario: dict) -> None:
    """Fires the roadmap step-7 background precomputation pass; never
    blocks the request that creates the scenario."""
    threading.Thread(target=precompute_hints, args=(scenario,), daemon=True).start()


@app.get("/api/model/status")
def model_status():
    """Roadmap step 5: current state of the serving classifier model —
    idle/training/trained/timeout_keep_current_model/failed_keep_current_model."""
    return get_model_status()


# Security fix (external review finding #3): /api/model/retrain used to run
# ANY script_path the caller supplied via subprocess — unauthenticated
# arbitrary code execution. Both the script to run and the model file it's
# allowed to produce are now restricted to resolve inside fixed allowlisted
# directories; anything that escapes them (including ../.. traversal) is
# rejected before it ever reaches subprocess.run.
_ALLOWED_TRAIN_SCRIPT_DIR = (BASE_DIR / "radar_classification" / "radarsense").resolve()
_ALLOWED_ONNX_OUTPUT_DIR = (BASE_DIR / "radar_classification" / "artifacts" / "onnx").resolve()


def _resolve_within(path_str: str, allowed_dir: Path, label: str) -> Path:
    candidate = (BASE_DIR / path_str).resolve() if not Path(path_str).is_absolute() else Path(path_str).resolve()
    try:
        candidate.relative_to(allowed_dir)
    except ValueError:
        raise HTTPException(403, f"{label} must resolve inside {allowed_dir} (got {candidate})")
    return candidate


class RetrainRequest(BaseModel):
    script_path: str
    new_onnx_path: str
    budget_s: float = 600.0


@app.post("/api/model/retrain")
def model_retrain(req: RetrainRequest):
    """Roadmap steps 5+6: kicks off retraining in the background — this
    endpoint returns immediately; poll /api/model/status for progress. The
    training script is run as its own OS subprocess with a hard wall-clock
    cutoff (`ml_bridge.training_utils.train_with_cutoff`), so a runaway job
    gets killed, not just abandoned. The classifier keeps serving the
    current model throughout, and only hot-swaps if the new one finishes in
    time and loads successfully.
    """
    script = _resolve_within(req.script_path, _ALLOWED_TRAIN_SCRIPT_DIR, "script_path")
    new_onnx = _resolve_within(req.new_onnx_path, _ALLOWED_ONNX_OUTPUT_DIR, "new_onnx_path")
    if not script.exists():
        raise HTTPException(404, f"no such training script: {script}")
    retrain_async(
        lambda: train_with_cutoff(str(script), budget_s=req.budget_s),
        str(new_onnx), budget_s=req.budget_s + 5.0,
    )
    return {"started": True, "script_path": str(script), "budget_s": req.budget_s}


class SensorReadRequest(BaseModel):
    scenario_id: str
    contact_id: str
    t: float
    sensor: str  # "eo_ir" | "rf" | "acoustic" (radar is /api/contact/classify)


@app.post("/api/contact/sensor_read")
def sensor_read(req: SensorReadRequest):
    """Roadmap feature 6 (multi-sensor fusion): EO/IR, RF and acoustic
    channels alongside the radar+ML classifier hint. One channel is jammed
    for the whole scenario (`scenario["sensor_outage"]`) — the trainee must
    cross-cue with whichever channels remain.
    """
    if req.sensor not in ("eo_ir", "rf", "acoustic"):
        raise HTTPException(400, "sensor must be one of eo_ir, rf, acoustic")
    scenario = _ACTIVE_SCENARIOS.get(req.scenario_id)
    if scenario is None:
        raise HTTPException(404, "unknown or expired scenario_id")
    contact = next((c for c in scenario["contacts"] if c.get("contact_id") == req.contact_id or c.get("id") == req.contact_id), None)
    if contact is None:
        raise HTTPException(404, "unknown contact_id")

    asset_xy = tuple(scenario.get("defended_asset_position", [0.0, 0.0, 0.0])[:2])
    return read_sensor(
        req.sensor, contact, req.t,
        scenario["params"].get("time_of_day", "day"), scenario["params"].get("terrain", "rural"),
        asset_xy, scenario.get("sensor_outage"),
    )


def _apply_hard_kill(scenario: dict, contact_id: str, t: float) -> dict:
    """Shared by the solo `/api/contact/engage` and the team-roles
    engagement-authority action: if the target is a live ares-driven swarm
    member, deterministically replays the swarm's simulation from t=0 up to
    `t` (identical to what's already been shown — see
    ml_bridge.swarm_kinematics' replay-and-diverge design), injects a real
    UAV_FAILURE at that instant, and lets ares's actual recovery/relay logic
    react from there. Returns {"reactive": bool, "contacts": [...]}.
    """
    scenario.setdefault("destroyed_contact_ids", [])
    if contact_id not in scenario["destroyed_contact_ids"]:
        scenario["destroyed_contact_ids"].append(contact_id)

    # a split-swarm scenario (red-team tactic 7) has two independent swarm
    # groups, each with its own replay metadata — find whichever one owns
    # this contact.
    meta = None
    for key in ("swarm_sim_meta", "swarm_sim_meta_2"):
        candidate = scenario.get(key)
        if candidate is not None and contact_id in candidate["contact_to_uav"]:
            meta = candidate
            break
    if meta is None:
        target_c = next((c for c in scenario["contacts"] if c.get("contact_id") == contact_id or c.get("id") == contact_id), None)
        if target_c is not None:
            target_c["state"] = "neutralized"
            target_c["engagement_action"] = "hard_kill"
            target_c["engagement_at"] = t
            target_c["despawn_t"] = min(target_c.get("despawn_t", float("inf")), t)
        return {"reactive": False, "contacts": []}

    uav_id = meta["contact_to_uav"][contact_id]
    local_time = max(0.0, t - meta["spawn_t"])
    if not any(uid == uav_id for uid, _ in meta["injected_failures"]):
        meta["injected_failures"].append([uav_id, local_time])

    trajectories = generate_swarm_paths(
        size=meta["size"], duration_s=meta["duration_s"], spawn_t=meta["spawn_t"],
        seed=meta["seed"], injected_failures=[tuple(f) for f in meta["injected_failures"]],
    )
    if not trajectories:
        raise HTTPException(500, "swarm replay failed")

    by_contact_id = {c["contact_id"]: c for c in scenario["contacts"]}
    for uid, pts in trajectories.items():
        cid = meta["uav_to_contact"].get(uid)
        if cid is None or not pts:
            continue
        c = by_contact_id[cid]
        c["path"] = [{"t": p["t"], "x": p["x"], "y": p["y"], "z": p["z"]} for p in pts]
        failed_at = next((p["t"] for p in pts if p.get("health") == "FAILED"), None)
        c["spawn_t"] = c["path"][0]["t"]
        if cid == contact_id or failed_at is not None:
            c["despawn_t"] = failed_at if failed_at is not None else min(c.get("despawn_t", float("inf")), t)
            c["state"] = "neutralized"
            c["engagement_action"] = "hard_kill"
            c["engagement_at"] = t
        else:
            c["despawn_t"] = c["path"][-1]["t"]

    updated_ids = set(meta["uav_to_contact"].values())
    sanitized = sanitize_for_client(scenario)
    updated_contacts = [c for c in sanitized["contacts"] if c["contact_id"] in updated_ids]
    return {"reactive": True, "contacts": updated_contacts}


class EngageRequest(BaseModel):
    scenario_id: str
    contact_id: str
    t: float
    action: str = "hard_kill"  # only a kinetic kill (hard_kill, or legacy "engage") has a live sim effect


@app.post("/api/contact/engage")
def engage_contact(req: EngageRequest):
    """Called the instant a solo trainee takes a kinetic (hard_kill) action
    on a contact — warn/soft_kill/monitor/ignore never reach this far (see
    the graduated-response ladder in scoring_engine/scorer.py). See
    `_apply_hard_kill` for what actually happens.
    """
    scenario = _ACTIVE_SCENARIOS.get(req.scenario_id)
    if scenario is None:
        raise HTTPException(404, "unknown or expired scenario_id")
    contact = next((c for c in scenario["contacts"] if c["contact_id"] == req.contact_id), None)
    if contact is None:
        raise HTTPException(404, "unknown contact_id")

    if req.action not in ("hard_kill", "engage"):
        return {"reactive": False, "contacts": []}

    contact["state"] = "neutralized"
    contact["engagement_action"] = "hard_kill"
    contact["engagement_at"] = req.t
    contact["despawn_t"] = min(contact.get("despawn_t", float("inf")), req.t)

    return _apply_hard_kill(scenario, req.contact_id, req.t)


# --- Roadmap feature 11: team roles with comms delay ---
# detector -> classifier -> engagement_authority: each role only learns
# what the previous role found out after `team_comms_delay_s` seconds,
# simulating separate operators relaying information instead of one person
# doing everything instantly. No new scoring axis needed — the delay just
# makes the existing detection-window/timeliness scoring harder, honestly.

TEAM_ROLES = ("detector", "classifier", "engagement_authority")


def _team_log(scenario: dict) -> dict:
    return scenario.setdefault("team_log", {})


def _team_entry(scenario: dict, contact_id: str) -> dict:
    log = _team_log(scenario)
    return log.setdefault(contact_id, {
        "detected_at": None, "classified_at": None, "classification": None,
        "ai_hint_requested": False, "ai_hint_label": None,
        "warn_at": None, "engagement_action": None, "engagement_at": None,
    })


class TeamActionRequest(BaseModel):
    scenario_id: str
    role: str
    contact_id: str
    t: float
    field: str   # detector: "detect" | classifier: "classify"|"ai_hint" | engagement_authority: warn/monitor/ignore/soft_kill/hard_kill
    value: str | None = None


@app.post("/api/team/action")
def team_action(req: TeamActionRequest):
    if req.role not in TEAM_ROLES:
        raise HTTPException(400, f"role must be one of {TEAM_ROLES}")
    scenario = _ACTIVE_SCENARIOS.get(req.scenario_id)
    if scenario is None:
        raise HTTPException(404, "unknown or expired scenario_id")
    contact = next((c for c in scenario["contacts"] if c["contact_id"] == req.contact_id), None)
    if contact is None:
        raise HTTPException(404, "unknown contact_id")

    delay = scenario.get("team_comms_delay_s", 5.0)
    entry = _team_entry(scenario, req.contact_id)

    if req.role == "detector":
        if req.field != "detect":
            raise HTTPException(400, "detector's only action is 'detect'")
        if entry["detected_at"] is None:
            entry["detected_at"] = req.t
        return {"ok": True, "entry": entry}

    if req.role == "classifier":
        if entry["detected_at"] is None or req.t < entry["detected_at"] + delay:
            wait = None if entry["detected_at"] is None else round(entry["detected_at"] + delay - req.t, 1)
            raise HTTPException(403, f"detection not yet relayed to classifier (wait {wait}s)" if wait else
                                       "contact not yet detected")
        if req.field == "classify":
            entry["classification"] = req.value
            entry["classified_at"] = req.t
        elif req.field == "ai_hint":
            degradation = scenario["params"].get("effective_sensor_degradation", 0.0)
            result = classify_contact(contact, req.t, degradation, siblings=scenario["contacts"])
            entry["ai_hint_requested"] = True
            entry["ai_hint_label"] = result["label"]
            return {"ok": True, "entry": entry, "hint": {"label": result["label"], "probabilities": result["probabilities"]}}
        else:
            raise HTTPException(400, "classifier's actions are 'classify' or 'ai_hint'")
        return {"ok": True, "entry": entry}

    # engagement_authority
    if entry["classified_at"] is None or req.t < entry["classified_at"] + delay:
        wait = None if entry["classified_at"] is None else round(entry["classified_at"] + delay - req.t, 1)
        raise HTTPException(403, f"classification not yet relayed to engagement authority (wait {wait}s)" if wait else
                                   "contact not yet classified")
    if req.field == "warn":
        entry["warn_at"] = req.t
        return {"ok": True, "entry": entry}
    if req.field not in ("monitor", "ignore", "soft_kill", "hard_kill"):
        raise HTTPException(400, "engagement_authority's actions are warn/monitor/ignore/soft_kill/hard_kill")
    entry["engagement_action"] = req.field
    entry["engagement_at"] = req.t
    if req.field == "hard_kill":
        contact["state"] = "neutralized"
        contact["engagement_action"] = "hard_kill"
        contact["engagement_at"] = req.t
        contact["despawn_t"] = min(contact.get("despawn_t", float("inf")), req.t)
        reactive = _apply_hard_kill(scenario, req.contact_id, req.t)
    else:
        reactive = {"reactive": False, "contacts": []}
    return {"ok": True, "entry": entry, **reactive}


@app.get("/api/team/state/{scenario_id}")
def team_state(scenario_id: str, role: str, t: float):
    """What this role can currently see: detector sees all live contacts'
    positions; classifier sees only contacts whose detection has been
    relayed to them; engagement_authority sees only contacts whose
    classification has been relayed to them."""
    if role not in TEAM_ROLES:
        raise HTTPException(400, f"role must be one of {TEAM_ROLES}")
    scenario = _ACTIVE_SCENARIOS.get(scenario_id)
    if scenario is None:
        raise HTTPException(404, "unknown or expired scenario_id")

    sanitized = sanitize_for_client(scenario)
    delay = scenario.get("team_comms_delay_s", 5.0)
    log = _team_log(scenario)

    visible = []
    for c in sanitized["contacts"]:
        entry = _team_entry(scenario, c["contact_id"])
        if role == "detector":
            visible.append({**c, "team": entry})
        elif role == "classifier":
            if entry.get("detected_at") is not None and t >= entry["detected_at"] + delay:
                visible.append({**c, "team": entry})
        else:
            if entry.get("classified_at") is not None and t >= entry["classified_at"] + delay:
                visible.append({**c, "team": entry})
    return {"role": role, "team_comms_delay_s": delay, "contacts": visible}


class SubmitTeamSessionRequest(BaseModel):
    scenario_id: str
    trainee_id: str
    unit_id: str
    difficulty: int
    participate_in_research: bool = False


@app.post("/api/session/submit_team")
def submit_team_session(req: SubmitTeamSessionRequest):
    """Assembles the standard scoring_engine trainee_log from whatever the
    three roles recorded in `scenario["team_log"]` over the session, then
    scores exactly like a solo session — the comms delay already made the
    timings harder; no separate scoring path is needed."""
    if req.scenario_id in _SCORED_SESSIONS_CACHE:
        return _SCORED_SESSIONS_CACHE[req.scenario_id]

    scenario = _ACTIVE_SCENARIOS.get(req.scenario_id)
    if scenario is None:
        saved = get_session_by_scenario(req.scenario_id)
        if saved:
            res = json.loads(saved["result_json"])
            res["session_id"] = saved["id"]
            _SCORED_SESSIONS_CACHE[req.scenario_id] = res
            return res
        raise HTTPException(404, "unknown or expired scenario_id. Please start a new scenario.")

    trainee_log = [{"contact_id": cid, **entry} for cid, entry in _team_log(scenario).items()]
    result = score_session(scenario, trainee_log)

    # Populate explicit score fields and operational bottleneck analysis
    det_s = result.get("detection_score", 0.0)
    cls_s = result.get("classification_score", 0.0)
    eng_s = result.get("engagement_score", 0.0)
    tml_s = result.get("timeliness_score", 0.0)

    lowest = min(det_s, cls_s, eng_s)
    if lowest == det_s and det_s < 70:
        bottleneck = f"Detection bottleneck: Operator missed or delayed radar acquisition on {result.get('contacts_missed', 0)} contact(s)."
    elif lowest == cls_s and cls_s < 70:
        bottleneck = f"Classification bottleneck: Multispectral signature recognition accuracy was low ({cls_s}%)."
    elif lowest == eng_s and eng_s < 70:
        bottleneck = f"Engagement bottleneck: Kinetic/soft-kill assignment delayed or conflicted with ROE."
    elif tml_s < 70:
        bottleneck = f"Latency bottleneck: Comms latency (+{scenario.get('team_comms_delay_s', 5.0)}s) degraded engagement timeliness."
    else:
        bottleneck = "Synchronized C2 performance: Coordinated detection, classification, and engagement across all stations."

    result["bottleneck"] = bottleneck
    result["score"] = result["overall_score"]
    result["team_score"] = result["overall_score"]

    aar_arm = seq_arm = None
    if req.participate_in_research:
        aar_arm = get_or_assign_arm("aar_ablation", req.trainee_id, AAR_ABLATION_ARMS)
        seq_arm = get_or_assign_arm("sequencing", req.trainee_id, SEQUENCING_ARMS)

    session_id = save_session(req.trainee_id, req.unit_id, req.difficulty, result,
                 aar_ablation_arm=aar_arm, sequencing_arm=seq_arm,
                 scenario=scenario, trainee_log=trainee_log)
    result["session_id"] = session_id
    _SCORED_SESSIONS_CACHE[req.scenario_id] = result
    del _ACTIVE_SCENARIOS[req.scenario_id]
    return result


class SubmitSessionRequest(BaseModel):
    scenario_id: str
    trainee_id: str
    unit_id: str
    difficulty: int
    trainee_log: list[dict]
    participate_in_research: bool = False


_SCORED_SESSIONS_CACHE: dict[str, dict] = {}


@app.post("/api/session/submit")
def submit_session(req: SubmitSessionRequest):
    if req.scenario_id in _SCORED_SESSIONS_CACHE:
        return _SCORED_SESSIONS_CACHE[req.scenario_id]

    scenario = _ACTIVE_SCENARIOS.get(req.scenario_id)
    if scenario is None:
        saved = get_session_by_scenario(req.scenario_id)
        if saved:
            res = json.loads(saved["result_json"])
            res["session_id"] = saved["id"]
            _SCORED_SESSIONS_CACHE[req.scenario_id] = res
            return res
        raise HTTPException(404, "unknown or expired scenario_id. Please start a new scenario.")

    result = score_session(scenario, req.trainee_log)
    result["score"] = result["overall_score"]

    aar_arm = seq_arm = None
    if req.participate_in_research:
        aar_arm = get_or_assign_arm("aar_ablation", req.trainee_id, AAR_ABLATION_ARMS)
        seq_arm = get_or_assign_arm("sequencing", req.trainee_id, SEQUENCING_ARMS)

    session_id = save_session(req.trainee_id, req.unit_id, req.difficulty, result,
                 aar_ablation_arm=aar_arm, sequencing_arm=seq_arm,
                 scenario=scenario, trainee_log=req.trainee_log)
    result["session_id"] = session_id
    _SCORED_SESSIONS_CACHE[req.scenario_id] = result
    del _ACTIVE_SCENARIOS[req.scenario_id]

    if aar_arm == "aar_withheld":
        # the AAR-ablation experiment's control arm: the score is recorded
        # (so trend/unit data still exists) but not shown back immediately —
        # that's the manipulation being tested, per research/PROTOCOLS.md.
        return {
            "scenario_id": result["scenario_id"],
            "session_id": session_id,
            "aar_withheld": True,
            "message": "Session recorded. Results will be reviewed in a later debrief, not shown immediately (you're in the AAR-timing study's control arm).",
        }
    return result


@app.get("/api/aar/sessions")
def aar_sessions(trainee_id: str = "", unit_id: str = ""):
    return list_sessions(trainee_id or None, unit_id or None)


@app.get("/api/aar/trend/trainee/{trainee_id}")
def aar_trainee_trend(trainee_id: str):
    return trainee_trend(trainee_id)


@app.get("/api/aar/session/{session_id}/replay")
def aar_session_replay(session_id: int):
    """Roadmap feature 9 (AAR replay with ghost track): full ground-truth
    scenario + the trainee's raw logged actions, for a timeline-scrubber UI
    to replay what actually happened contact-by-contact. Only sessions
    saved after this feature shipped carry replay data."""
    replay = get_session_replay(session_id)
    if replay is None:
        raise HTTPException(404, "unknown session_id")
    return replay


@app.get("/api/aar/error_taxonomy")
def aar_error_taxonomy(trainee_id: str = "", unit_id: str = ""):
    """Roadmap feature 9: counts of each scoring-flag type across recent
    matching sessions — which kinds of mistakes a trainee/unit actually
    makes, not just an aggregate score."""
    return error_taxonomy(trainee_id or None, unit_id or None)


@app.get("/api/aar/missed_heatmap")
def aar_missed_heatmap(trainee_id: str = "", unit_id: str = ""):
    """Roadmap feature 9: (x, y) of every missed contact across recent
    sessions, for a heatmap of which regions a trainee/unit tends to miss."""
    return missed_positions(trainee_id or None, unit_id or None)


@app.get("/api/aar/export.csv")
def aar_export_csv(trainee_id: str = "", unit_id: str = ""):
    """Roadmap feature 9: CSV export of session summaries. (PDF export is
    the print-friendly /aar/session/{id}/print page below — browser
    print-to-PDF — rather than a bundled PDF-generation dependency.)"""
    import csv
    import io
    sessions = list_sessions(trainee_id or None, unit_id or None, limit=10_000)
    buf = io.StringIO()
    fields = ["id", "created_at", "trainee_id", "unit_id", "difficulty", "overall_score",
              "detection_score", "classification_score", "engagement_score", "contacts_missed"]
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for s in sessions:
        writer.writerow(s)
    return Response(content=buf.getvalue(), media_type="text/csv",
                     headers={"Content-Disposition": "attachment; filename=sessions.csv"})


@app.get("/aar/session/{session_id}/print")
def aar_session_print(session_id: int):
    """A print-friendly single-session report — the practical 'export to
    PDF' path (browser print-to-PDF) without a bundled PDF-generation
    dependency."""
    replay = get_session_replay(session_id)
    if replay is None:
        raise HTTPException(404, "unknown session_id")
    result = replay["result"]
    rows = "".join(
        f"<tr><td>{c['contact_id']}</td><td>{c['true_type']}</td><td>{c['contact_score']}</td>"
        f"<td>{c.get('flag') or ''}</td></tr>"
        for c in result.get("per_contact", [])
    )
    html = f"""<!doctype html><html><head><meta charset="utf-8">
    <title>Session {session_id} report</title>
    <style>body{{font-family:system-ui,sans-serif;padding:24px}}table{{border-collapse:collapse;width:100%}}
    td,th{{border:1px solid #ccc;padding:4px 8px;text-align:left}}</style></head><body>
    <h1>Session {session_id} — {replay['trainee_id']} / {replay['unit_id']}</h1>
    <p>Created: {replay['created_at']} | Difficulty: {replay['difficulty']}</p>
    <p>Overall: {result['overall_score']} | Detection: {result['detection_score']} |
       Classification: {result['classification_score']} | Engagement: {result['engagement_score']} |
       Timeliness: {result.get('timeliness_score', '-')}</p>
    <table><thead><tr><th>Contact</th><th>True type</th><th>Score</th><th>Flag</th></tr></thead>
    <tbody>{rows}</tbody></table>
    <p style="margin-top:24px;color:#666">Use your browser's Print &rarr; Save as PDF for a PDF copy.</p>
    </body></html>"""
    return Response(content=html, media_type="text/html")


@app.get("/api/aar/certification/{trainee_id}")
def aar_certification(trainee_id: str, threshold: float = 80.0, window: int = 5):
    """Roadmap feature 10: pass/fail readiness grade."""
    return certification_status(trainee_id, threshold, window)


@app.get("/api/aar/certification/unit/{unit_id}")
def aar_unit_certification(unit_id: str, threshold: float = 80.0, window: int = 5):
    return unit_certification_summary(unit_id, threshold, window)


@app.get("/api/aar/trend/unit/{unit_id}")
def aar_unit_trend(unit_id: str):
    return unit_trend(unit_id)


# --- Roadmap feature 8: instructor console ---

@app.get("/api/instructor/active_scenarios")
def instructor_active_scenarios():
    """What an instructor can currently pick to inject into."""
    return [
        {"scenario_id": sid, "duration_s": s["duration_s"], "contact_count": len(s["contacts"]),
         "sensor_outage": s.get("sensor_outage"), "params": {k: v for k, v in s["params"].items() if k != "seed"}}
        for sid, s in _ACTIVE_SCENARIOS.items()
    ]


class InjectThreatRequest(BaseModel):
    scenario_id: str
    contact_type: str  # one of ContactType's values, e.g. "hostile_drone", "decoy", "bird"
    t: float


@app.post("/api/instructor/inject_threat")
def instructor_inject_threat(req: InjectThreatRequest):
    scenario = _ACTIVE_SCENARIOS.get(req.scenario_id)
    if scenario is None:
        raise HTTPException(404, "unknown or expired scenario_id")
    try:
        contact = spawn_live_contact(req.contact_type, req.t)
    except ValueError as e:
        raise HTTPException(400, str(e))
    scenario["contacts"].append(contact)
    sanitized_contact = sanitize_for_client({**scenario, "contacts": [contact]})["contacts"]
    return {"injected": True, "contact": sanitized_contact[0] if sanitized_contact else None}


class JamSensorRequest(BaseModel):
    scenario_id: str
    sensor: str | None  # None clears the outage


@app.post("/api/instructor/jam_sensor")
def instructor_jam_sensor(req: JamSensorRequest):
    scenario = _ACTIVE_SCENARIOS.get(req.scenario_id)
    if scenario is None:
        raise HTTPException(404, "unknown or expired scenario_id")
    if req.sensor is not None and req.sensor not in ("eo_ir", "rf", "acoustic"):
        raise HTTPException(400, "sensor must be eo_ir, rf, acoustic, or null to clear")
    scenario["sensor_outage"] = req.sensor
    return {"sensor_outage": scenario["sensor_outage"]}


@app.get("/api/scenario/{scenario_id}/refresh")
def scenario_refresh(scenario_id: str):
    """Lets the trainee's page pick up instructor-injected threats and
    sensor-outage changes without a full reload — poll this periodically
    during a session."""
    scenario = _ACTIVE_SCENARIOS.get(scenario_id)
    if scenario is None:
        raise HTTPException(404, "unknown or expired scenario_id")
    return sanitize_for_client(scenario)


APP_DIST = BASE_DIR / "app" / "dist"
APP_INDEX = APP_DIST / "index.html"


def serve_app_or_fallback(fallback_file: Path):
    if APP_INDEX.exists():
        return FileResponse(APP_INDEX)
    return FileResponse(fallback_file)


@app.get("/styleguide")
def styleguide_page():
    if APP_INDEX.exists():
        return FileResponse(APP_INDEX)
    raise HTTPException(404, "React app not built yet. Run 'npm run build' in app/")


@app.get("/swarm")
def swarm_page():
    if APP_INDEX.exists():
        return FileResponse(APP_INDEX)
    swarm_html = BASE_DIR / "swarm_simulation" / "dashboard" / "index.html"
    if swarm_html.exists():
        return FileResponse(swarm_html)
    raise HTTPException(404, "Swarm dashboard not found")


@app.get("/instructor")
def instructor_page():
    return serve_app_or_fallback(BASE_DIR / "trainer_frontend" / "instructor.html")


@app.get("/trainer")
def index():
    return serve_app_or_fallback(BASE_DIR / "trainer_frontend" / "index.html")


@app.get("/")
def landing_page():
    landing_index = BASE_DIR / "landing" / "dist" / "index.html"
    if landing_index.exists():
        return FileResponse(landing_index)
    if APP_INDEX.exists():
        return FileResponse(APP_INDEX)
    return FileResponse(BASE_DIR / "trainer_frontend" / "index.html")


@app.get("/aar")
def aar_page():
    return serve_app_or_fallback(BASE_DIR / "aar_dashboard" / "aar.html")


@app.get("/vr")
def vr_page():
    return serve_app_or_fallback(BASE_DIR / "trainer_frontend" / "vr.html")


@app.get("/team")
def team_page():
    return serve_app_or_fallback(BASE_DIR / "trainer_frontend" / "team.html")


app.mount("/static", StaticFiles(directory=BASE_DIR / "trainer_frontend"), name="static")
LANDING_DIST = BASE_DIR / "landing" / "dist"
if LANDING_DIST.exists():
    app.mount("/landing-static", StaticFiles(directory=LANDING_DIST, check_dir=False), name="landing-static")
if (APP_DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=APP_DIST / "assets", check_dir=False), name="app-assets")


@app.get("/{filename:path}")
def serve_dist_static(filename: str):
    target = APP_DIST / filename
    if target.exists() and target.is_file():
        return FileResponse(target)
    raise HTTPException(404, "File not found")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8090))
    host = "0.0.0.0" if os.environ.get("VYOMKAVACH_ALLOW_REMOTE") == "1" or os.environ.get("RENDER") else "127.0.0.1"
    uvicorn.run("server:app", host=host, port=port)


