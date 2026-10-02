"""Top-level app tying the four new pieces together:

scenario_engine  -> generates a scenario (ground truth kept server-side)
trainer_frontend -> trainee plays the scenario, logs detections/classifications/engagements
scoring_engine    -> grades the trainee log against ground truth
aar_dashboard     -> persists the score and renders trends across sessions

Run with: uvicorn server:app --reload --port 8090
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from scenario_engine import generate_scenario, list_scripted, load_scripted
from scenario_engine.generator import ScenarioParams, sanitize_for_client
from scoring_engine import score_session
from aar_dashboard import (
    init_db, save_session, list_sessions, trainee_trend, unit_trend,
    get_or_assign_arm, experiment_results, weakness_profile,
)
from ml_bridge import classify_contact, generate_swarm_paths
from research import next_difficulty

AAR_ABLATION_ARMS = ["aar_shown", "aar_withheld"]
SEQUENCING_ARMS = ["blocked", "randomized"]

BASE_DIR = Path(__file__).parent

app = FastAPI(title="Drone Threat Trainer")

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


@app.post("/api/scenario/new")
def new_scenario(req: NewScenarioRequest):
    params = ScenarioParams(
        difficulty=req.difficulty,
        time_of_day=req.time_of_day,
        terrain=req.terrain,
        sensor_degradation=req.sensor_degradation,
        swarm_attack=req.swarm_attack,
        seed=req.seed,
    )
    scenario = generate_scenario(params, duration_s=req.duration_s)
    _ACTIVE_SCENARIOS[scenario["scenario_id"]] = scenario
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
    )
    scenario = generate_scenario(params, duration_s=req.duration_s)
    _ACTIVE_SCENARIOS[scenario["scenario_id"]] = scenario
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
    return sanitize_for_client(scenario)


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
    """
    scenario = _ACTIVE_SCENARIOS.get(req.scenario_id)
    if scenario is None:
        raise HTTPException(404, "unknown or expired scenario_id")
    contact = next((c for c in scenario["contacts"] if c["contact_id"] == req.contact_id), None)
    if contact is None:
        raise HTTPException(404, "unknown contact_id")
    degradation = scenario["params"].get("effective_sensor_degradation", scenario["params"].get("sensor_degradation", 0.0))
    result = classify_contact(contact, req.t, degradation, siblings=scenario["contacts"])
    return {"label": result["label"], "probabilities": result["probabilities"], "source": result["source"]}


class EngageRequest(BaseModel):
    scenario_id: str
    contact_id: str
    t: float
    action: str = "hard_kill"  # only a kinetic kill (hard_kill, or legacy "engage") has a live sim effect


@app.post("/api/contact/engage")
def engage_contact(req: EngageRequest):
    """Called the instant a trainee takes a kinetic (hard_kill) action on a
    contact (not just logged for scoring at the end) — warn/soft_kill/
    monitor/ignore never reach this far (see the graduated-response ladder
    in scoring_engine/scorer.py; only a kinetic kill can plausibly destroy a
    swarm member). If the target is a live ares-driven swarm member, this
    deterministically replays the swarm's simulation from t=0 up to
    `req.t` (identical to what the trainee already saw — see
    ml_bridge.swarm_kinematics' replay-and-diverge design), injects a real
    UAV_FAILURE at that instant, and lets ares's actual recovery/relay logic
    react from there — the remaining swarm members' future paths in the
    scenario are updated in place and returned so the client can refresh
    what it's rendering. Non-swarm contacts (and swarm contacts in a
    scenario where ares wasn't available) are a no-op here; their
    engagement is still scored normally at session submit.
    """
    scenario = _ACTIVE_SCENARIOS.get(req.scenario_id)
    if scenario is None:
        raise HTTPException(404, "unknown or expired scenario_id")
    contact = next((c for c in scenario["contacts"] if c["contact_id"] == req.contact_id), None)
    if contact is None:
        raise HTTPException(404, "unknown contact_id")

    if req.action not in ("hard_kill", "engage"):
        return {"reactive": False, "contacts": []}

    meta = scenario.get("swarm_sim_meta")
    if meta is None or req.contact_id not in meta["contact_to_uav"]:
        return {"reactive": False, "contacts": []}

    uav_id = meta["contact_to_uav"][req.contact_id]
    local_time = max(0.0, req.t - meta["spawn_t"])
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
        c["despawn_t"] = failed_at if failed_at is not None else c["path"][-1]["t"]

    updated_ids = set(meta["uav_to_contact"].values())
    sanitized = sanitize_for_client(scenario)
    updated_contacts = [c for c in sanitized["contacts"] if c["contact_id"] in updated_ids]
    return {"reactive": True, "contacts": updated_contacts}


class SubmitSessionRequest(BaseModel):
    scenario_id: str
    trainee_id: str
    unit_id: str
    difficulty: int
    trainee_log: list[dict]
    participate_in_research: bool = False


@app.post("/api/session/submit")
def submit_session(req: SubmitSessionRequest):
    scenario = _ACTIVE_SCENARIOS.get(req.scenario_id)
    if scenario is None:
        raise HTTPException(404, "unknown or expired scenario_id")
    result = score_session(scenario, req.trainee_log)

    aar_arm = seq_arm = None
    if req.participate_in_research:
        aar_arm = get_or_assign_arm("aar_ablation", req.trainee_id, AAR_ABLATION_ARMS)
        seq_arm = get_or_assign_arm("sequencing", req.trainee_id, SEQUENCING_ARMS)

    save_session(req.trainee_id, req.unit_id, req.difficulty, result,
                 aar_ablation_arm=aar_arm, sequencing_arm=seq_arm)
    del _ACTIVE_SCENARIOS[req.scenario_id]

    if aar_arm == "aar_withheld":
        # the AAR-ablation experiment's control arm: the score is recorded
        # (so trend/unit data still exists) but not shown back immediately —
        # that's the manipulation being tested, per research/PROTOCOLS.md.
        return {
            "scenario_id": result["scenario_id"],
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


@app.get("/api/aar/trend/unit/{unit_id}")
def aar_unit_trend(unit_id: str):
    return unit_trend(unit_id)


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "trainer_frontend" / "index.html")


@app.get("/aar")
def aar_page():
    return FileResponse(BASE_DIR / "aar_dashboard" / "aar.html")


@app.get("/vr")
def vr_page():
    return FileResponse(BASE_DIR / "trainer_frontend" / "vr.html")


app.mount("/static", StaticFiles(directory=BASE_DIR / "trainer_frontend"), name="static")
