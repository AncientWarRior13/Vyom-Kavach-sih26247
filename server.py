"""Top-level app tying the four new pieces together:

scenario_engine  -> generates a scenario (ground truth kept server-side)
trainer_frontend -> trainee plays the scenario, logs detections/classifications/engagements
scoring_engine    -> grades the trainee log against ground truth
aar_dashboard     -> persists the score and renders trends across sessions

Run with: uvicorn server:app --reload --port 8090
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from scenario_engine import generate_scenario, list_scripted, load_scripted, spawn_live_contact
from scenario_engine.generator import ScenarioParams, sanitize_for_client
from scoring_engine import score_session
from aar_dashboard import (
    init_db, save_session, list_sessions, trainee_trend, unit_trend,
    get_or_assign_arm, experiment_results, weakness_profile,
    get_session_replay, error_taxonomy, missed_positions,
    certification_status, unit_certification_summary,
)
from ml_bridge import classify_contact, generate_swarm_paths
from ml_bridge.multi_sensor import read_sensor
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
    contact = next((c for c in scenario["contacts"] if c["contact_id"] == req.contact_id), None)
    if contact is None:
        raise HTTPException(404, "unknown contact_id")

    asset_xy = tuple(scenario.get("defended_asset_position", [0.0, 0.0, 0.0])[:2])
    return read_sensor(
        req.sensor, contact, req.t,
        scenario["params"].get("time_of_day", "day"), scenario["params"].get("terrain", "rural"),
        asset_xy, scenario.get("sensor_outage"),
    )


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

    # a split-swarm scenario (red-team tactic 7) has two independent swarm
    # groups, each with its own replay metadata — find whichever one owns
    # this contact.
    meta = None
    for key in ("swarm_sim_meta", "swarm_sim_meta_2"):
        candidate = scenario.get(key)
        if candidate is not None and req.contact_id in candidate["contact_to_uav"]:
            meta = candidate
            break
    if meta is None:
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
                 aar_ablation_arm=aar_arm, sequencing_arm=seq_arm,
                 scenario=scenario, trainee_log=req.trainee_log)
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


@app.get("/instructor")
def instructor_page():
    return FileResponse(BASE_DIR / "trainer_frontend" / "instructor.html")


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
