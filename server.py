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

from scenario_engine import generate_scenario
from scenario_engine.generator import ScenarioParams, sanitize_for_client
from scoring_engine import score_session
from aar_dashboard import init_db, save_session, list_sessions, trainee_trend, unit_trend

BASE_DIR = Path(__file__).parent

app = FastAPI(title="Drone Threat Trainer")

# full ground-truth scenarios, kept server-side only, keyed by scenario_id
_ACTIVE_SCENARIOS: dict[str, dict] = {}


@app.on_event("startup")
def _startup():
    init_db()


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


class SubmitSessionRequest(BaseModel):
    scenario_id: str
    trainee_id: str
    unit_id: str
    difficulty: int
    trainee_log: list[dict]


@app.post("/api/session/submit")
def submit_session(req: SubmitSessionRequest):
    scenario = _ACTIVE_SCENARIOS.get(req.scenario_id)
    if scenario is None:
        raise HTTPException(404, "unknown or expired scenario_id")
    result = score_session(scenario, req.trainee_log)
    save_session(req.trainee_id, req.unit_id, req.difficulty, result)
    del _ACTIVE_SCENARIOS[req.scenario_id]
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


app.mount("/static", StaticFiles(directory=BASE_DIR / "trainer_frontend"), name="static")
