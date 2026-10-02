"""ARES-X Live Autonomy Simulation Server
Exposes a high-performance FastAPI + WebSocket server streaming real-time SwarmState
at 10Hz to the Next.js Mission Command Center frontend, and handling bidirectional
operator commands (playback control, scenario selection, disturbance injection).
"""
from __future__ import annotations

import asyncio
import glob
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ares.scenario import Scenario
from ares.sim import Simulator
from ares.state import Health, Role, PoI, PoIStatus, dist
from ares.bridge import to_frontend_state

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ares.server")

SCENARIOS_DIR = os.path.join(REPO_ROOT, "config", "scenarios")
RESULTS_RAW_DIR = os.path.join(REPO_ROOT, "results", "raw")


class SimulationManager:
    def __init__(self):
        self.scenario_name = "baseline"
        self.scenario_path = os.path.join(SCENARIOS_DIR, "baseline.yaml")
        self.scenario: Optional[Scenario] = None
        self.sim: Optional[Simulator] = None
        self.is_paused: bool = False
        self.session_id: str = "live_session"
        self.active_websockets: Set[WebSocket] = set()
        self.loop_task: Optional[asyncio.Task] = None
        self.recorded_frames: list[dict] = []
        self.max_recorded_frames: int = 3000
        self.dt_step: float = 0.2

    def load_scenario(self, scenario_name: str) -> bool:
        filename = scenario_name if scenario_name.endswith(".yaml") else f"{scenario_name}.yaml"
        path = os.path.join(SCENARIOS_DIR, filename)
        if not os.path.exists(path):
            # Try finding by scenario name inside files
            for f in glob.glob(os.path.join(SCENARIOS_DIR, "*.yaml")):
                try:
                    s = Scenario.load(f)
                    if s.name == scenario_name or os.path.basename(f) == filename:
                        path = f
                        break
                except Exception:
                    continue

        if not os.path.exists(path):
            return False

        self.scenario_name = os.path.splitext(os.path.basename(path))[0]
        self.scenario_path = path
        self.scenario = Scenario.load(path)
        self.sim = Simulator(self.scenario)
        self.is_paused = False
        self.recorded_frames.clear()
        self.session_id = f"live_{self.scenario_name}"
        logger.info(f"Loaded scenario '{self.scenario_name}' from {path}")
        return True

    def reset(self):
        if self.scenario_path and os.path.exists(self.scenario_path):
            self.scenario = Scenario.load(self.scenario_path)
            self.sim = Simulator(self.scenario)
            self.is_paused = False
            self.recorded_frames.clear()
            logger.info("Simulation reset to t=0.")

    def inject_uav_failure(self, target_uav: str, reason: str = "Operator injected failure"):
        if not self.sim:
            return
        state = self.sim.state
        uav = state.uavs.get(target_uav)
        if uav and uav.alive:
            uav.health = Health.FAILED
            uav.role = Role.FAILED
            uav.current_task = None
            uav.target = None
            self.sim.log.emit(
                state.time,
                "UAV_FAILURE_DETECTED",
                source=uav.id,
                trigger="manual_operator_injection",
                old_state=Health.OK.value,
                new_state=Health.FAILED.value,
                reason=reason,
            )
            # Recompute connectivity & detect fault
            self.sim.fault_detector.detect(state, self.sim.log)
            logger.info(f"Injected failure on {target_uav}")

    def inject_packet_loss(self, center: list[float] | None = None, radius: float = 400.0, loss_rate: float = 0.65):
        if not self.sim:
            return
        state = self.sim.state
        expire_time = state.time + 60.0
        affected = 0
        for (a, b), lk in list(state.links.items()):
            u_a = state.uavs.get(a)
            u_b = state.uavs.get(b)
            if center and u_a and u_b:
                d_a = dist(u_a.position, center)
                d_b = dist(u_b.position, center)
                if min(d_a, d_b) > radius:
                    continue
            key = tuple(sorted((a, b)))
            state.degraded_links[key] = {
                "packet_loss": loss_rate,
                "expire_time": expire_time,
            }
            affected += 1
            self.sim.log.emit(
                state.time,
                "LINK_DEGRADED",
                source=f"{a},{b}",
                trigger="manual_operator_injection",
                old_state="nominal",
                new_state="degraded",
                reason=f"injected RF loss {int(loss_rate*100)}%",
            )
        logger.info(f"Injected packet loss on {affected} links (rate={loss_rate})")

    def inject_emergency_poi(self, pid: str | None = None, name: str | None = None, position: list[float] | None = None, priority: float = 1.0):
        if not self.sim:
            return
        state = self.sim.state
        if not pid:
            pid = f"POI_SOS_{len(state.pois)+1:02d}"
        if not position:
            position = [1550.0, 1250.0, 30.0]

        state.pois[pid] = PoI(
            id=pid,
            position=position,
            priority=priority,
            required_data=["imagery", "thermal"],
            deadline=state.time + 180.0,
            risk=0.4,
            status=PoIStatus.UNVISITED,
        )
        self.sim.log.emit(
            state.time,
            "PRIORITY_ESCALATED",
            source=pid,
            trigger="manual_operator_injection",
            old_state="none",
            new_state="UNVISITED",
            reason=f"Emergency SOS PoI injected with priority={priority}",
        )
        logger.info(f"Injected emergency PoI {pid} at {position} (priority={priority})")

    def get_current_state_payload(self) -> dict:
        if not self.sim:
            return {}
        return to_frontend_state(
            sim=self.sim,
            scenario_name=self.scenario_name,
            session_id=self.session_id,
            is_paused=self.is_paused,
        )


sim_mgr = SimulationManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load default baseline scenario and start 10Hz loop
    sim_mgr.load_scenario("baseline")
    sim_mgr.loop_task = asyncio.create_task(simulation_tick_loop())
    logger.info("Simulation server started.")
    yield
    # Shutdown
    if sim_mgr.loop_task:
        sim_mgr.loop_task.cancel()
    logger.info("Simulation server stopped.")


app = FastAPI(title="ARES-X Simulation Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def simulation_tick_loop():
    while True:
        try:
            await asyncio.sleep(0.1)  # 10Hz tick rate
            if not sim_mgr.sim or sim_mgr.is_paused:
                continue

            # Advance simulator
            sim_mgr.sim.step(dt=sim_mgr.dt_step)

            # Auto-loop or pause when duration reached
            if sim_mgr.scenario and sim_mgr.sim.state.time >= sim_mgr.scenario.duration_seconds:
                sim_mgr.is_paused = True

            payload = sim_mgr.get_current_state_payload()

            # Save frame for replay buffer
            sim_mgr.recorded_frames.append(payload)
            if len(sim_mgr.recorded_frames) > sim_mgr.max_recorded_frames:
                sim_mgr.recorded_frames.pop(0)

            # Broadcast to all connected WebSockets
            if sim_mgr.active_websockets:
                message = json.dumps(payload)
                dead = set()
                for ws in sim_mgr.active_websockets:
                    try:
                        await ws.send_text(message)
                    except Exception:
                        dead.add(ws)
                sim_mgr.active_websockets.difference_update(dead)

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in simulation loop: {e}", exc_info=True)


# ==============================================================================
# REST API Endpoints
# ==============================================================================

@app.get("/api/health")
async def health_check():
    return {
        "status": "online",
        "scenario": sim_mgr.scenario_name,
        "is_paused": sim_mgr.is_paused,
        "clients_connected": len(sim_mgr.active_websockets),
        "sim_time": sim_mgr.sim.state.time if sim_mgr.sim else 0.0,
    }


@app.get("/api/scenarios")
async def list_scenarios():
    scenarios = []
    for filepath in sorted(glob.glob(os.path.join(SCENARIOS_DIR, "*.yaml"))):
        basename = os.path.splitext(os.path.basename(filepath))[0]
        try:
            sc = Scenario.load(filepath)
            scenarios.append({
                "id": basename,
                "name": sc.name,
                "file": os.path.basename(filepath),
                "uav_count": sc.uav_count,
                "num_pois": len(sc.pois),
                "duration_seconds": sc.duration_seconds,
            })
        except Exception:
            scenarios.append({
                "id": basename,
                "name": basename,
                "file": os.path.basename(filepath),
            })
    return {"scenarios": scenarios}


class ScenarioSelectRequest(BaseModel):
    scenario: str


@app.post("/api/scenarios/select")
async def select_scenario(req: ScenarioSelectRequest):
    success = sim_mgr.load_scenario(req.scenario)
    if not success:
        raise HTTPException(status_code=404, detail=f"Scenario '{req.scenario}' not found")
    # Broadcast new state immediately
    payload = sim_mgr.get_current_state_payload()
    if sim_mgr.active_websockets:
        message = json.dumps(payload)
        for ws in list(sim_mgr.active_websockets):
            try:
                await ws.send_text(message)
            except Exception:
                pass
    return {"status": "ok", "scenario": sim_mgr.scenario_name}


class PlaybackRequest(BaseModel):
    action: str  # "PAUSE", "RESUME", "RESET"


@app.post("/api/sim/playback")
async def control_playback(req: PlaybackRequest):
    act = req.action.upper()
    if act == "PAUSE":
        sim_mgr.is_paused = True
    elif act == "RESUME":
        sim_mgr.is_paused = False
    elif act == "RESET":
        sim_mgr.reset()
    payload = sim_mgr.get_current_state_payload()
    if sim_mgr.active_websockets:
        message = json.dumps(payload)
        for ws in list(sim_mgr.active_websockets):
            try:
                await ws.send_text(message)
            except Exception:
                pass
    return {"status": "ok", "is_paused": sim_mgr.is_paused}


@app.get("/api/runs")
async def list_runs():
    runs = []
    # 1. Live session
    if sim_mgr.recorded_frames:
        first_t = sim_mgr.recorded_frames[0]["timestamp"]
        last_t = sim_mgr.recorded_frames[-1]["timestamp"]
        runs.append({
            "session_id": "live_current_run",
            "frames": len(sim_mgr.recorded_frames),
            "first_ts": first_t,
            "last_ts": last_t,
            "size_kb": round(len(json.dumps(sim_mgr.recorded_frames)) / 1024, 1),
        })

    # 2. Known disk scenarios in config/scenarios/
    for filepath in sorted(glob.glob(os.path.join(SCENARIOS_DIR, "*.yaml"))):
        basename = os.path.splitext(os.path.basename(filepath))[0]
        try:
            sc = Scenario.load(filepath)
            runs.append({
                "session_id": sc.name,
                "frames": int(sc.duration_seconds / 0.5),
                "first_ts": 0.0,
                "last_ts": float(sc.duration_seconds),
                "size_kb": 128,
            })
        except Exception:
            pass

    return {"runs": runs}


@app.get("/api/runs/{session_id}")
async def get_run_frames(session_id: str, stride: int = 10, max_frames: int = 600):
    if session_id == "live_current_run":
        selected = sim_mgr.recorded_frames[::max(1, stride)]
        return {"session_id": session_id, "frames": selected[:max_frames]}

    # Find matching scenario
    scenario_to_run = None
    for filepath in glob.glob(os.path.join(SCENARIOS_DIR, "*.yaml")):
        try:
            sc = Scenario.load(filepath)
            if sc.name == session_id or os.path.splitext(os.path.basename(filepath))[0] == session_id:
                scenario_to_run = sc
                break
        except Exception:
            continue

    if not scenario_to_run:
        # Check if baseline matches
        scenario_to_run = Scenario.load(os.path.join(SCENARIOS_DIR, "baseline.yaml"))

    # Generate sampled frames using pure Simulator
    temp_sim = Simulator(scenario_to_run)
    dt = 0.5
    total_steps = int(scenario_to_run.duration_seconds / dt)
    frames = []

    for step_idx in range(total_steps):
        temp_sim.step(dt)
        if step_idx % max(1, stride) == 0:
            frame = to_frontend_state(temp_sim, scenario_name=scenario_to_run.name, session_id=session_id)
            frames.append(frame)
            if len(frames) >= max_frames:
                break

    return {"session_id": session_id, "frames": frames}


# ==============================================================================
# WebSocket Endpoint for Live Frontend Telemetry & Control
# ==============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    sim_mgr.active_websockets.add(websocket)
    logger.info(f"Client connected. Active clients: {len(sim_mgr.active_websockets)}")

    # Send initial state immediately
    initial_payload = sim_mgr.get_current_state_payload()
    if initial_payload:
        await websocket.send_text(json.dumps(initial_payload))

    try:
        while True:
            text = await websocket.receive_text()
            try:
                msg = json.loads(text)
                action = msg.get("action")
                logger.info(f"WebSocket command received: {action} (data={msg})")

                if action == "PAUSE":
                    sim_mgr.is_paused = True
                elif action == "RESUME":
                    sim_mgr.is_paused = False
                elif action == "RESET":
                    sim_mgr.reset()
                elif action == "SELECT_SCENARIO":
                    sc_name = msg.get("scenario")
                    if sc_name:
                        sim_mgr.load_scenario(sc_name)
                elif action == "INJECT":
                    inject_type = msg.get("type")
                    target = msg.get("target")
                    params = msg.get("params", {})

                    if inject_type == "UAV_FAILURE":
                        sim_mgr.inject_uav_failure(target or "uav_03", reason=params.get("reason", "Motor failure"))
                    elif inject_type == "PACKET_LOSS":
                        center = params.get("center", [1100, 900, 40])
                        radius = params.get("radius", 400.0)
                        loss_rate = params.get("loss_rate", 0.65)
                        sim_mgr.inject_packet_loss(center=center, radius=radius, loss_rate=loss_rate)
                    elif inject_type == "EMERGENCY_POI":
                        sim_mgr.inject_emergency_poi(
                            pid=params.get("id"),
                            name=params.get("name"),
                            position=params.get("position", [1550, 1250, 30]),
                            priority=params.get("priority", 1.0),
                        )

                # Immediately send updated state following command
                await websocket.send_text(json.dumps(sim_mgr.get_current_state_payload()))

            except json.JSONDecodeError:
                pass

    except WebSocketDisconnect:
        sim_mgr.active_websockets.discard(websocket)
        logger.info(f"Client disconnected. Remaining: {len(sim_mgr.active_websockets)}")
    except Exception as e:
        sim_mgr.active_websockets.discard(websocket)
        logger.error(f"WebSocket error: {e}")
