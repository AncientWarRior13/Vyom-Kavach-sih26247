"""Bridges scenario_engine's "hostile swarm attack" contacts to the real
ares autonomy stack (`swarm_simulation/ares`), instead of straight-line
synthetic paths.

ares was built for a *friendly* resilient-survey swarm (allocator + relay +
recovery reacting to failures/degraded links while visiting PoIs). We reuse
it as-is to drive *hostile* swarm kinematics for the trainer: the "PoIs" are
points around the defended asset the swarm is converging on, and we inject a
scripted UAV_FAILURE partway through (representing an automated point-defense
kill) so the remaining swarm members visibly reallocate/reroute using ares's
real recovery planner — not a scripted animation.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

SWARM_SIM_ROOT = Path(__file__).resolve().parent.parent / "swarm_simulation"
if str(SWARM_SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SWARM_SIM_ROOT))


def _build_scenario(size: int, duration_s: float, seed: int, inject_failure: bool):
    from ares.scenario import Scenario, PoISpec, ScenarioEvent

    target = [random.uniform(300, 600), random.uniform(300, 600), 0.0]
    launch = [target[0] - random.uniform(1500, 2500), target[1] - random.uniform(1500, 2500), 0.0]

    pois = [
        PoISpec(
            id=f"asset_{i}",
            position=[target[0] + random.uniform(-60, 60), target[1] + random.uniform(-60, 60), random.uniform(80, 250)],
            priority=1.0,
            required_data=["imagery"],
            deadline=duration_s,
            risk=0.5,
        )
        for i in range(max(1, size // 3))
    ]

    events = []
    if inject_failure and size > 1:
        events.append(ScenarioEvent(time=duration_s * random.uniform(0.3, 0.5), type="UAV_FAILURE",
                                     data={"uav_id": f"uav_{random.randint(0, size - 1):02d}"}))

    return Scenario(
        name="hostile_swarm_inject",
        seed=seed,
        duration_seconds=duration_s,
        dt_seconds=1.0,
        geofence_min=[min(launch[0], target[0]) - 200, min(launch[1], target[1]) - 200, 0],
        geofence_max=[max(launch[0], target[0]) + 200, max(launch[1], target[1]) + 200, 400],
        gcs_position=launch,
        uav_count=size,
        battery_percent=100.0,
        max_speed_mps=random.uniform(10, 18),
        comm_range_m=180.0,
        battery_drain_rate=0.03,
        battery_drain_rate_relay_bonus=0.005,
        battery_reserve_pct=10.0,
        home_position=launch,
        pois=pois,
        events=events,
    )


def generate_swarm_paths(size: int, duration_s: float, spawn_t: float, seed: int | None = None) -> dict:
    """Runs the real ares Simulator and returns {uav_id: [{t,x,y,z}, ...]}
    with timestamps shifted so motion starts at `spawn_t`. Falls back to
    returning None if ares can't be imported/run in this environment (e.g.
    PyYAML missing) so the caller can fall back to straight-line paths.
    """
    try:
        from ares.sim import Simulator
    except Exception:
        return None

    seed = seed if seed is not None else random.randint(0, 1_000_000)
    scenario = _build_scenario(size, duration_s, seed, inject_failure=True)

    try:
        sim = Simulator(scenario)
        trajectories: dict[str, list[dict]] = {uid: [] for uid in sim.state.uavs}
        steps = int(duration_s / scenario.dt_seconds)
        for _ in range(steps):
            for uid, uav in sim.state.uavs.items():
                trajectories[uid].append({
                    "t": spawn_t + sim.state.time,
                    "x": uav.position[0], "y": uav.position[1], "z": uav.position[2],
                    "health": uav.health.value,
                })
            sim.step(scenario.dt_seconds)
        return trajectories
    except Exception:
        return None
