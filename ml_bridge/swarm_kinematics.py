"""Bridges scenario_engine's "hostile swarm attack" contacts to the real
ares autonomy stack (`swarm_simulation/ares`), instead of straight-line
synthetic paths.

ares was built for a *friendly* resilient-survey swarm (allocator + relay +
recovery reacting to failures/degraded links while visiting PoIs). We reuse
it as-is to drive *hostile* swarm kinematics for the trainer: the "PoIs" are
points around the defended asset the swarm is converging on.

No failure is scripted anymore. A swarm member only goes down when the
trainee actually engages it (see server.py's `/api/contact/engage`), at
which point `regenerate_swarm_paths` deterministically replays the exact
same simulation from t=0 up to the engagement moment (same `seed` -> same
scenario construction -> same path, so what the trainee already saw on
their scope doesn't retroactively change), injects the failure at that
instant, and continues the real ares recovery/relay logic forward from
there — so the reallocation genuinely reacts to when and which member the
trainee engaged, not to a random scripted timer.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

SWARM_SIM_ROOT = Path(__file__).resolve().parent.parent / "swarm_simulation"
if str(SWARM_SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SWARM_SIM_ROOT))


def _build_scenario(size: int, duration_s: float, seed: int, rng: random.Random):
    """All randomness here comes from `rng` (seeded from `seed`), not the
    global `random` module — this is what makes replay-and-diverge for live
    engagement actually deterministic. (Previously this used the global
    `random` module despite taking a `seed` parameter, so a `seed` didn't
    actually make scenario construction reproducible — fixed here.)
    """
    from ares.scenario import Scenario, PoISpec, ScenarioEvent

    target = [rng.uniform(300, 600), rng.uniform(300, 600), 0.0]
    launch = [target[0] - rng.uniform(1500, 2500), target[1] - rng.uniform(1500, 2500), 0.0]

    pois = [
        PoISpec(
            id=f"asset_{i}",
            position=[target[0] + rng.uniform(-60, 60), target[1] + rng.uniform(-60, 60), rng.uniform(80, 250)],
            priority=1.0,
            required_data=["imagery"],
            deadline=duration_s,
            risk=0.5,
        )
        for i in range(max(1, size // 3))
    ]

    return Scenario(
        name="hostile_swarm_reactive",
        seed=seed,
        duration_seconds=duration_s,
        dt_seconds=1.0,
        geofence_min=[min(launch[0], target[0]) - 200, min(launch[1], target[1]) - 200, 0],
        geofence_max=[max(launch[0], target[0]) + 200, max(launch[1], target[1]) + 200, 400],
        gcs_position=launch,
        uav_count=size,
        battery_percent=100.0,
        max_speed_mps=rng.uniform(10, 18),
        comm_range_m=180.0,
        battery_drain_rate=0.03,
        battery_drain_rate_relay_bonus=0.005,
        battery_reserve_pct=10.0,
        home_position=launch,
        pois=pois,
        events=[],
    ), ScenarioEvent


def pick_swarm_seed(seed: int | None) -> int:
    """Resolve the seed used for a swarm, once, at scenario-generation time
    — callers must persist and reuse this exact value for any later
    reactive replay (see server.py's swarm_sim_meta), or replay won't
    reproduce what the trainee already saw.
    """
    return seed if seed is not None else random.randint(0, 1_000_000)


def generate_swarm_paths(size: int, duration_s: float, spawn_t: float, seed: int,
                          injected_failures: list[tuple[str, float]] | None = None) -> dict | None:
    """Runs the real ares Simulator deterministically from `seed` and
    returns {uav_id: [{t,x,y,z,health}, ...]} with timestamps shifted by
    `spawn_t`. `injected_failures`: list of (uav_id, local_time_seconds)
    — local_time is relative to simulation start (i.e. t - spawn_t), applied
    as real `UAV_FAILURE` scenario events so ares's own recovery/relay logic
    reacts to them. Returns None if ares can't be imported/run here, so the
    caller can fall back to straight-line paths.
    """
    try:
        from ares.sim import Simulator
    except Exception:
        return None

    try:
        rng = random.Random(seed)
        scenario, ScenarioEvent = _build_scenario(size, duration_s, seed, rng)
        scenario.events = sorted(
            [ScenarioEvent(time=t, type="UAV_FAILURE", data={"uav_id": uid}) for uid, t in (injected_failures or [])],
            key=lambda e: e.time,
        )

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
