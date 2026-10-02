"""Randomized-geometry scenario generator.

Every scenario in config/scenarios/ has fixed UAV/PoI/event positions, so
running the batch experiment with different seeds currently replays the
same four layouts (see docs/experiments.md's determinism note) - none of
the shipped allocators consume randomness, so a fixed-geometry scenario
produces bit-identical output regardless of seed.

This generator produces a *new* randomized scenario dict (same schema as
scenario.py::Scenario.load expects) per seed: UAV start spread, PoI count/
positions/priorities, and failure/degradation event timing/targets are all
drawn from the seed's RNG. This is what makes a seed sweep across the
batch experiment actually mean something.
"""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class GeneratorConfig:
    uav_count_range: tuple[int, int] = (6, 12)
    poi_count_range: tuple[int, int] = (4, 8)
    area_size: tuple[float, float] = (500.0, 450.0)
    comm_range_range: tuple[float, float] = (100.0, 150.0)
    max_speed_range: tuple[float, float] = (9.0, 14.0)
    duration_seconds: float = 600.0
    num_failure_events_range: tuple[int, int] = (1, 3)
    num_degradation_events_range: tuple[int, int] = (0, 2)
    num_priority_events_range: tuple[int, int] = (0, 2)


def generate_scenario(seed: int, name: str | None = None, cfg: GeneratorConfig = GeneratorConfig()) -> dict:
    """Returns a scenario dict matching the YAML schema in docs/scenarios.md,
    fully determined by `seed` - same seed always produces the same
    scenario, different seeds produce genuinely different topologies."""
    rng = random.Random(seed)
    name = name or f"generated_seed_{seed}"

    width, height = cfg.area_size
    gcs_position = [0.0, height / 2, 0.0]
    home_position = list(gcs_position)

    uav_count = rng.randint(*cfg.uav_count_range)
    comm_range = round(rng.uniform(*cfg.comm_range_range), 1)
    max_speed = round(rng.uniform(*cfg.max_speed_range), 1)

    poi_count = rng.randint(*cfg.poi_count_range)
    pois = []
    for i in range(poi_count):
        pois.append({
            "id": f"poi_{i+1:02d}",
            "position": [
                round(rng.uniform(80, width), 1),
                round(rng.uniform(20, height - 20), 1),
                40.0,
            ],
            "priority": round(rng.uniform(0.3, 0.85), 2),
            "required_data": ["imagery"],
            "deadline": cfg.duration_seconds - 50,
            "risk": round(rng.uniform(0.05, 0.35), 2),
        })

    uav_ids = [f"uav_{i+1:02d}" for i in range(uav_count)]
    events = []

    num_failures = rng.randint(*cfg.num_failure_events_range)
    for _ in range(num_failures):
        events.append({
            "time": round(rng.uniform(60, cfg.duration_seconds * 0.7), 1),
            "type": "UAV_FAILURE",
            "uav_id": rng.choice(uav_ids),
        })

    num_degradations = rng.randint(*cfg.num_degradation_events_range)
    for _ in range(num_degradations):
        a, b = rng.sample(uav_ids, 2)
        events.append({
            "time": round(rng.uniform(60, cfg.duration_seconds * 0.7), 1),
            "type": "LINK_DEGRADATION",
            "link": [a, b],
            "packet_loss": round(rng.uniform(0.3, 0.8), 2),
            "duration": round(rng.uniform(30, 120), 1),
        })

    num_priorities = rng.randint(*cfg.num_priority_events_range)
    for i in range(num_priorities):
        events.append({
            "time": round(rng.uniform(cfg.duration_seconds * 0.3, cfg.duration_seconds * 0.8), 1),
            "type": "NEW_PRIORITY_POI",
            "poi_id": f"poi_gen_{i+1}",
            "priority": round(rng.uniform(0.9, 1.0), 2),
            "position": [
                round(rng.uniform(80, width), 1),
                round(rng.uniform(20, height - 20), 1),
                40.0,
            ],
        })

    events.sort(key=lambda e: e["time"])

    return {
        "scenario": {
            "name": name,
            "seed": seed,
            "duration_seconds": cfg.duration_seconds,
            "dt_seconds": 1.0,
            "geofence": {"min": [-50, -50, 0], "max": [width + 50, height + 50, 120]},
        },
        "gcs": {"position": gcs_position},
        "swarm": {
            "count": uav_count,
            "battery_percent": 100,
            "max_speed_mps": max_speed,
            "comm_range_m": comm_range,
            "battery_drain_rate_pct_per_s": 0.03,
            "battery_drain_rate_relay_bonus": 0.005,
            "battery_reserve_pct": 20.0,
            "home_position": home_position,
        },
        "pois": pois,
        "events": events,
    }


def write_scenario_yaml(scenario_dict: dict, path: str) -> None:
    import yaml

    with open(path, "w") as f:
        yaml.safe_dump(scenario_dict, f, sort_keys=False)
