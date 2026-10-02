"""Scenario loading: turns a YAML file into plain dataclasses.

The autonomy code never reads YAML directly - everything scenario-specific
flows through this module so new scenarios never require code changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import yaml


@dataclass
class PoISpec:
    id: str
    position: list[float]
    priority: float
    required_data: list[str] = field(default_factory=list)
    deadline: float = 0.0
    risk: float = 0.0


@dataclass
class ScenarioEvent:
    time: float
    type: str
    data: dict[str, Any]


@dataclass
class Scenario:
    name: str
    seed: int
    duration_seconds: float
    dt_seconds: float
    geofence_min: list[float]
    geofence_max: list[float]
    gcs_position: list[float]
    uav_count: int
    battery_percent: float
    max_speed_mps: float
    comm_range_m: float
    battery_drain_rate: float
    battery_drain_rate_relay_bonus: float
    battery_reserve_pct: float
    home_position: list[float]
    pois: list[PoISpec]
    events: list[ScenarioEvent]
    # optional true-vehicle parameters for the dynamic flight model
    # (alpha, mass, drag_coeff, wind); ignored by the kinematic simulator
    dynamics: dict[str, Any] | None = None

    @staticmethod
    def load(path: str) -> "Scenario":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return Scenario.from_dict(raw)

    @staticmethod
    def from_dict(raw: dict) -> "Scenario":
        sc = raw["scenario"]
        swarm = raw["swarm"]
        gcs = raw["gcs"]

        pois = [
            PoISpec(
                id=p["id"],
                position=p["position"],
                priority=p["priority"],
                required_data=p.get("required_data", []),
                deadline=p.get("deadline", 0.0),
                risk=p.get("risk", 0.0),
            )
            for p in raw.get("pois", [])
        ]

        events = sorted(
            (
                ScenarioEvent(time=e["time"], type=e["type"], data=e)
                for e in raw.get("events", [])
            ),
            key=lambda e: e.time,
        )

        return Scenario(
            name=sc["name"],
            seed=sc.get("seed", 0),
            duration_seconds=sc["duration_seconds"],
            dt_seconds=sc.get("dt_seconds", 1.0),
            geofence_min=sc["geofence"]["min"],
            geofence_max=sc["geofence"]["max"],
            gcs_position=gcs["position"],
            uav_count=swarm["count"],
            battery_percent=swarm.get("battery_percent", 100.0),
            max_speed_mps=swarm.get("max_speed_mps", 10.0),
            comm_range_m=swarm.get("comm_range_m", 120.0),
            battery_drain_rate=swarm.get("battery_drain_rate_pct_per_s", 0.03),
            battery_drain_rate_relay_bonus=swarm.get(
                "battery_drain_rate_relay_bonus", 0.0
            ),
            battery_reserve_pct=swarm.get("battery_reserve_pct", 20.0),
            home_position=swarm.get("home_position", gcs["position"]),
            pois=pois,
            events=events,
            dynamics=raw.get("dynamics"),
        )
