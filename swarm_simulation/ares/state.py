"""Canonical swarm state model (see PROJECT_CONTEXT.md section 25)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Role(str, Enum):
    IDLE = "IDLE"
    SURVEYOR = "SURVEYOR"
    RELAY = "RELAY"
    SURVEY_RELAY_HYBRID = "SURVEY_RELAY_HYBRID"
    BACKUP_RELAY = "BACKUP_RELAY"
    EMERGENCY_RESPONDER = "EMERGENCY_RESPONDER"
    RETURN_TO_BASE = "RETURN_TO_BASE"
    CHARGING = "CHARGING"
    RECOVERING = "RECOVERING"
    FAILED = "FAILED"


class Health(str, Enum):
    OK = "OK"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"


class PoIStatus(str, Enum):
    UNVISITED = "UNVISITED"
    ASSIGNED = "ASSIGNED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETE = "COMPLETE"


class NetworkState(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"
    DISCONNECTED = "DISCONNECTED"
    RECOVERING = "RECOVERING"


def dist(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


@dataclass
class PoI:
    id: str
    position: list[float]
    priority: float
    required_data: list[str]
    deadline: float
    risk: float
    status: PoIStatus = PoIStatus.UNVISITED
    assigned_uav: Optional[str] = None
    completed_at: Optional[float] = None


@dataclass
class UAVState:
    id: str
    position: list[float]
    home_position: list[float]
    battery: float = 100.0
    velocity: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    role: Role = Role.IDLE
    health: Health = Health.OK
    current_task: Optional[str] = None
    target: Optional[list[float]] = None
    neighbors: list[str] = field(default_factory=list)
    link_metrics: dict = field(default_factory=dict)
    connectivity_path: list[str] = field(default_factory=list)
    flight_time_remaining: float = 0.0
    max_speed: float = 10.0
    comm_range: float = 120.0
    is_backup: bool = False

    @property
    def alive(self) -> bool:
        return self.health != Health.FAILED


@dataclass
class LinkMetric:
    a: str
    b: str
    distance: float
    pdr: float
    latency_ms: float
    packet_loss_override: float = 0.0


@dataclass
class SwarmState:
    time: float = 0.0
    gcs_position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    uavs: dict[str, UAVState] = field(default_factory=dict)
    pois: dict[str, PoI] = field(default_factory=dict)
    links: dict[tuple[str, str], LinkMetric] = field(default_factory=dict)
    connected_to_gcs: set[str] = field(default_factory=set)
    global_network_state: NetworkState = NetworkState.HEALTHY
    disconnected_since: dict[str, float] = field(default_factory=dict)
    degraded_links: dict[tuple[str, str], dict] = field(default_factory=dict)
    relay_reallocations: int = 0
    role_changes: int = 0
    collisions: int = 0
    geofence_violations: int = 0
    min_separation_observed: float = float("inf")

    def alive_uavs(self) -> list[UAVState]:
        return [u for u in self.uavs.values() if u.alive]
