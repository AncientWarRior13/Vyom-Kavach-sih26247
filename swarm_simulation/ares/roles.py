"""Dynamic role scoring (section 5). Weights are configurable so different
policies can be compared experimentally."""
from __future__ import annotations

from dataclasses import dataclass

from .state import SwarmState, UAVState, Role, dist
from .network import GCS_ID


@dataclass
class RoleWeights:
    w_battery: float = 1.0
    w_connectivity: float = 1.2
    w_distance: float = 0.8
    w_priority: float = 1.0
    w_necessity: float = 1.5
    w_energy_cost: float = 0.6
    w_safety_risk: float = 1.0


def _load_default_weights() -> RoleWeights:
    try:
        from . import config

        return config.load_role_weights()
    except Exception:
        return RoleWeights()


DEFAULT_WEIGHTS = _load_default_weights()


def battery_suitability(uav: UAVState, reserve_pct: float) -> float:
    usable = max(0.0, uav.battery - reserve_pct)
    return min(1.0, usable / 60.0)


def connectivity_contribution(state: SwarmState, uav: UAVState) -> float:
    """How much this UAV currently bridges GCS-connected nodes to
    disconnected ones - approximated via neighbor count on both sides."""
    if not uav.neighbors:
        return 0.0
    connected_neighbors = sum(1 for n in uav.neighbors if n in state.connected_to_gcs or n == GCS_ID)
    return min(1.0, connected_neighbors / max(1, len(uav.neighbors)))


def distance_suitability(uav: UAVState, target: list[float], comm_range: float) -> float:
    d = dist(uav.position, target)
    return max(0.0, 1.0 - d / (comm_range * 3))


def relay_role_score(
    state: SwarmState,
    uav: UAVState,
    reserve_pct: float,
    weights: RoleWeights = DEFAULT_WEIGHTS,
    gap_midpoint: list[float] | None = None,
) -> float:
    battery = battery_suitability(uav, reserve_pct)
    connectivity = connectivity_contribution(state, uav)
    distance = (
        distance_suitability(uav, gap_midpoint, uav.comm_range)
        if gap_midpoint
        else 0.5
    )
    necessity = 1.0  # caller only scores candidates when a relay is needed
    energy_cost = 1.0 - battery
    safety_risk = 0.0 if uav.battery > reserve_pct else 1.0

    return (
        weights.w_battery * battery
        + weights.w_connectivity * connectivity
        + weights.w_distance * distance
        + weights.w_necessity * necessity
        - weights.w_energy_cost * energy_cost
        - weights.w_safety_risk * safety_risk
    )
