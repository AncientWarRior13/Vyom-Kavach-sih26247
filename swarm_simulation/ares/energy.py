"""Battery / energy intelligence (section 12)."""
from __future__ import annotations

from .state import UAVState, Role, dist


def drain_rate(uav: UAVState, base_rate: float, relay_bonus: float) -> float:
    rate = base_rate
    if uav.role in (Role.RELAY, Role.SURVEY_RELAY_HYBRID, Role.BACKUP_RELAY):
        rate += relay_bonus
    return rate


def energy_to_point(uav: UAVState, point: list[float], drain_pct_per_s: float) -> float:
    """Estimated battery percent consumed to fly from current position to
    `point` at max speed."""
    d = dist(uav.position, point)
    if uav.max_speed <= 0:
        return float("inf")
    travel_time = d / uav.max_speed
    return travel_time * drain_pct_per_s


def energy_between(p1: list[float], p2: list[float], max_speed: float, drain_pct_per_s: float) -> float:
    if max_speed <= 0:
        return float("inf")
    return (dist(p1, p2) / max_speed) * drain_pct_per_s


def has_sufficient_reserve(
    uav: UAVState,
    candidate_target: list[float],
    base_rate: float,
    reserve_pct: float,
) -> bool:
    """Section 12 constraint: EstimatedEnergyAfter(a) >= SafetyReserve.

    Checks that after flying to the candidate target, the UAV would still
    have enough battery to fly home from that target position.
    """
    cost_to_target = energy_to_point(uav, candidate_target, base_rate)
    battery_after_target = uav.battery - cost_to_target
    cost_home_from_target = energy_between(candidate_target, uav.home_position, uav.max_speed, base_rate)
    return (battery_after_target - cost_home_from_target) >= reserve_pct
