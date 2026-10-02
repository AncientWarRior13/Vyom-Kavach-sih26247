"""Resilience Potential Map (section 21) - the project's core innovation.

Instead of reactively patching a broken link at its literal midpoint, this
scores a grid of candidate positions for how much they would improve swarm
survivability if a relay/backup UAV moved there, and lets the relay
allocator pick the best one rather than the nearest one.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

from .state import SwarmState, UAVState, dist
from .network import GCS_ID
from .energy import energy_to_point


@dataclass
class ResilienceWeights:
    w_connectivity_gain: float = 2.0
    w_redundancy_gain: float = 1.0
    w_coverage_gain: float = 0.5
    w_progress_gain: float = 1.5
    w_energy_cost: float = 1.0
    w_collision_risk: float = 1.5
    w_latency_penalty: float = 0.5


def _load_default_weights() -> ResilienceWeights:
    try:
        from . import config

        return config.load_resilience_weights()
    except Exception:
        return ResilienceWeights()


DEFAULT_WEIGHTS = _load_default_weights()


def _connectivity_gain(state: SwarmState, point: list[float], comm_range: float) -> float:
    """How many currently GCS-disconnected *task-relevant* UAVs would fall
    within comm_range of a relay placed at `point`, weighted by how close
    it gets them relative to the raw distance they'd otherwise need to
    bridge.

    Deliberately excludes IDLE/BACKUP_RELAY UAVs even when they happen to
    be outside state.connected_to_gcs (e.g. still staged near GCS,
    momentarily out of range) - those don't need reconnecting, and
    counting them here used to reward a relay for parking near a cluster
    of idle standby drones instead of progressing toward the UAV that
    actually needs the path repaired.
    """
    from .state import Role

    gain = 0.0
    disconnected = [
        u for u in state.alive_uavs()
        if u.id not in state.connected_to_gcs
        and u.role in (Role.SURVEYOR, Role.SURVEY_RELAY_HYBRID, Role.RELAY)
    ]
    for u in disconnected:
        d = dist(point, u.position)
        if d <= comm_range:
            gain += 1.0 - (d / comm_range)
    # the point is only useful if it also has its own path back to GCS or a
    # connected node
    reachable_to_backbone = point_reaches_backbone(state, point, comm_range)
    return gain * reachable_to_backbone


def point_reaches_backbone(state: SwarmState, point: list[float], comm_range: float) -> float:
    d_gcs = dist(point, state.gcs_position)
    if d_gcs <= comm_range:
        return 1.0
    best = 0.0
    for uid in state.connected_to_gcs:
        u = state.uavs.get(uid)
        if not u:
            continue
        d = dist(point, u.position)
        if d <= comm_range:
            best = max(best, 1.0 - (d / comm_range) * 0.3)
    return best


def _redundancy_gain(state: SwarmState, point: list[float], comm_range: float) -> float:
    """Extra alternate-path value: count how many already-connected nodes
    would newly neighbor this point (more mesh redundancy, not just one
    bridge)."""
    count = 0
    for uid in state.connected_to_gcs:
        u = state.uavs.get(uid)
        if u and dist(point, u.position) <= comm_range:
            count += 1
    return min(1.0, count / 3.0)


def _coverage_gain(state: SwarmState, point: list[float]) -> float:
    from .state import PoIStatus

    unvisited = [p for p in state.pois.values() if p.status == PoIStatus.UNVISITED]
    if not unvisited:
        return 0.0
    nearest = min(dist(point, p.position) for p in unvisited)
    return max(0.0, 1.0 - nearest / 300.0)


def _collision_risk(state: SwarmState, point: list[float]) -> float:
    risk = 0.0
    for u in state.alive_uavs():
        d = dist(point, u.position)
        if d < 10.0:
            risk += (10.0 - d) / 10.0
    return min(1.0, risk)


def _latency_penalty(state: SwarmState, point: list[float]) -> float:
    return dist(point, state.gcs_position) / 1000.0


def _progress_gain(point: list[float], final_target: list[float]) -> float:
    """Smooth, always-present pull toward the UAV this relay chain is
    actually trying to reach - unlike connectivity_gain, this has no hard
    comm_range cutoff. Without it, a multi-hop chain has nothing counter-
    acting latency_penalty's pull back toward GCS until a hop happens to
    land within direct range of the final target, which collapses long
    chains back toward the anchor instead of spanning the gap."""
    return 1.0 / (1.0 + dist(point, final_target) / 200.0)


def resilience_potential(
    state: SwarmState,
    point: list[float],
    candidate: UAVState,
    base_drain_rate: float,
    comm_range: float,
    final_target: list[float],
    weights: ResilienceWeights = DEFAULT_WEIGHTS,
) -> float:
    connectivity_gain = _connectivity_gain(state, point, comm_range)
    redundancy_gain = _redundancy_gain(state, point, comm_range)
    coverage_gain = _coverage_gain(state, point)
    progress_gain = _progress_gain(point, final_target)
    energy_cost = energy_to_point(candidate, point, base_drain_rate) / 10.0
    collision_risk = _collision_risk(state, point)
    latency_penalty = _latency_penalty(state, point)

    return (
        weights.w_connectivity_gain * connectivity_gain
        + weights.w_redundancy_gain * redundancy_gain
        + weights.w_coverage_gain * coverage_gain
        + weights.w_progress_gain * progress_gain
        - weights.w_energy_cost * energy_cost
        - weights.w_collision_risk * collision_risk
        - weights.w_latency_penalty * latency_penalty
    )


def sample_grid_around(anchor: list[float], gap_point: list[float], n: int = 5) -> list[list[float]]:
    """Sample an n x n grid of candidate relay positions on the plane between
    (and slightly around) the anchor and the gap point."""
    mid = [(a + b) / 2.0 for a, b in zip(anchor, gap_point)]
    span = max(dist(anchor, gap_point) * 0.6, 20.0)
    points = []
    for dx, dy in itertools.product(
        [(-0.5 + i / (n - 1)) for i in range(n)], repeat=2
    ):
        points.append([mid[0] + dx * span, mid[1] + dy * span, mid[2]])
    return points


def best_relay_position(
    state: SwarmState,
    anchor: list[float],
    gap_point: list[float],
    candidate: UAVState,
    base_drain_rate: float,
    comm_range: float,
    final_target: list[float] | None = None,
    weights: ResilienceWeights = DEFAULT_WEIGHTS,
) -> tuple[list[float], float]:
    """Grid-search the resilience potential field and return the best point
    plus its score. This is what makes relay placement predictive rather
    than a naive geometric midpoint.

    `final_target` is the UAV this placement is ultimately trying to
    reconnect - for a single-hop repair it's the same as `gap_point`, but
    for one hop of a multi-hop chain (see relay.py's chain repair),
    `gap_point` is only this hop's local waypoint while `final_target`
    keeps every hop's search biased toward the real destination.
    """
    if final_target is None:
        final_target = gap_point

    candidates = sample_grid_around(anchor, gap_point) + [
        [(a + b) / 2.0 for a, b in zip(anchor, gap_point)]
    ]
    scored = [
        (p, resilience_potential(state, p, candidate, base_drain_rate, comm_range, final_target, weights))
        for p in candidates
    ]
    scored.sort(key=lambda t: -t[1])
    return scored[0]
