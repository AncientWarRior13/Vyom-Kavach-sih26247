"""Formal connectivity-guarantee analysis (research contribution, Phase B3).

Proves a lower bound on the number of relay UAVs required to bridge a
disconnected UAV back to a connected anchor point, independent of *which*
relay-placement algorithm is used - this is a structural resource bound,
not a property of `ares/relay.py`'s specific implementation. It exists to
answer: "is this scenario even repairable with the UAVs available, or is
failure a resource-count problem no algorithm could fix?"

## Theorem (minimum relay chain length)

Let A be a point already connected to the GCS (the "anchor"), U a
disconnected UAV, D = |AU| the straight-line distance between them, and R
the common communication range of the relay-capable UAVs (for
heterogeneous ranges, use R = min over the candidate pool - a smaller R
only requires more relays, so this keeps the bound valid). Let
s in (0, 1] be a safety margin on R (comm range is treated as a hard
cutoff in practice, so real deployments target `s * R < R` to leave
margin - `ares/relay.py::CHAIN_SAFETY_FACTOR` is the concrete value used
elsewhere in this codebase).

For any set of k intermediate relay points p_1, ..., p_k placed between A
and U such that every consecutive pair (including (A, p_1) and (p_k, U))
is within range s*R, the path A -> p_1 -> ... -> p_k -> U consists of
k + 1 segments. By the triangle inequality, the sum of the segment
lengths is at least D (with equality iff every point lies exactly on the
straight line A-U):

    sum_{i=0}^{k} |seg_i| >= D

Since every segment is bounded above by s*R:

    (k + 1) * (s * R) >= sum |seg_i| >= D

    =>  k >= D / (s * R) - 1

Therefore the minimum feasible k is:

    k_min = max(0, ceil(D / (s * R)) - 1)

**This is a lower bound that holds for *any* relay-placement algorithm**,
not just the resilience-potential-map or graph-based repair implemented
in this codebase - no algorithm can bridge the gap with fewer than
`k_min` relays, because doing so would require at least one segment
longer than `s * R`, contradicting the range constraint. It is achievable
(not just a bound) by placing points evenly along the straight line A-U,
which is exactly the "worst case is best case" degenerate configuration -
any obstacle avoidance, collision avoidance, or other real-world
constraint that prevents straight-line placement can only require *more*
relays, never fewer, so k_min is tight in open space and a true lower
bound in general.

## Corollary (repair feasibility)

A gap is repairable at all only if the number of currently available
relay-capable UAVs (`IDLE`/`BACKUP_RELAY`, not already committed) is at
least k_min, *and* committing that many doesn't exceed the relay
commitment cap (`ares/relay.py::MAX_RELAY_FRACTION`, section 7's
mission-vs-connectivity tradeoff). If either fails, the gap is
mathematically unrepairable this tick regardless of algorithm quality -
this is what `is_repair_feasible()` below checks.
"""
from __future__ import annotations

import math

from .state import SwarmState, Role, dist


def min_relay_chain_length(distance: float, comm_range: float, safety_factor: float = 0.85) -> int:
    """k_min from the theorem above: the provable minimum number of relay
    UAVs needed to bridge `distance` given `comm_range`, independent of
    placement algorithm. This matches `ares/relay.py::GraphRelayAllocator
    ._hops_needed`'s formula exactly - that function is this theorem's
    concrete application, not a coincidence."""
    if distance <= 0:
        return 0
    effective_range = max(comm_range * safety_factor, 1e-9)
    return max(0, math.ceil(distance / effective_range) - 1)


def is_repair_feasible(
    distance: float,
    comm_range: float,
    available_relays: int,
    commitment_budget: int,
    safety_factor: float = 0.85,
) -> tuple[bool, int]:
    """Returns (feasible, k_min). Infeasible means: no relay-placement
    algorithm, however good, can repair this gap with the UAVs and
    commitment budget currently available - it is a resource shortage,
    not an algorithm quality problem."""
    k_min = min_relay_chain_length(distance, comm_range, safety_factor)
    feasible = available_relays >= k_min and commitment_budget >= k_min
    return feasible, k_min


def chain_efficiency(actual_relays_used: int, distance: float, comm_range: float, safety_factor: float = 0.85) -> float:
    """How close an actual repair came to the theoretical optimum:
    1.0 means the algorithm used exactly k_min relays (optimal), values
    below 1.0 mean it used more than necessary. Returns None-equivalent
    (float('nan')) when no relays were needed at all (k_min == 0),
    since efficiency is undefined for a already-in-range gap."""
    k_min = min_relay_chain_length(distance, comm_range, safety_factor)
    if k_min == 0:
        return float("nan")
    if actual_relays_used <= 0:
        return 0.0
    return min(1.0, k_min / actual_relays_used)


def scenario_repair_budget(state: SwarmState, max_relay_fraction: float) -> int:
    """The commitment budget available *right now* in a live SwarmState -
    mirrors ares/relay.py::GraphRelayAllocator.allocate's own budget
    computation, exposed here so external analysis doesn't need to
    duplicate or drift from that logic."""
    alive = state.alive_uavs()
    committed = sum(1 for u in alive if u.role in (Role.RELAY, Role.BACKUP_RELAY))
    max_relays = max(1, int(len(alive) * max_relay_fraction))
    return max(0, max_relays - committed)


def nearest_connected_distance(state: SwarmState, uav_id: str) -> float:
    """Straight-line distance from `uav_id` to the nearest GCS-connected
    point (another connected UAV, or the GCS itself) - the D used in the
    theorem for this specific gap."""
    u = state.uavs[uav_id]
    best = dist(u.position, state.gcs_position)
    for uid in state.connected_to_gcs:
        other = state.uavs.get(uid)
        if other:
            best = min(best, dist(u.position, other.position))
    return best
