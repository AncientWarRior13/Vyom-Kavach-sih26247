"""Communication graph builder + link quality estimator (section 6, 8)."""
from __future__ import annotations

from collections import deque

from .state import SwarmState, LinkMetric, NetworkState, dist

GCS_ID = "__gcs__"


def _load_network_config() -> dict:
    try:
        from . import config

        return config.load_network_config()
    except Exception:
        return {
            "coverage_weight": 0.6,
            "pdr_weight": 0.4,
            "network_state_thresholds": {"healthy_min": 0.85, "degraded_min": 0.6, "critical_min": 0.3},
            "pdr_model": {"edge_falloff_exponent": 2, "edge_pdr_scale": 0.98},
            "latency_model": {"base_latency_ms": 15.0, "max_extra_latency_ms": 60.0},
        }


_NET_CFG = _load_network_config()


def _pdr_from_distance(distance: float, comm_range: float) -> float:
    """Simple monotonic falloff model: near-perfect PDR close in, degrading
    toward the edge of range, zero beyond it. Shape controlled by
    config/network.yaml::pdr_model."""
    if distance >= comm_range:
        return 0.0
    ratio = distance / comm_range
    m = _NET_CFG["pdr_model"]
    return max(0.0, 1.0 - ratio ** m["edge_falloff_exponent"]) * m["edge_pdr_scale"]


def _latency_from_distance(distance: float, comm_range: float) -> float:
    ratio = min(distance / comm_range, 1.0)
    m = _NET_CFG["latency_model"]
    return m["base_latency_ms"] + ratio * m["max_extra_latency_ms"]


def build_links(state: SwarmState) -> None:
    """Rebuild the link table from current UAV positions and GCS position.

    Applies any active degraded-link overrides injected by the scenario
    event injector (see recovery/sim for how degraded_links is populated).
    """
    state.links.clear()
    alive = state.alive_uavs()

    # UAV <-> GCS virtual links
    for u in alive:
        d = dist(u.position, state.gcs_position)
        comm_range = u.comm_range
        if d <= comm_range:
            pdr = _pdr_from_distance(d, comm_range)
            lat = _latency_from_distance(d, comm_range)
            state.links[(GCS_ID, u.id)] = LinkMetric(GCS_ID, u.id, d, pdr, lat)

    # UAV <-> UAV links
    for i, a in enumerate(alive):
        for b in alive[i + 1 :]:
            d = dist(a.position, b.position)
            comm_range = min(a.comm_range, b.comm_range)
            if d <= comm_range:
                pdr = _pdr_from_distance(d, comm_range)
                lat = _latency_from_distance(d, comm_range)
                key = tuple(sorted((a.id, b.id)))
                override = state.degraded_links.get(key)
                if override:
                    pdr = max(0.0, pdr * (1.0 - override["packet_loss"]))
                state.links[key] = LinkMetric(a.id, b.id, d, pdr, lat)

    # update neighbor lists
    for u in alive:
        u.neighbors = []
    for (a, b) in state.links:
        if a != GCS_ID:
            state.uavs[a].neighbors.append(b)
        if b != GCS_ID:
            state.uavs[b].neighbors.append(a)


def _adjacency(state: SwarmState, min_pdr: float = 0.05) -> dict[str, set[str]]:
    adj: dict[str, set[str]] = {GCS_ID: set()}
    for u in state.alive_uavs():
        adj.setdefault(u.id, set())
    for (a, b), link in state.links.items():
        if link.pdr < min_pdr:
            continue
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    return adj


def compute_connectivity(state: SwarmState) -> None:
    """BFS from GCS over usable links; records connectivity_path per UAV and
    the connected_to_gcs set."""
    adj = _adjacency(state)
    visited = {GCS_ID: [GCS_ID]}
    q = deque([GCS_ID])
    while q:
        node = q.popleft()
        for nxt in adj.get(node, ()):
            if nxt not in visited:
                visited[nxt] = visited[node] + [nxt]
                q.append(nxt)

    state.connected_to_gcs = {n for n in visited if n != GCS_ID}
    for u in state.alive_uavs():
        u.connectivity_path = visited.get(u.id, [])


def path_redundancy(state: SwarmState, uav_id: str) -> int:
    """Rough redundancy proxy: number of neighbors that are themselves
    connected to GCS via a different first hop than uav's own path."""
    adj = _adjacency(state)
    return len(adj.get(uav_id, set()))


def connectivity_health(state: SwarmState) -> float:
    """Fraction of alive, task-relevant UAVs (SURVEYOR/HYBRID) that have a
    path to the GCS, weighted with average PDR on active links."""
    alive = state.alive_uavs()
    if not alive:
        return 0.0
    connected = sum(1 for u in alive if u.id in state.connected_to_gcs)
    coverage = connected / len(alive)
    pdrs = [l.pdr for l in state.links.values()]
    avg_pdr = sum(pdrs) / len(pdrs) if pdrs else 0.0
    return _NET_CFG["coverage_weight"] * coverage + _NET_CFG["pdr_weight"] * avg_pdr


def update_network_state(state: SwarmState) -> None:
    health = connectivity_health(state)
    t = _NET_CFG["network_state_thresholds"]
    if health >= t["healthy_min"]:
        state.global_network_state = NetworkState.HEALTHY
    elif health >= t["degraded_min"]:
        state.global_network_state = NetworkState.DEGRADED
    elif health >= t["critical_min"]:
        state.global_network_state = NetworkState.CRITICAL
    elif health > 0.0:
        state.global_network_state = NetworkState.RECOVERING
    else:
        state.global_network_state = NetworkState.DISCONNECTED
