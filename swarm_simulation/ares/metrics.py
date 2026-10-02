"""Dedicated metrics engine (sections 18-19). Exports JSON + CSV."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field

from .state import SwarmState, PoIStatus, NetworkState
from .network import connectivity_health
from .recovery import RecoveryPlanner


@dataclass
class StepSample:
    time: float
    connectivity_health: float
    network_state: str
    mean_pdr: float
    mean_latency_ms: float
    connected_fraction: float


class MetricsEngine:
    def __init__(self) -> None:
        self.samples: list[StepSample] = []
        self.disconnect_events = 0
        self._was_disconnected = False

    def record_step(self, state: SwarmState) -> None:
        pdrs = [l.pdr for l in state.links.values()]
        lats = [l.latency_ms for l in state.links.values()]
        alive = state.alive_uavs()
        connected_fraction = (
            len(state.connected_to_gcs) / len(alive) if alive else 0.0
        )
        health = connectivity_health(state)

        self.samples.append(
            StepSample(
                time=state.time,
                connectivity_health=health,
                network_state=state.global_network_state.value,
                mean_pdr=sum(pdrs) / len(pdrs) if pdrs else 0.0,
                mean_latency_ms=sum(lats) / len(lats) if lats else 0.0,
                connected_fraction=connected_fraction,
            )
        )

        is_disc = state.global_network_state == NetworkState.DISCONNECTED
        if is_disc and not self._was_disconnected:
            self.disconnect_events += 1
        self._was_disconnected = is_disc

    def summarize(
        self,
        state: SwarmState,
        recovery: RecoveryPlanner,
        dt: float,
        start_time: float,
        battery_reserve_violations: int,
    ) -> dict:
        n = len(self.samples)
        pois = list(state.pois.values())
        completed = [p for p in pois if p.status == PoIStatus.COMPLETE]
        total_priority = sum(p.priority for p in pois) or 1.0
        priority_completed = sum(p.priority for p in completed)

        completion_time = max((p.completed_at for p in completed), default=None)

        latencies = sorted(s.mean_latency_ms for s in self.samples if s.mean_latency_ms > 0)
        p95_latency = latencies[int(0.95 * (len(latencies) - 1))] if latencies else 0.0
        mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
        mean_pdr = sum(s.mean_pdr for s in self.samples) / n if n else 0.0
        connectivity_availability = (
            sum(1 for s in self.samples if s.network_state in ("HEALTHY", "DEGRADED")) / n
            if n
            else 0.0
        )

        mission = {
            "mission_completion_rate": len(completed) / len(pois) if pois else 1.0,
            "completion_time_s": completion_time,
            "priority_weighted_score": priority_completed / total_priority,
        }
        comm = {
            "packet_delivery_ratio": mean_pdr,
            "mean_latency_ms": mean_latency,
            "p95_latency_ms": p95_latency,
            "connectivity_availability": connectivity_availability,
            "total_communication_downtime_s": recovery.tracker.total_downtime,
            "number_of_disconnect_events": self.disconnect_events,
        }
        autonomy = {
            "relay_reallocations": state.relay_reallocations,
            "role_changes": state.role_changes,
            "mean_reconfiguration_time_s": (
                sum(recovery.tracker.recovery_times) / len(recovery.tracker.recovery_times)
                if recovery.tracker.recovery_times
                else None  # None means no recovery ever completed, not "instant" - never treat as 0
            ),
            "successful_recoveries": recovery.tracker.recoveries_completed,
        }
        safety = {
            "collision_count": state.collisions,
            "minimum_inter_uav_distance_m": (
                state.min_separation_observed if state.min_separation_observed != float("inf") else None
            ),
            "geofence_violations": state.geofence_violations,
            "battery_reserve_violations": battery_reserve_violations,
        }

        score = overall_score(mission, comm, autonomy, safety)

        return {
            "scenario_time_s": state.time,
            "mission": mission,
            "communication": comm,
            "autonomy": autonomy,
            "safety": safety,
            "internal_score_estimate": score,
        }

    def export(self, out_json: str, out_csv: str, summary: dict) -> None:
        with open(out_json, "w") as f:
            json.dump(summary, f, indent=2)
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["time", "connectivity_health", "network_state", "mean_pdr", "mean_latency_ms", "connected_fraction"]
            )
            for s in self.samples:
                writer.writerow(
                    [s.time, s.connectivity_health, s.network_state, s.mean_pdr, s.mean_latency_ms, s.connected_fraction]
                )


def overall_score(mission: dict, comm: dict, autonomy: dict, safety: dict) -> dict:
    """Section 19 - internal benchmarking approximation of the challenge
    score. NOT a claim of reproducing the organizers' hidden scoring."""
    mission_score = 0.5 * mission["mission_completion_rate"] + 0.5 * mission["priority_weighted_score"]
    comm_score = 0.5 * comm["packet_delivery_ratio"] + 0.5 * comm["connectivity_availability"]
    autonomy_score = min(1.0, autonomy["successful_recoveries"] * 0.2 + 0.3)
    mean_reconfig = autonomy["mean_reconfiguration_time_s"]
    if mean_reconfig is None:
        # never recovered at all is worse than any measured slow recovery
        recovery_score = 0.0
    else:
        recovery_penalty = min(1.0, mean_reconfig / 60.0)
        recovery_score = max(0.0, 1.0 - recovery_penalty)
    safety_score = max(
        0.0,
        1.0
        - 0.3 * safety["collision_count"]
        - 0.1 * safety["geofence_violations"]
        - 0.1 * safety["battery_reserve_violations"],
    )
    innovation_score = 0.5  # placeholder: proactive-resilience usage not scored numerically yet

    try:
        from . import config

        w = config.load_score_weights()
    except Exception:
        w = {
            "mission_weight": 0.25, "communication_weight": 0.25, "autonomy_weight": 0.20,
            "recovery_weight": 0.15, "safety_weight": 0.10, "innovation_weight": 0.05,
        }

    overall = (
        w["mission_weight"] * mission_score
        + w["communication_weight"] * comm_score
        + w["autonomy_weight"] * autonomy_score
        + w["recovery_weight"] * recovery_score
        + w["safety_weight"] * safety_score
        + w["innovation_weight"] * innovation_score
    )
    return {
        "mission_score": mission_score,
        "communication_score": comm_score,
        "autonomy_score": autonomy_score,
        "recovery_score": recovery_score,
        "safety_score": safety_score,
        "innovation_score": innovation_score,
        "overall_score": overall,
        "label": "INTERNAL_BENCHMARK_ONLY_NOT_OFFICIAL",
    }
