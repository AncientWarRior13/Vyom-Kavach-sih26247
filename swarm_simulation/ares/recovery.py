"""Failure detection + recovery pipeline (section 9)."""
from __future__ import annotations

from dataclasses import dataclass, field

from .state import SwarmState, Role, Health, PoIStatus, NetworkState
from .eventlog import EventLog


@dataclass
class RecoveryTracker:
    """Tracks in-flight recoveries so we can measure recovery_time and
    communication_downtime once connectivity is restored."""

    active_since: float | None = None
    downtime_accum: float = 0.0
    recoveries_completed: int = 0
    recovery_times: list[float] = field(default_factory=list)
    total_downtime: float = 0.0


class FaultDetector:
    def detect(self, state: SwarmState, log: EventLog) -> None:
        for uav in state.uavs.values():
            if uav.health == Health.FAILED and uav.role != Role.FAILED:
                self._handle_uav_failure(state, uav, log)

    def _handle_uav_failure(self, state: SwarmState, uav, log: EventLog) -> None:
        old_role = uav.role
        uav.role = Role.FAILED
        uav.current_task = None
        uav.target = None

        # release any PoI this UAV was working on back to the pool
        for poi in state.pois.values():
            if poi.assigned_uav == uav.id and poi.status != PoIStatus.COMPLETE:
                poi.status = PoIStatus.UNVISITED
                poi.assigned_uav = None

        log.emit(
            state.time,
            "UAV_FAILURE_DETECTED",
            source=uav.id,
            trigger="health_check",
            old_state=old_role.value,
            new_state=Role.FAILED.value,
            reason="UAV reported hard failure",
        )


class RecoveryPlanner:
    """Tracks network health transitions to compute recovery_time and
    communication_downtime; emits RECOVERY_STARTED / RECOVERY_COMPLETED."""

    def __init__(self) -> None:
        self.tracker = RecoveryTracker()

    def update(self, state: SwarmState, dt: float, log: EventLog) -> None:
        unhealthy = state.global_network_state in (
            NetworkState.CRITICAL,
            NetworkState.DISCONNECTED,
            NetworkState.RECOVERING,
        )

        if unhealthy and self.tracker.active_since is None:
            self.tracker.active_since = state.time
            log.emit(
                state.time,
                "RECOVERY_STARTED",
                source="swarm",
                trigger="network_state",
                old_state="HEALTHY_OR_DEGRADED",
                new_state=state.global_network_state.value,
                reason="connectivity health dropped below CRITICAL threshold",
            )

        if unhealthy:
            self.tracker.downtime_accum += dt
            self.tracker.total_downtime += dt

        if not unhealthy and self.tracker.active_since is not None:
            recovery_time = state.time - self.tracker.active_since
            self.tracker.recovery_times.append(recovery_time)
            self.tracker.recoveries_completed += 1
            log.emit(
                state.time,
                "RECOVERY_COMPLETED",
                source="swarm",
                trigger="network_state",
                old_state="CRITICAL_OR_DISCONNECTED",
                new_state=state.global_network_state.value,
                reason=f"connectivity restored after {recovery_time:.1f}s",
            )
            self.tracker.active_since = None
            self.tracker.downtime_accum = 0.0
