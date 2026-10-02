"""Static (non-adaptive) baselines used for the experiment plan (section 31).

These exist purely as weaker comparison points to demonstrate that dynamic,
communication-aware allocation actually helps - not as things ARES-X would
ship with.
"""
from __future__ import annotations

from .state import SwarmState, Role, PoIStatus, dist
from .allocator import TaskAllocator
from .relay import RelayAllocator
from .eventlog import EventLog

STATIC_RELAY_COUNT = 2


class StaticTaskAllocator(TaskAllocator):
    """Baseline A/B: assign every PoI to its nearest idle UAV once, at the
    first opportunity, in priority order but never reconsidered afterward -
    no battery-reserve check, no re-allocation if the assignee later fails."""

    def __init__(self) -> None:
        self._assigned_once = False

    def assign(self, state: SwarmState, base_drain_rate: float, reserve_pct: float, log: EventLog) -> None:
        if self._assigned_once:
            return
        unvisited = [p for p in state.pois.values() if p.status == PoIStatus.UNVISITED]
        available = [u for u in state.alive_uavs() if u.role == Role.IDLE]
        if not unvisited or not available:
            return
        unvisited.sort(key=lambda p: -p.priority)

        for poi in unvisited:
            if not available:
                break
            available.sort(key=lambda u: dist(u.position, poi.position))
            chosen = available.pop(0)
            chosen.role = Role.SURVEYOR
            chosen.current_task = poi.id
            chosen.target = poi.position
            poi.status = PoIStatus.ASSIGNED
            poi.assigned_uav = chosen.id
            state.role_changes += 1
            log.emit(
                state.time, "TASK_ASSIGNMENT", source=chosen.id, trigger="static_allocation",
                old_state=Role.IDLE.value, new_state=Role.SURVEYOR.value,
                reason=f"one-shot static assignment to {poi.id}",
            )
        self._assigned_once = True


class StaticRelayAllocator(RelayAllocator):
    """Baseline A: picks a fixed number of relay UAVs once, at fixed
    geometric positions between GCS and the PoI centroid, and never
    reassigns them - even if they fail or the network disconnects."""

    def __init__(self, relay_count: int = STATIC_RELAY_COUNT) -> None:
        self.relay_count = relay_count
        self._assigned_once = False

    def allocate(self, state: SwarmState, base_drain_rate: float, reserve_pct: float, log: EventLog) -> None:
        if self._assigned_once:
            return
        if not state.pois:
            return
        centroid = [
            sum(p.position[i] for p in state.pois.values()) / len(state.pois)
            for i in range(3)
        ]
        available = [u for u in state.alive_uavs() if u.role == Role.IDLE]
        available.sort(key=lambda u: dist(u.position, state.gcs_position))

        for i in range(min(self.relay_count, len(available))):
            frac = (i + 1) / (self.relay_count + 1)
            target = [
                state.gcs_position[j] + frac * (centroid[j] - state.gcs_position[j])
                for j in range(3)
            ]
            chosen = available[i]
            chosen.role = Role.RELAY
            chosen.target = target
            state.relay_reallocations += 1
            state.role_changes += 1
            log.emit(
                state.time, "RELAY_ASSIGNMENT", source=chosen.id, trigger="static_allocation",
                old_state=Role.IDLE.value, new_state=Role.RELAY.value,
                reason="one-shot static relay position, no future reassignment",
            )
        self._assigned_once = True
