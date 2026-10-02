"""Deterministic greedy task allocation (section 20 - TaskAllocator interface).

Interface is a class so alternative allocators (Hungarian, min-cost flow,
learned) can be swapped in later without touching the simulator loop.
"""
from __future__ import annotations

from .state import SwarmState, Role, PoIStatus, dist
from .energy import has_sufficient_reserve
from .eventlog import EventLog


class TaskAllocator:
    def assign(self, state: SwarmState, base_drain_rate: float, reserve_pct: float, log: EventLog) -> None:
        raise NotImplementedError


class GreedyPriorityAllocator(TaskAllocator):
    """Assigns unvisited PoIs to the best available idle/backup UAV, ranked
    by priority first and travel cost second."""

    def assign(self, state: SwarmState, base_drain_rate: float, reserve_pct: float, log: EventLog) -> None:
        unvisited = [
            p for p in state.pois.values() if p.status == PoIStatus.UNVISITED
        ]
        if not unvisited:
            all_done = bool(state.pois) and all(p.status == PoIStatus.COMPLETE for p in state.pois.values())
            if all_done:
                for p in state.pois.values():
                    p.status = PoIStatus.UNVISITED
                    p.assigned_uav = None
                unvisited = list(state.pois.values())
            else:
                return
        unvisited.sort(key=lambda p: -p.priority)

        available = [
            u
            for u in state.alive_uavs()
            if u.role in (Role.IDLE, Role.BACKUP_RELAY) and u.current_task is None
        ]

        for poi in unvisited:
            if not available:
                break
            candidates = [
                u
                for u in available
                if has_sufficient_reserve(u, poi.position, base_drain_rate, reserve_pct)
            ]
            if not candidates:
                continue
            candidates.sort(key=lambda u: dist(u.position, poi.position))
            chosen = candidates[0]

            old_role = chosen.role
            chosen.role = Role.SURVEYOR
            chosen.current_task = poi.id
            chosen.target = poi.position
            state.role_changes += 1
            poi.status = PoIStatus.ASSIGNED
            poi.assigned_uav = chosen.id
            available.remove(chosen)

            log.emit(
                state.time,
                "TASK_ASSIGNMENT",
                source=chosen.id,
                trigger="unvisited_poi",
                old_state=old_role.value,
                new_state=chosen.role.value,
                reason=f"assigned to {poi.id} (priority={poi.priority})",
            )
