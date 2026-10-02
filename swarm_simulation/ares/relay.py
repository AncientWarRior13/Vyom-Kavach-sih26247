"""Graph-based relay selection + proactive resilience (sections 9, 10, 20)."""
from __future__ import annotations

import math

from .state import SwarmState, Role, dist
from .network import GCS_ID
from .roles import relay_role_score
from .resilience import best_relay_position
from .eventlog import EventLog

CHAIN_SAFETY_FACTOR = 0.85  # consecutive hops target this fraction of comm_range, not the full range

def _load_proactive_margin() -> float:
    try:
        from . import config

        return config.load_proactive_battery_margin()
    except Exception:
        return 2.0


def _load_max_relay_fraction() -> float:
    try:
        from . import config

        return config.load_max_relay_fraction()
    except Exception:
        return 0.6


PROACTIVE_BATTERY_MARGIN = _load_proactive_margin()  # multiple of reserve_pct that triggers pre-positioning
MAX_RELAY_FRACTION = _load_max_relay_fraction()  # cap on the share of the swarm committed to RELAY/BACKUP_RELAY


class RelayAllocator:
    def allocate(
        self,
        state: SwarmState,
        base_drain_rate: float,
        reserve_pct: float,
        log: EventLog,
    ) -> None:
        raise NotImplementedError


class GraphRelayAllocator(RelayAllocator):
    """Reactively repairs broken GCS paths and (optionally) proactively
    pre-positions a backup relay when an active relay's battery is running
    low. `use_resilience_map=False` and `proactive=False` are used by the
    Baseline C comparison policy (section 31) to isolate the contribution
    of the resilience-potential-map innovation from plain dynamic relay
    allocation."""

    def __init__(self, use_resilience_map: bool = True, proactive: bool = True) -> None:
        self.use_resilience_map = use_resilience_map
        self.proactive = proactive

    def allocate(self, state: SwarmState, base_drain_rate: float, reserve_pct: float, log: EventLog) -> None:
        # Cap how many *additional* UAVs this tick's relay logic may commit,
        # so it can't starve task allocation of every free UAV chasing full
        # connectivity (PROJECT_CONTEXT.md section 7's MissionUtility
        # tradeoff - see docs/experiments.md's "Why mission completion is
        # 95% instead of 100%" for the failure mode this fixes: on one
        # scenario the old unbounded version committed all 8 UAVs to relay
        # duty and left nothing to survey a late-arriving PoI).
        alive = state.alive_uavs()
        committed = sum(1 for u in alive if u.role in (Role.RELAY, Role.BACKUP_RELAY))
        max_relays = max(1, int(len(alive) * MAX_RELAY_FRACTION))
        budget = max(0, max_relays - committed)

        budget = self._repair_broken_paths(state, base_drain_rate, reserve_pct, log, budget)
        if self.proactive:
            self._proactive_repositioning(state, reserve_pct, log, budget)

    def _candidates(self, state: SwarmState, budget: int | None = None) -> list:
        pool = [
            u
            for u in state.alive_uavs()
            if u.role in (Role.IDLE, Role.BACKUP_RELAY)
        ]
        return pool if budget is None else pool[:budget]

    def _nearest_connected_point(self, state: SwarmState, from_pos: list[float]) -> list[float]:
        best = state.gcs_position
        best_d = dist(from_pos, state.gcs_position)
        for uid in state.connected_to_gcs:
            u = state.uavs.get(uid)
            if not u:
                continue
            d = dist(from_pos, u.position)
            if d < best_d:
                best_d = d
                best = u.position
        return best

    def _relays_committed_to_gap(self, state: SwarmState, gap_uav_id: str) -> list:
        """Every relay already dispatched for this gap (current_task ==
        'relay_for::<gap_uav_id>'), regardless of whether it's arrived
        yet. Used two ways: (1) to avoid re-dispatching a *duplicate*
        relay onto the same spot while the first is still in transit (see
        docs/experiments.md's "Bugs found and fixed"), and (2) to detect
        when the existing chain has become *insufficient* because the gap
        UAV kept moving further away after the chain was first sized -
        target-seeking movement continues regardless of connectivity
        status, so a UAV flying to a far PoI can outgrow a chain that was
        correctly sized at the moment it first disconnected. Without
        re-checking sufficiency here, the chain is sized once and never
        extended even as the gap widens - see
        docs/connectivity_analysis.md's discovery of this while trying to
        exercise the k_min>=2 case in a live scenario."""
        task_marker = f"relay_for::{gap_uav_id}"
        return [
            u for u in state.alive_uavs()
            if u.current_task == task_marker and u.role == Role.RELAY
        ]

    def _relay_still_in_transit(self, u) -> bool:
        """A RELAY UAV just dispatched to a post is, by construction,
        disconnected from GCS until it physically arrives - that's an
        artifact of movement lag, not a real network failure. Without this
        check, a relay-in-transit looks exactly like a newly-broken path
        and gets its *own* rescue relay dispatched, which can itself be
        in transit and need another, cascading (see
        docs/experiments.md's "Bugs found and fixed")."""
        if u.role != Role.RELAY or not u.target:
            return False
        return dist(u.position, u.target) > 5.0

    def _additional_hops_needed(self, chain_tail: list[float], gap_pos: list[float], comm_range: float) -> int:
        """Unlike _hops_needed (which always returns >= 1, correct only
        when called on a point already confirmed out of range - see its
        docstring and ares/connectivity_bounds.py), this checks
        sufficiency FIRST: 0 means the chain tail can already reach the
        gap directly (nothing more to deploy), only calling into
        _hops_needed's chain-length math when a genuine shortfall exists."""
        effective_range = max(comm_range * CHAIN_SAFETY_FACTOR, 1.0)
        if dist(chain_tail, gap_pos) <= effective_range:
            return 0
        return self._hops_needed(chain_tail, gap_pos, comm_range)

    def _repair_broken_paths(self, state: SwarmState, base_drain_rate: float, reserve_pct: float, log: EventLog, budget: int) -> int:
        """Returns the remaining budget after this pass, for
        _proactive_repositioning to respect too."""
        approx_range = state.alive_uavs()[0].comm_range if state.alive_uavs() else 120.0
        gaps_needing_work = []
        for u in state.alive_uavs():
            if u.role not in (Role.SURVEYOR, Role.SURVEY_RELAY_HYBRID, Role.RELAY):
                continue
            if u.id in state.connected_to_gcs:
                continue
            if self._relay_still_in_transit(u):
                continue
            existing = self._relays_committed_to_gap(state, u.id)
            if existing:
                # extend from the *planned* end of the existing chain
                # (target, not live position - a relay still flying there
                # is already "spoken for") rather than restarting from the
                # true nearest-connected point, which wouldn't see an
                # in-transit relay as connected yet and would duplicate it
                chain_tail = min((r.target or r.position for r in existing), key=lambda p: dist(p, u.position))
            else:
                chain_tail = self._nearest_connected_point(state, u.position)

            shortfall = self._additional_hops_needed(chain_tail, u.position, approx_range)
            if shortfall > 0:
                gaps_needing_work.append((u, chain_tail, existing, shortfall))

        if not gaps_needing_work or budget <= 0:
            return budget

        candidates = self._candidates(state, budget)
        budget_used_marker = len(candidates)
        # Tracks every relay target that's either already occupied (an
        # existing relay's parked position/target) or picked during this
        # tick's repair pass. Without the *existing* half of this, a later
        # tick's repair for a *different* gap can still converge on almost
        # the same spot as an already-parked relay - redundancy_gain
        # rewards being near an already-connected node, which perversely
        # attracts new relays to cluster on top of existing ones instead
        # of spreading out (see docs/experiments.md's "Bugs found and
        # fixed"). Within-tick-only deconfliction caught the same-tick
        # case but not this cross-tick one.
        assigned_targets_this_tick: list[list[float]] = [
            u.target
            for u in state.alive_uavs()
            if u.role == Role.RELAY and u.target
        ]
        for gap_uav, chain_tail, existing, shortfall in gaps_needing_work:
            if not candidates:
                break
            self._repair_one_gap(
                state, chain_tail, gap_uav, candidates, base_drain_rate, reserve_pct, log,
                assigned_targets_this_tick, already_placed=len(existing), shortfall=shortfall,
            )

        consumed = budget_used_marker - len(candidates)
        return budget - consumed

    def _deconflict_target(self, target: list[float], assigned_targets: list[list[float]]) -> list[float]:
        """Pushes `target` away from any target already claimed this tick
        that it would otherwise collide with. Iterates because pushing away
        from one conflict can create a new one against another target."""
        from .safety import MIN_SAFE_SEPARATION_M

        for _ in range(len(assigned_targets) + 1):
            conflict = next(
                (t for t in assigned_targets if dist(target, t) < MIN_SAFE_SEPARATION_M),
                None,
            )
            if conflict is None:
                return target
            d = dist(target, conflict)
            if d < 1e-6:
                direction = [1.0, 0.0, 0.0]
                d = 1.0
            else:
                direction = [(t - c) / d for t, c in zip(target, conflict)]
            push = MIN_SAFE_SEPARATION_M - d
            target = [t + direction[i] * push for i, t in enumerate(target)]
        return target

    def _clamp_to_range(self, chain_anchor: list[float], target: list[float], comm_range: float) -> list[float]:
        """The resilience map's grid search optimizes for connectivity/
        safety/coverage and can occasionally pick a point slightly outside
        comm range of the previous hop, defeating the point of chaining.
        Pull it back onto the effective-range boundary along the same
        direction rather than silently accepting a link that won't work."""
        effective_range = comm_range * CHAIN_SAFETY_FACTOR
        d = dist(chain_anchor, target)
        if d <= effective_range or d < 1e-9:
            return target
        scale = effective_range / d
        return [a + (t - a) * scale for a, t in zip(chain_anchor, target)]

    def _hops_needed(self, anchor: list[float], gap_point: list[float], comm_range: float) -> int:
        """How many intermediate relays are needed to bridge `anchor` to
        `gap_point`, given each hop can only cover `comm_range *
        CHAIN_SAFETY_FACTOR`. A gap narrower than one hop needs 1 relay
        (the original single-relay case); wider gaps need a chain."""
        effective_range = max(comm_range * CHAIN_SAFETY_FACTOR, 1.0)
        span = dist(anchor, gap_point)
        return max(1, math.ceil(span / effective_range) - 1)

    def _repair_one_gap(self, state, anchor, gap_uav, candidates, base_drain_rate, reserve_pct, log,
                         assigned_targets_this_tick, already_placed: int = 0, shortfall: int | None = None) -> None:
        """Repairs the path to `gap_uav`, placing a *chain* of relays when
        the gap is wider than one comm-range hop (section 9's requirement
        to rebuild the graph, not just patch the nearest single link).

        `anchor` here is already the correct starting point for this
        pass - either the true nearest connected point (no chain exists
        yet) or the planned end of an already-deployed partial chain
        (`_repair_broken_paths` resolves which). `already_placed` is only
        used for hop-count display in the log; `shortfall` is exactly how
        many *new* relays this call should place, already computed by the
        caller via `_additional_hops_needed` - a UAV can keep flying
        further from its chain after the chain was first sized (e.g.
        toward a distant PoI; connectivity status doesn't pause
        target-seeking movement), so a chain correctly sized at first
        detection can become insufficient later, and this shortfall-based
        extension is what lets it catch up instead of staying
        permanently undersized.
        """
        hops_needed = shortfall if shortfall is not None else self._hops_needed(anchor, gap_uav.position, candidates[0].comm_range if candidates else 120.0)
        hops_needed = min(hops_needed, len(candidates))
        chain_anchor = anchor

        for hop_index in range(hops_needed):
            if not candidates:
                return
            # Waypoint is recomputed each iteration relative to where the
            # chain *actually* is (chain_anchor), not the original anchor -
            # if an earlier hop's resilience-optimized position drifted
            # from the ideal even-spacing point (it usually does), later
            # waypoints must adapt or the chain stops lining up with
            # itself. remaining_hops includes this hop.
            remaining_hops = hops_needed - hop_index
            frac = 1.0 / (remaining_hops + 1)
            waypoint = [a + frac * (g - a) for a, g in zip(chain_anchor, gap_uav.position)]

            scored = sorted(
                candidates,
                key=lambda c: -relay_role_score(state, c, reserve_pct, gap_midpoint=waypoint),
            )
            chosen = scored[0]
            if self.use_resilience_map:
                target, potential = best_relay_position(
                    state, chain_anchor, waypoint, chosen, base_drain_rate, chosen.comm_range,
                    final_target=gap_uav.position,
                )
                target = self._clamp_to_range(chain_anchor, target, chosen.comm_range)
                is_last_hop = hop_index == hops_needed - 1
                if is_last_hop and dist(target, gap_uav.position) > chosen.comm_range * CHAIN_SAFETY_FACTOR:
                    # The resilience map optimized this hop's link back to
                    # chain_anchor but broke the final link to the actual
                    # gap_uav - fall back to the plain waypoint, which by
                    # construction (even spacing) satisfies both ends.
                    target = waypoint
            else:
                target = waypoint
                potential = 0.0

            target = self._deconflict_target(target, assigned_targets_this_tick)
            assigned_targets_this_tick.append(target)

            old_role = chosen.role
            chosen.role = Role.RELAY
            chosen.target = target
            chosen.current_task = f"relay_for::{gap_uav.id}"
            state.relay_reallocations += 1
            state.role_changes += 1
            candidates.remove(chosen)

            total_hops = already_placed + hops_needed
            hop_note = f" [hop {already_placed + hop_index + 1}/{total_hops}]" if total_hops > 1 else ""
            log.emit(
                state.time,
                "RELAY_ASSIGNMENT",
                source=chosen.id,
                trigger="broken_gcs_path",
                old_state=old_role.value,
                new_state=chosen.role.value,
                reason=f"repositioning to bridge {gap_uav.id} back to GCS{hop_note} (resilience_potential={potential:.2f})",
            )
            chain_anchor = target

    def _proactive_repositioning(self, state: SwarmState, reserve_pct: float, log: EventLog, budget: int) -> None:
        active_relays = [
            u for u in state.alive_uavs() if u.role in (Role.RELAY, Role.SURVEY_RELAY_HYBRID)
        ]
        fragile = [r for r in active_relays if r.battery < reserve_pct * PROACTIVE_BATTERY_MARGIN]
        if not fragile or budget <= 0:
            return

        candidates = self._candidates(state, budget)
        for relay in fragile:
            if not candidates:
                break
            scored = sorted(
                candidates,
                key=lambda c: -relay_role_score(state, c, reserve_pct, gap_midpoint=relay.position),
            )
            chosen = scored[0]
            old_role = chosen.role
            chosen.role = Role.BACKUP_RELAY
            chosen.target = relay.position
            chosen.is_backup = True
            candidates.remove(chosen)

            log.emit(
                state.time,
                "PROACTIVE_RELAY_STAGING",
                source=chosen.id,
                trigger="fragile_relay_battery",
                old_state=old_role.value,
                new_state=chosen.role.value,
                reason=f"pre-positioning backup for fragile relay {relay.id} (battery={relay.battery:.1f}%)",
                confidence=0.7,
            )
