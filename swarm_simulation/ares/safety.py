"""Deterministic safety supervisor - higher priority than mission optimization
(section 13). Nothing in allocator/relay/recovery is allowed to bypass this."""
from __future__ import annotations

from .state import SwarmState, Role, dist
from .eventlog import EventLog

def _load_safety_config() -> dict:
    try:
        from . import config

        return config.load_safety_config()
    except Exception:
        return {"collision_distance_m": 3.0, "min_safe_separation_m": 8.0}


_SAFETY_CFG = _load_safety_config()
COLLISION_DISTANCE_M = _SAFETY_CFG["collision_distance_m"]
MIN_SAFE_SEPARATION_M = _SAFETY_CFG["min_safe_separation_m"]


class SafetySupervisor:
    def __init__(self, geofence_min: list[float], geofence_max: list[float],
                 velocity_aware: bool = False, lookahead_s: float = 3.0):
        """`velocity_aware` (off by default, so kinematic results are
        unchanged) makes `check_separation` account for how fast UAVs move:

        * swept detection - the minimum separation over the whole tick is
          computed from each pair's start/end positions, so two UAVs that
          pass through each other between two 1 s ticks are counted
          instead of missed;
        * predictive deconfliction - a closing pair whose constant-velocity
          closest approach within `lookahead_s` would violate
          MIN_SAFE_SEPARATION_M is pushed apart *before* it gets there.

        Call `begin_tick` at the start of each simulation step."""
        self.geofence_min = geofence_min
        self.geofence_max = geofence_max
        self.velocity_aware = velocity_aware
        self.lookahead_s = lookahead_s
        self._prev_positions: dict[str, list[float]] = {}
        self.predictive_deconflictions = 0

    def begin_tick(self, state: SwarmState) -> None:
        """Record start-of-tick positions for swept separation checks."""
        if self.velocity_aware:
            self._prev_positions = {u.id: list(u.position) for u in state.alive_uavs()}

    def enforce_geofence(self, state: SwarmState, log: EventLog) -> None:
        for u in state.alive_uavs():
            clamped = list(u.position)
            violated = False
            for i in range(3):
                if clamped[i] < self.geofence_min[i]:
                    clamped[i] = self.geofence_min[i]
                    violated = True
                elif clamped[i] > self.geofence_max[i]:
                    clamped[i] = self.geofence_max[i]
                    violated = True
            if violated:
                state.geofence_violations += 1
                u.position = clamped
                log.emit(
                    state.time,
                    "SAFETY_OVERRIDE",
                    source=u.id,
                    trigger="geofence",
                    old_state="out_of_bounds",
                    new_state="clamped",
                    reason="position clamped to geofence boundary",
                )

    def check_separation(self, state: SwarmState, log: EventLog) -> None:
        """Counts a collision below COLLISION_DISTANCE_M *and* always
        pushes the pair apart below MIN_SAFE_SEPARATION_M (which is always
        the larger threshold) - previously the push-apart only ran in the
        d < MIN_SAFE_SEPARATION_M branch and was skipped entirely once a
        pair got closer than COLLISION_DISTANCE_M, so a genuine collision
        was never corrected and re-counted every tick for the rest of the
        run (98+ collisions in one generated-scenario run traced back to
        exactly this)."""
        if self.velocity_aware:
            return self._check_separation_velocity_aware(state, log)
        alive = state.alive_uavs()
        for i, a in enumerate(alive):
            for b in alive[i + 1 :]:
                d = dist(a.position, b.position)
                if d < state.min_separation_observed:
                    state.min_separation_observed = d

                if d < COLLISION_DISTANCE_M:
                    state.collisions += 1
                    log.emit(
                        state.time,
                        "SAFETY_OVERRIDE",
                        source=f"{a.id},{b.id}",
                        trigger="min_separation",
                        old_state="proximity_violation",
                        new_state="collision_recorded",
                        reason=f"separation {d:.2f}m < {COLLISION_DISTANCE_M}m",
                    )

                if d < MIN_SAFE_SEPARATION_M:
                    if d < 1e-6:
                        # exactly coincident: no direction to push along,
                        # pick an arbitrary axis rather than divide by zero
                        dx = [1.0, 0.0, 0.0]
                        norm = 1.0
                    else:
                        dx = [bi - ai for ai, bi in zip(a.position, b.position)]
                        norm = d
                    push = (MIN_SAFE_SEPARATION_M - d) / 2.0
                    a.position = [
                        ai - push * (dxi / norm) for ai, dxi in zip(a.position, dx)
                    ]
                    b.position = [
                        bi + push * (dxi / norm) for bi, dxi in zip(b.position, dx)
                    ]

    def _check_separation_velocity_aware(self, state: SwarmState, log: EventLog) -> None:
        import numpy as np

        alive = state.alive_uavs()
        for i, a in enumerate(alive):
            for b in alive[i + 1 :]:
                pa, pb = np.array(a.position), np.array(b.position)
                qa = np.array(self._prev_positions.get(a.id, a.position))
                qb = np.array(self._prev_positions.get(b.id, b.position))

                # closest approach during the tick (linear motion start -> end)
                r0, r1 = qb - qa, pb - pa
                dr = r1 - r0
                denom = float(dr @ dr)
                s_star = 0.0 if denom < 1e-12 else float(np.clip(-(r0 @ dr) / denom, 0.0, 1.0))
                d_swept = float(np.linalg.norm(r0 + s_star * dr))
                d_end = float(np.linalg.norm(r1))
                state.min_separation_observed = min(state.min_separation_observed, d_swept)

                if d_swept < COLLISION_DISTANCE_M:
                    state.collisions += 1
                    log.emit(
                        state.time, "SAFETY_OVERRIDE", source=f"{a.id},{b.id}",
                        trigger="min_separation", old_state="proximity_violation",
                        new_state="collision_recorded",
                        reason=f"swept separation {d_swept:.2f}m < {COLLISION_DISTANCE_M}m "
                               f"(end-of-tick {d_end:.2f}m)",
                    )

                if d_end < MIN_SAFE_SEPARATION_M:
                    self._push_apart(a, b, r1, d_end, MIN_SAFE_SEPARATION_M - d_end)
                    continue

                # predictive: constant relative velocity over the lookahead
                v_rel = np.array(b.velocity) - np.array(a.velocity)
                vv = float(v_rel @ v_rel)
                if vv < 1e-9 or float(r1 @ v_rel) >= 0.0:
                    continue  # not closing
                t_cpa = float(np.clip(-(r1 @ v_rel) / vv, 0.0, self.lookahead_s))
                r_pred = r1 + v_rel * t_cpa
                d_pred = float(np.linalg.norm(r_pred))
                if d_pred < MIN_SAFE_SEPARATION_M:
                    if d_pred < 1e-6:
                        # head-on: deflect sideways (perpendicular to closing direction)
                        side = np.cross(v_rel, [0.0, 0.0, 1.0])
                        if np.linalg.norm(side) < 1e-9:
                            side = np.array([1.0, 0.0, 0.0])
                        r_pred, d_pred = side, float(np.linalg.norm(side))
                    # spread the needed correction over the time left, so it is
                    # a nudge each tick rather than a teleport at the last moment
                    frac = min(1.0, 1.0 / max(t_cpa, 1.0))
                    self._push_apart(a, b, r_pred, d_pred, (MIN_SAFE_SEPARATION_M - d_pred) * frac)
                    self.predictive_deconflictions += 1
                    log.emit(
                        state.time, "SAFETY_OVERRIDE", source=f"{a.id},{b.id}",
                        trigger="predicted_conflict", old_state="closing",
                        new_state="deflected",
                        reason=f"predicted separation {d_pred:.1f}m in {t_cpa:.1f}s "
                               f"< {MIN_SAFE_SEPARATION_M}m",
                    )

    @staticmethod
    def _push_apart(a, b, direction, norm: float, amount: float) -> None:
        """Move a and b apart by `amount` in total along `direction` (a -> b)."""
        unit = [x / max(norm, 1e-9) for x in direction]
        half = amount / 2.0
        a.position = [ai - half * ui for ai, ui in zip(a.position, unit)]
        b.position = [bi + half * ui for bi, ui in zip(b.position, unit)]

    def enforce_battery_reserve(
        self, state: SwarmState, reserve_pct: float, log: EventLog
    ) -> None:
        for u in state.alive_uavs():
            if u.role in (Role.RETURN_TO_BASE, Role.CHARGING):
                continue
            # Distance-aware dynamic RTB: guarantee enough energy to fly back to home base
            d_home = dist(u.position, u.home_position)
            fly_home_time = d_home / max(u.max_speed, 1.0)
            est_drain_rate = 0.03
            energy_to_home = fly_home_time * est_drain_rate
            rtb_threshold = reserve_pct + energy_to_home

            if u.battery <= rtb_threshold:
                old_role = u.role
                u.role = Role.RETURN_TO_BASE
                u.current_task = None
                u.target = list(u.home_position)
                log.emit(
                    state.time,
                    "RETURN_TO_BASE",
                    source=u.id,
                    trigger="battery_reserve",
                    old_state=old_role.value,
                    new_state=Role.RETURN_TO_BASE.value,
                    reason=f"battery {u.battery:.1f}% <= dynamic RTB threshold {rtb_threshold:.1f}% (reserve {reserve_pct}% + {energy_to_home:.1f}% return cost)",
                )
