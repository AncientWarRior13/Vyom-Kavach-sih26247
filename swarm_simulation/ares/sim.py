"""Simulator: the event-driven loop tying every module together (sections
15-17, 37). This is simulator-agnostic autonomy logic driving a lightweight
kinematic UAV model; a PX4/Gazebo adapter can later replace just the
movement/telemetry layer without touching allocator/relay/recovery/safety.
"""
from __future__ import annotations

import math
import random

from .scenario import Scenario
from .state import SwarmState, UAVState, PoI, PoIStatus, Role, Health, dist
from .network import build_links, compute_connectivity, update_network_state
from .allocator import GreedyPriorityAllocator
from .relay import GraphRelayAllocator
from .recovery import FaultDetector, RecoveryPlanner
from .safety import SafetySupervisor
from .metrics import MetricsEngine
from .eventlog import EventLog

def _load_sim_constants():
    try:
        from . import config

        cfg = config.load_safety_config()
        return cfg["poi_arrival_radius_m"], cfg["charge_rate_pct_per_s"]
    except Exception:
        return 5.0, 0.4


POI_ARRIVAL_RADIUS_M, CHARGE_RATE_PCT_PER_S = _load_sim_constants()


class Simulator:
    def __init__(self, scenario: Scenario, task_allocator=None, relay_allocator=None,
                 enable_link_predictor_diagnostics: bool = False,
                 flight_model: str = "kinematic", flight_controller: str = "adaptive",
                 flight_weights_path: str | None = None,
                 velocity_aware_safety: bool | None = None,
                 arrival_radius_m: float | None = None):
        """`task_allocator`/`relay_allocator` default to the proposed
        dynamic, communication-aware policies. Pass alternate
        implementations (see `ares.baselines`) to run the Baseline A/B/C
        comparisons from section 31 without duplicating the simulator.

        `enable_link_predictor_diagnostics`: opt-in, off by default. Loads
        the trained `LinkQualityPredictor` (ares/link_predictor.py, Phase B1)
        and records its per-link prediction alongside the actual PDR each
        tick, purely for comparison/monitoring (`self.link_quality_predictions`).
        This does NOT feed into any allocator decision - the predictor is a
        documented negative result (loses to a naive baseline by 14.2%, see
        docs/link_predictor_analysis.md) and stays out of the decision path;
        this only makes it observable at runtime instead of untouched dead code."""
        self.scenario = scenario
        # how close a UAV must get to its target to count as arrived; defaults
        # to config/safety.yaml. Tighten it to model precision tasks
        # (hover-and-image, landing) where tracking accuracy matters.
        self.arrival_radius_m = POI_ARRIVAL_RADIUS_M if arrival_radius_m is None else arrival_radius_m
        self.rng = random.Random(scenario.seed)
        self.state = self._init_state()

        # `flight_model="dynamic"` swaps the straight-line kinematic seek for
        # the point-mass quadrotor + flight controller in ares/flight.py
        # (`flight_controller`: nominal | adaptive | oracle). Off by default,
        # so every existing result is unchanged.
        if flight_model not in ("kinematic", "dynamic"):
            raise ValueError(f"flight_model must be 'kinematic' or 'dynamic', got {flight_model!r}")
        self.flight = None
        if flight_model == "dynamic":
            from .flight import DynamicFlightModel

            self.flight = DynamicFlightModel(
                self.state, controller=flight_controller, weights_path=flight_weights_path,
                vehicle_defaults=scenario.dynamics, seed=scenario.seed,
            )
        self.log = EventLog()
        self.metrics = MetricsEngine()
        self.task_allocator = task_allocator or GreedyPriorityAllocator()
        self.relay_allocator = relay_allocator or GraphRelayAllocator()
        self.fault_detector = FaultDetector()
        self.recovery_planner = RecoveryPlanner()
        # velocity-aware separation (swept + predictive) defaults to on only
        # for the dynamic flight model, where inertia makes it matter; the
        # kinematic simulator keeps its original 1 s end-of-tick check.
        if velocity_aware_safety is None:
            velocity_aware_safety = self.flight is not None
        self.safety = SafetySupervisor(scenario.geofence_min, scenario.geofence_max,
                                       velocity_aware=velocity_aware_safety)
        self._events_processed = 0
        self.battery_reserve_violations = 0

        self.link_predictor = None
        self.link_quality_predictions: dict = {}
        self._prev_link_distance: dict = {}
        if enable_link_predictor_diagnostics:
            import os
            from .link_predictor import LinkQualityPredictor

            weights_path = os.path.join(os.path.dirname(__file__), "link_predictor_weights.json")
            self.link_predictor = LinkQualityPredictor.load(weights_path)

    def _init_state(self) -> SwarmState:
        sc = self.scenario
        state = SwarmState(time=0.0, gcs_position=list(sc.gcs_position))

        # Spread radius must grow with swarm size, or adjacent UAVs on a
        # fixed-radius circle pack closer together as more are added - a
        # fixed 15m radius was fine up to ~10 UAVs but produced genuine
        # t=0 collisions (adjacent chord < COLLISION_DISTANCE_M) at 20,
        # discovered building config/scenarios/wide_gap_multihop.yaml.
        # Sized so adjacent spawn spacing clears MIN_SAFE_SEPARATION_M,
        # not just the harder collision threshold, with margin.
        from .safety import MIN_SAFE_SEPARATION_M

        spread = max(15.0, (MIN_SAFE_SEPARATION_M * 1.5 * sc.uav_count) / (2 * math.pi))

        for i in range(sc.uav_count):
            angle = 2 * math.pi * i / sc.uav_count
            pos = [
                sc.gcs_position[0] + spread * math.cos(angle),
                sc.gcs_position[1] + spread * math.sin(angle),
                sc.gcs_position[2] + 30.0,
            ]
            uid = f"uav_{i+1:02d}"
            state.uavs[uid] = UAVState(
                id=uid,
                position=pos,
                home_position=list(sc.home_position),
                battery=sc.battery_percent,
                role=Role.IDLE,
                max_speed=sc.max_speed_mps,
                comm_range=sc.comm_range_m,
            )

        for p in sc.pois:
            state.pois[p.id] = PoI(
                id=p.id,
                position=list(p.position),
                priority=p.priority,
                required_data=list(p.required_data),
                deadline=p.deadline,
                risk=p.risk,
            )

        return state

    # ---- event injection -------------------------------------------------

    def _process_events(self) -> None:
        sc = self.scenario
        t = self.state.time
        while self._events_processed < len(sc.events) and sc.events[self._events_processed].time <= t:
            ev = sc.events[self._events_processed]
            self._apply_event(ev)
            self._events_processed += 1

    def _apply_event(self, ev) -> None:
        state = self.state
        if ev.type == "UAV_FAILURE":
            uav = state.uavs.get(ev.data["uav_id"])
            if uav:
                uav.health = Health.FAILED
                self.log.emit(
                    state.time, "UAV_FAILURE_DETECTED", source=uav.id,
                    trigger="scenario_event", old_state=Health.OK.value,
                    new_state=Health.FAILED.value, reason="injected hard failure",
                )

        elif ev.type == "LINK_DEGRADATION":
            a, b = ev.data["link"]
            key = tuple(sorted((a, b)))
            state.degraded_links[key] = {
                "packet_loss": ev.data["packet_loss"],
                "expire_time": state.time + ev.data.get("duration", 60.0),
            }
            self.log.emit(
                state.time, "LINK_DEGRADED", source=f"{a},{b}",
                trigger="scenario_event", old_state="nominal", new_state="degraded",
                reason=f"packet_loss={ev.data['packet_loss']}",
            )

        elif ev.type == "NEW_PRIORITY_POI":
            pid = ev.data["poi_id"]
            state.pois[pid] = PoI(
                id=pid,
                position=list(ev.data["position"]),
                priority=ev.data["priority"],
                required_data=ev.data.get("required_data", ["imagery"]),
                deadline=ev.data.get("deadline", state.time + 120),
                risk=ev.data.get("risk", 0.3),
            )
            self.log.emit(
                state.time, "PRIORITY_ESCALATED", source=pid,
                trigger="scenario_event", old_state="none", new_state="UNVISITED",
                reason=f"new PoI injected with priority={ev.data['priority']}",
            )

        elif ev.type == "DYNAMICS_CHANGE":
            # true-vehicle parameter shift (thrust loss, payload, drag, wind).
            # Only meaningful with the dynamic flight model; ignored otherwise.
            if self.flight is not None:
                affected = self.flight.apply_change(ev.data)
                changes = {k: ev.data[k] for k in ("alpha", "mass", "drag_coeff", "wind") if k in ev.data}
                self.log.emit(
                    state.time, "DYNAMICS_CHANGE", source=",".join(affected),
                    trigger="scenario_event", old_state="nominal", new_state="perturbed",
                    reason=str(changes),
                )

        elif ev.type == "UAV_RETURN":
            uav = state.uavs.get(ev.data["uav_id"])
            if uav and uav.alive:
                old = uav.role
                uav.role = Role.RETURN_TO_BASE
                uav.current_task = None
                uav.target = list(uav.home_position)
                self.log.emit(
                    state.time, "RETURN_TO_BASE", source=uav.id,
                    trigger="scenario_event", old_state=old.value,
                    new_state=Role.RETURN_TO_BASE.value, reason="scheduled recall",
                )

    def _expire_degraded_links(self) -> None:
        t = self.state.time
        expired = [k for k, v in self.state.degraded_links.items() if v["expire_time"] <= t]
        for k in expired:
            del self.state.degraded_links[k]

    # ---- kinematics / energy ----------------------------------------------

    def _move(self, dt: float) -> None:
        if self.flight is not None:
            self.flight.advance(self.state, dt)
            for u in self.state.alive_uavs():
                if u.target and dist(u.position, u.target) < self.arrival_radius_m:
                    self._on_arrival(u)
            return
        for u in self.state.alive_uavs():
            if not u.target:
                u.velocity = [0.0, 0.0, 0.0]
                continue
            d = dist(u.position, u.target)
            step = min(u.max_speed * dt, d)
            if d < 1e-6:
                u.velocity = [0.0, 0.0, 0.0]
                continue
            direction = [(t - p) / d for p, t in zip(u.position, u.target)]
            u.position = [p + direction[i] * step for i, p in enumerate(u.position)]
            effective_speed = step / max(dt, 1e-5)
            u.velocity = [direction[i] * effective_speed for i in range(3)]

            if dist(u.position, u.target) < self.arrival_radius_m:
                self._on_arrival(u)

    def _on_arrival(self, u: UAVState) -> None:
        state = self.state
        if u.role == Role.RETURN_TO_BASE:
            u.role = Role.CHARGING
            u.target = None
        elif u.current_task and u.current_task in state.pois:
            poi = state.pois[u.current_task]
            if poi.status != PoIStatus.COMPLETE:
                poi.status = PoIStatus.COMPLETE
                poi.completed_at = state.time
                self.log.emit(
                    state.time, "TASK_ASSIGNMENT", source=u.id,
                    trigger="arrival", old_state=PoIStatus.ASSIGNED.value,
                    new_state=PoIStatus.COMPLETE.value, reason=f"survey of {poi.id} complete",
                )
            u.role = Role.IDLE
            u.current_task = None
            u.target = None

    def _update_link_predictions(self, dt: float) -> None:
        """Diagnostic-only: record the trained predictor's PDR estimate next
        to the actual PDR for every current link. Never consulted by
        relay.py/allocator.py - see the constructor docstring."""
        predictor = self.link_predictor
        if predictor is None:
            return
        state = self.state
        predictions = {}
        for (a, b), lk in state.links.items():
            prev_d = self._prev_link_distance.get((a, b), lk.distance)
            rate = (lk.distance - prev_d) / max(dt, 1e-6)
            ua = state.uavs.get(a)
            ub = state.uavs.get(b)
            comm_range = min(
                ua.comm_range if ua else lk.distance,
                ub.comm_range if ub else lk.distance,
            )
            predicted = predictor.predict_one(
                distance_now=lk.distance,
                distance_rate=rate,
                comm_range=comm_range,
                battery_a=ua.battery if ua else 100.0,
                battery_b=ub.battery if ub else 100.0,
                a_is_relay=1.0 if ua and ua.role in (Role.RELAY, Role.BACKUP_RELAY) else 0.0,
                b_is_relay=1.0 if ub and ub.role in (Role.RELAY, Role.BACKUP_RELAY) else 0.0,
            )
            predictions[(a, b)] = {"predicted_pdr": predicted, "actual_pdr": lk.pdr}
            self._prev_link_distance[(a, b)] = lk.distance
        self.link_quality_predictions = predictions

    def _update_energy(self, dt: float) -> None:
        sc = self.scenario
        for u in self.state.alive_uavs():
            if u.role == Role.CHARGING:
                u.battery = min(100.0, u.battery + CHARGE_RATE_PCT_PER_S * dt)
                if u.battery >= 100.0:
                    u.role = Role.IDLE
            else:
                rate = sc.battery_drain_rate
                if u.role in (Role.RELAY, Role.SURVEY_RELAY_HYBRID, Role.BACKUP_RELAY):
                    rate += sc.battery_drain_rate_relay_bonus
                u.battery = max(0.0, u.battery - rate * dt)

    # ---- main loop ----------------------------------------------------

    def step(self, dt: float | None = None) -> None:
        sc = self.scenario
        if dt is None:
            dt = sc.dt_seconds

        self.safety.begin_tick(self.state)
        self._process_events()
        self._expire_degraded_links()
        self._move(dt)
        self._update_energy(dt)

        self.safety.enforce_geofence(self.state, self.log)
        self.safety.check_separation(self.state, self.log)

        build_links(self.state)
        compute_connectivity(self.state)
        update_network_state(self.state)
        self._update_link_predictions(dt)

        self.fault_detector.detect(self.state, self.log)
        self.recovery_planner.update(self.state, dt, self.log)

        before_rtb = sum(1 for u in self.state.uavs.values() if u.role == Role.RETURN_TO_BASE)
        self.safety.enforce_battery_reserve(self.state, sc.battery_reserve_pct, self.log)
        after_rtb = sum(1 for u in self.state.uavs.values() if u.role == Role.RETURN_TO_BASE)
        self.battery_reserve_violations += max(0, after_rtb - before_rtb)

        self.task_allocator.assign(self.state, sc.battery_drain_rate, sc.battery_reserve_pct, self.log)
        self.relay_allocator.allocate(self.state, sc.battery_drain_rate, sc.battery_reserve_pct, self.log)

        self.metrics.record_step(self.state)
        self.state.time += dt

    def run(self) -> dict:
        sc = self.scenario
        dt = sc.dt_seconds
        steps = int(sc.duration_seconds / dt)

        for _ in range(steps):
            self.step(dt)

        return self.metrics.summarize(
            self.state,
            self.recovery_planner,
            dt,
            0.0,
            self.battery_reserve_violations,
        )

