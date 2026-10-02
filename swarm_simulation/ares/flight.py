"""Opt-in dynamic flight model for the simulator (SPEC-02).

The default `Simulator` moves UAVs with a straight-line kinematic seek.
`DynamicFlightModel` replaces that movement layer with the point-mass
quadrotor in ares/dynamics.py driven by a flight controller
(`nominal` PD, `adaptive`, or the cheating `oracle` upper bound), so
thrust loss / payload / drag / wind actually change how the swarm flies.

Everything above the movement layer (allocator, relay, recovery, safety)
still just sets `uav.target`; nothing there knows this exists.

Each UAV's cruise speed is its scenario `max_speed`, capped at the fastest
speed the controllers were trained for (`V_MAX_TRAINED`, 12 m/s).
Deliberate simplification: battery drain is not coupled to thrust demand.
"""
from __future__ import annotations

import os

import numpy as np

from .control.adaptive_controller import (
    AdaptiveModel, AdaptivePolicy, ControlConfig, NominalPolicy, OraclePolicy, THETA_NAMES,
)
from .dynamics import PointMassQuad, VehicleParams
from .state import SwarmState

DEFAULT_WEIGHTS = os.path.join(os.path.dirname(__file__), "control", "adaptive_weights.json")
PLANT_DT = 0.02          # 50 Hz inner loop
VELOCITY_NOISE = 0.003   # m/s, matches training
GROUND_Z = 0.0
V_MAX_TRAINED = 12.0     # top of the speed range in ares/control/training.py


class DynamicFlightModel:
    CONTROLLERS = ("nominal", "adaptive", "oracle")

    def __init__(self, state: SwarmState, controller: str = "adaptive",
                 weights_path: str | None = None, vehicle_defaults: dict | None = None,
                 seed: int = 0):
        if controller not in self.CONTROLLERS:
            raise ValueError(f"controller must be one of {self.CONTROLLERS}, got {controller!r}")
        self.controller = controller
        self.ids = list(state.uavs)
        self._index = {uid: i for i, uid in enumerate(self.ids)}
        n = len(self.ids)

        v_max = np.array([[min(state.uavs[u].max_speed, V_MAX_TRAINED)] for u in self.ids])
        self.cfg = ControlConfig(dt=PLANT_DT, v_max=v_max)
        self.params = VehicleParams.nominal(n)
        self.plant = PointMassQuad(self.params, PLANT_DT, self.cfg.a_max, VELOCITY_NOISE,
                                   np.random.default_rng(seed))
        self.plant.reset(p=np.array([state.uavs[u].position for u in self.ids], float))

        if controller == "nominal":
            self.policy = NominalPolicy(self.cfg)
        elif controller == "oracle":
            self.policy = OraclePolicy(self.params, self.cfg)
        else:
            self.policy = AdaptivePolicy(AdaptiveModel.load(weights_path or DEFAULT_WEIGHTS), self.cfg)
        self.policy.reset(n)

        self.hold: dict[str, np.ndarray] = {}
        self.latent: dict[str, list[float]] = {}
        # readable parameter estimate per UAV (adaptive controller only):
        # {uav_id: {gain, drag, bias_x, bias_y, bias_z}}
        self.param_estimate: dict[str, dict[str, float]] = {}
        if vehicle_defaults:
            self.apply_change(vehicle_defaults)

    def apply_change(self, change: dict) -> list[str]:
        """Change true vehicle parameters. Keys: uav_id (omit for all UAVs),
        alpha (thrust efficiency), mass (ratio to nominal), drag_coeff,
        wind [x, y, z] (m/s^2). Returns the affected UAV ids."""
        uid = change.get("uav_id")
        if uid in (None, "all"):
            sel = np.arange(len(self.ids))
        elif uid in self._index:
            sel = np.array([self._index[uid]])
        else:
            return []
        if "alpha" in change:
            self.params.alpha[sel] = change["alpha"]
        if "mass" in change:
            self.params.mass[sel] = change["mass"]
        if "drag_coeff" in change:
            self.params.drag_coeff[sel] = change["drag_coeff"]
        if "wind" in change:
            self.params.wind[sel] = np.asarray(change["wind"], float)
        return [self.ids[i] for i in sel]

    def advance(self, state: SwarmState, dt: float) -> None:
        """Advance every UAV by `dt` seconds of 50 Hz control, then write
        position/velocity back to the swarm state."""
        uavs = [state.uavs[u] for u in self.ids]
        alive = np.array([u.alive for u in uavs])
        # the swarm state is the source of truth for position (safety code
        # may have clamped it); the plant owns velocity.
        self.plant.p = np.array([u.position for u in uavs], float)

        goals = np.empty_like(self.plant.p)
        for i, u in enumerate(uavs):
            if u.target:
                self.hold.pop(u.id, None)
                goals[i] = u.target
            else:
                goals[i] = self.hold.setdefault(u.id, self.plant.p[i].copy())

        frozen_p, frozen_v = self.plant.p.copy(), self.plant.v.copy()
        z = None
        for _ in range(max(1, round(dt / PLANT_DT))):
            u_cmd, z = self.policy.act(self.plant.p, self.plant.observe_velocity(), goals)
            self.plant.step(u_cmd)
            # ground contact: a sagging UAV parked at home rests on the
            # ground instead of sinking through it
            below = self.plant.p[:, 2] < GROUND_Z
            self.plant.p[below, 2] = GROUND_Z
            self.plant.v[below, 2] = np.maximum(self.plant.v[below, 2], 0.0)
            # a failed UAV stops being controlled and stays where it was
            self.plant.p[~alive] = frozen_p[~alive]
            self.plant.v[~alive] = frozen_v[~alive]

        for i, u in enumerate(uavs):
            if not alive[i]:
                continue
            u.position = self.plant.p[i].tolist()
            u.velocity = self.plant.v[i].tolist()
        if z is not None:
            self.latent = {uid: z[i].tolist() for i, uid in enumerate(self.ids)}
            if isinstance(self.policy, AdaptivePolicy) and self.policy.hist.full:
                est = self.policy.model.estimate_params(z)
                self.param_estimate = {
                    uid: dict(zip(THETA_NAMES, (round(float(x), 3) for x in est[i])))
                    for i, uid in enumerate(self.ids)}
