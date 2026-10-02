"""Adaptive quadrotor controller with online system identification
(RMA-style; see docs/adaptive_control.md).

Three pieces:

* `AdaptiveModel`   - encoder E(history) -> latent z, forward-dynamics head
                      f(v, u, z) -> acceleration, and a residual policy
                      pi(error, v, a_des, z). Pure numpy inference.
* Policies          - `NominalPolicy` (PD assuming the nominal vehicle),
                      `OraclePolicy` (PD + exact model inversion, an upper
                      bound that cheats by reading the true parameters) and
                      `AdaptivePolicy` (nominal PD + learned residual on z).
                      All are batched: `act(p, v, goal) -> (u, z)`.
* `AdaptiveDroneController` - the per-UAV interface from SPEC-02 section 3.3.

The command `u` is an acceleration relative to hover in the nominal-vehicle
convention used by ares/dynamics.py.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from ..dynamics import clip_accel
from .mlp import MLP

HISTORY_LEN = 25          # 0.5 s at 50 Hz
FEATURES = 9              # v(3), u(3), observed accel(3) per transition
Z_DIM = 8
V_SCALE, U_SCALE, A_SCALE, E_SCALE = 10.0, 10.0, 10.0, 20.0
THETA_DIM = 5             # identifiable parameters: gain, drag, bias xyz
THETA_NAMES = ("gain", "drag", "bias_x", "bias_y", "bias_z")
E_CLIP = 60.0             # policy sees position error clipped to this norm (m)


@dataclass
class ControlConfig:
    dt: float = 0.02
    a_max: float = 10.0     # commanded-acceleration saturation, m/s^2
    v_max: float | np.ndarray = 6.0   # cruise speed limit, m/s (scalar, or (N, 1) per vehicle)
    lam: float = 1.5        # position error -> desired velocity gain
    kv: float = 4.0         # velocity error -> acceleration gain


def pd_accel(e: np.ndarray, v: np.ndarray, cfg: ControlConfig) -> np.ndarray:
    """Nominal velocity-limited PD law. `e` is goal - position."""
    v_des = clip_accel(cfg.lam * e, cfg.v_max)
    return cfg.kv * (v_des - v)


class HistoryBuffer:
    """Batched sliding window of the last HISTORY_LEN transitions
    (v_prev, u_prev, (v - v_prev) / dt), scaled for the networks."""

    def __init__(self, n: int, dt: float):
        self.n, self.dt = n, dt
        self.buf = np.zeros((n, HISTORY_LEN, FEATURES))
        self.count = 0
        self._prev_v: np.ndarray | None = None
        self._prev_u: np.ndarray | None = None

    def observe(self, v: np.ndarray) -> None:
        if self._prev_v is not None and self._prev_u is not None:
            trans = np.concatenate(
                [self._prev_v / V_SCALE, self._prev_u / U_SCALE,
                 (v - self._prev_v) / self.dt / A_SCALE], axis=1)
            self.buf[:, :-1] = self.buf[:, 1:]
            self.buf[:, -1] = trans
            self.count += 1
        self._prev_v = v.copy()

    def record_action(self, u: np.ndarray) -> None:
        self._prev_u = u.copy()

    @property
    def full(self) -> bool:
        return self.count >= HISTORY_LEN

    def window(self) -> np.ndarray:
        return self.buf.reshape(self.n, -1)


class AdaptiveModel:
    def __init__(self, z_dim: int = Z_DIM, hidden: int = 64,
                 rng: np.random.Generator | None = None, state: dict | None = None):
        self.z_dim, self.hidden = z_dim, hidden
        if state is not None:
            self.encoder = MLP([], weights=[tuple(x) for x in state["encoder"]])
            self.dynamics = MLP([], weights=[tuple(x) for x in state["dynamics"]])
            self.policy = MLP([], weights=[tuple(x) for x in state["policy"]])
            self.param_head = MLP([], weights=[tuple(x) for x in state["param_head"]])
            self.theta_mu = np.array(state["theta_mu"], float)
            self.theta_sd = np.array(state["theta_sd"], float)
        else:
            rng = rng or np.random.default_rng(0)
            self.encoder = MLP([HISTORY_LEN * FEATURES, hidden, hidden, z_dim], rng)
            self.dynamics = MLP([6 + z_dim, hidden, hidden, 3], rng)
            self.policy = MLP([9 + z_dim, hidden, hidden, 3], rng)
            # auxiliary readout z -> true parameters (privileged sim labels,
            # used only to shape z during training and to make it readable)
            self.param_head = MLP([z_dim, 32, THETA_DIM], rng)
            self.theta_mu = np.zeros(THETA_DIM)
            self.theta_sd = np.ones(THETA_DIM)
            # start the residual heads at zero so the untrained model is exactly nominal
            for net in (self.dynamics, self.policy):
                net.W[-1] *= 0.01

    def encode(self, window: np.ndarray) -> np.ndarray:
        return np.tanh(self.encoder.forward(window))

    def predict_accel(self, v: np.ndarray, u: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Forward dynamics f(s, u, z): nominal prediction (a = u) plus a
        learned, z-conditioned correction."""
        x = np.concatenate([v / V_SCALE, u / U_SCALE, z], axis=1)
        return u + self.dynamics.forward(x) * A_SCALE

    def residual(self, e: np.ndarray, v: np.ndarray, a_des: np.ndarray, z: np.ndarray) -> np.ndarray:
        # Training goals lie within a ~70 m arena; missions run for hundreds
        # of metres. The nominal PD law already saturates far from the goal,
        # so clipping the error the network sees keeps it in-distribution
        # without changing what the far-field command does.
        e = clip_accel(e, E_CLIP)
        x = np.concatenate([e / E_SCALE, v / V_SCALE, a_des / U_SCALE, z], axis=1)
        return self.policy.forward(x) * U_SCALE

    def estimate_params(self, z: np.ndarray) -> np.ndarray:
        """Readable parameter estimate (gain, drag, bias_x, bias_y, bias_z)
        from the latent. `bias` is wind plus the hover-thrust mismatch."""
        return self.param_head.forward(z) * self.theta_sd + self.theta_mu

    def to_dict(self) -> dict:
        return {"z_dim": self.z_dim, "hidden": self.hidden,
                "encoder": self.encoder.to_list(), "dynamics": self.dynamics.to_list(),
                "policy": self.policy.to_list(), "param_head": self.param_head.to_list(),
                "theta_mu": self.theta_mu.tolist(), "theta_sd": self.theta_sd.tolist()}

    def save(self, path: str, meta: dict | None = None) -> None:
        with open(path, "w") as f:
            json.dump({"meta": meta or {}, **self.to_dict()}, f)

    @staticmethod
    def load(path: str) -> "AdaptiveModel":
        with open(path) as f:
            d = json.load(f)
        return AdaptiveModel(z_dim=d["z_dim"], hidden=d["hidden"], state=d)


class NominalPolicy:
    """PD controller that believes the vehicle is exactly nominal."""

    def __init__(self, cfg: ControlConfig | None = None):
        self.cfg = cfg or ControlConfig()

    def reset(self, n: int) -> None:
        pass

    def act(self, p, v, goal):
        u = clip_accel(pd_accel(goal - p, v, self.cfg), self.cfg.a_max)
        return u, None


class OraclePolicy:
    """Upper bound: inverts the true dynamics using the true parameters.
    Never usable on a real vehicle - it exists to label training data and
    to bound what any adaptive scheme could achieve."""

    def __init__(self, params, cfg: ControlConfig | None = None):
        self.params, self.cfg = params, cfg or ControlConfig()

    def reset(self, n: int) -> None:
        pass

    def target_accel(self, p, v, goal) -> np.ndarray:
        """Unclipped model-inversion command."""
        prm = self.params
        a_des = pd_accel(goal - p, v, self.cfg)
        speed = np.linalg.norm(v, axis=-1, keepdims=True)
        return (a_des + prm.drag[:, None] * speed * v - prm.bias) / prm.gain[:, None]

    def act(self, p, v, goal):
        return clip_accel(self.target_accel(p, v, goal), self.cfg.a_max), None


class AdaptivePolicy:
    """Nominal PD plus a learned residual conditioned on the online latent z.
    Until the history window fills (0.5 s) it falls back to the nominal law."""

    def __init__(self, model: AdaptiveModel, cfg: ControlConfig | None = None):
        self.model, self.cfg = model, cfg or ControlConfig()
        self.hist: HistoryBuffer | None = None

    def reset(self, n: int) -> None:
        self.hist = HistoryBuffer(n, self.cfg.dt)

    def act(self, p, v, goal):
        n = len(p)
        if self.hist is None or self.hist.n != n:
            self.reset(n)
        self.hist.observe(v)
        e = goal - p
        a_des = pd_accel(e, v, self.cfg)
        if self.hist.full:
            z = self.model.encode(self.hist.window())
            u = a_des + self.model.residual(e, v, a_des, z)
        else:
            z = np.zeros((n, self.model.z_dim))
            u = a_des
        u = clip_accel(u, self.cfg.a_max)
        self.hist.record_action(u)
        return u, z


@dataclass
class UAVCommand:
    """Acceleration-relative-to-hover command for one UAV."""
    accel: np.ndarray = field(default_factory=lambda: np.zeros(3))


class AdaptiveDroneController:
    """Per-UAV interface from SPEC-02 section 3.3."""

    def __init__(self, model: AdaptiveModel | None = None, weights_path: str | None = None,
                 cfg: ControlConfig | None = None):
        if model is None:
            if weights_path is None:
                raise ValueError("provide a model or a weights_path")
            model = AdaptiveModel.load(weights_path)
        self.z_dim = model.z_dim
        self._policy = AdaptivePolicy(model, cfg)
        self._policy.reset(1)

    def reset(self) -> None:
        self._policy.reset(1)

    def estimate_params(self, z: np.ndarray) -> dict[str, float]:
        """Readable estimate from a latent returned by `step`. Thrust gain
        (thrust_scale / mass) and vertical bias are recovered well; drag
        moderately; horizontal wind only weakly (see docs/adaptive_control.md)."""
        est = self._policy.model.estimate_params(np.asarray(z, float)[None, :])[0]
        return dict(zip(THETA_NAMES, (float(x) for x in est)))

    def step(self, state, target_waypoint) -> tuple[UAVCommand, np.ndarray]:
        """`state` is an ares.state.UAVState (position, velocity used).
        Returns the command and the current latent parameter estimate z."""
        p = np.asarray(state.position, float)[None, :]
        v = np.asarray(state.velocity, float)[None, :]
        goal = np.asarray(target_waypoint, float)[None, :]
        u, z = self._policy.act(p, v, goal)
        return UAVCommand(accel=u[0]), z[0]
