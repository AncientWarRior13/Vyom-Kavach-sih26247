"""Point-mass quadrotor plant with real-world parameter mismatch.

The autonomy stack commands an acceleration relative to hover (the flight
controller's thrust vector divided by the nominal mass, with gravity
compensation folded in). What the vehicle actually does depends on
parameters the planner does not know:

    thrust efficiency  alpha   (motor wear, low battery, prop damage)
    mass ratio         m / m0  (payload)
    quadratic drag     c_d     (F_drag = c_d |v| v)
    wind               w       (constant disturbance acceleration)

With commanded thrust T = m0 (g + u), the true acceleration is

    a = k (g + u) - g - d |v| v + w,   k = alpha / m,  d = c_d / m

which simplifies to  a = k u + (k - 1) g - d |v| v + w.  The (k - 1) g term
is why thrust loss or extra payload makes a nominal controller sag: it
hovers with the wrong thrust. Everything is batched over a leading axis so
one call steps thousands of randomised vehicles (training) or the whole
swarm (simulator integration).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GRAVITY = 9.81


def clip_accel(u: np.ndarray, a_max: float) -> np.ndarray:
    """Limit the magnitude of an acceleration command (actuator saturation)."""
    norm = np.linalg.norm(u, axis=-1, keepdims=True)
    scale = np.minimum(1.0, a_max / np.maximum(norm, 1e-9))
    return u * scale


@dataclass
class ParamRanges:
    """Sampling ranges for the randomised parameters. `NOMINAL` has none;
    `TRAIN` matches SPEC-02 section 3.1; `EVAL_15` is the +/-15% acceptance
    band (SPEC-02 section 5.1)."""
    alpha: tuple[float, float] = (1.0, 1.0)
    mass: tuple[float, float] = (1.0, 1.0)
    drag: tuple[float, float] = (0.0, 0.0)
    wind_xy: float = 0.0
    wind_z: float = 0.0


NOMINAL = ParamRanges()
TRAIN = ParamRanges(alpha=(0.7, 1.2), mass=(0.8, 1.3), drag=(0.0, 0.2), wind_xy=0.7, wind_z=0.3)
EVAL_15 = ParamRanges(alpha=(0.85, 1.15), mass=(0.85, 1.15), drag=(0.05, 0.15), wind_xy=0.5, wind_z=0.2)


@dataclass
class VehicleParams:
    """Batched true parameters. All arrays share length N."""
    alpha: np.ndarray
    mass: np.ndarray
    drag_coeff: np.ndarray
    wind: np.ndarray  # (N, 3)

    @property
    def n(self) -> int:
        return len(self.alpha)

    @property
    def gain(self) -> np.ndarray:
        return self.alpha / self.mass

    @property
    def drag(self) -> np.ndarray:
        return self.drag_coeff / self.mass

    @property
    def bias(self) -> np.ndarray:
        """Constant acceleration offset: wind plus the hover-thrust mismatch."""
        b = self.wind.copy()
        b[:, 2] += (self.gain - 1.0) * GRAVITY
        return b

    def theta(self) -> np.ndarray:
        """Identifiable parameter vector (gain, drag, bias xyz) - what a
        perfect latent `z` would have to encode."""
        return np.column_stack([self.gain, self.drag, self.bias])

    @staticmethod
    def nominal(n: int) -> "VehicleParams":
        return VehicleParams(np.ones(n), np.ones(n), np.zeros(n), np.zeros((n, 3)))

    @staticmethod
    def sample(n: int, rng: np.random.Generator, ranges: ParamRanges = TRAIN) -> "VehicleParams":
        wind = np.column_stack([
            rng.uniform(-ranges.wind_xy, ranges.wind_xy, n),
            rng.uniform(-ranges.wind_xy, ranges.wind_xy, n),
            rng.uniform(-ranges.wind_z, ranges.wind_z, n),
        ])
        return VehicleParams(
            alpha=rng.uniform(*ranges.alpha, n),
            mass=rng.uniform(*ranges.mass, n),
            drag_coeff=rng.uniform(*ranges.drag, n),
            wind=wind,
        )

    @staticmethod
    def single(alpha=1.0, mass=1.0, drag_coeff=0.0, wind=(0.0, 0.0, 0.0)) -> "VehicleParams":
        return VehicleParams(
            np.array([alpha], float), np.array([mass], float),
            np.array([drag_coeff], float), np.array([wind], float),
        )


class PointMassQuad:
    """Batched plant. `p`, `v` are (N, 3). `step(u)` advances one tick."""

    def __init__(self, params: VehicleParams, dt: float = 0.02, a_max: float = 10.0,
                 v_noise: float = 0.0, rng: np.random.Generator | None = None):
        self.params = params
        self.dt = dt
        self.a_max = a_max
        self.v_noise = v_noise
        self.rng = rng or np.random.default_rng(0)
        self.p = np.zeros((params.n, 3))
        self.v = np.zeros((params.n, 3))

    def reset(self, p: np.ndarray | None = None, v: np.ndarray | None = None) -> None:
        self.p = np.zeros((self.params.n, 3)) if p is None else np.array(p, float)
        self.v = np.zeros((self.params.n, 3)) if v is None else np.array(v, float)

    def accel(self, u: np.ndarray) -> np.ndarray:
        prm = self.params
        speed = np.linalg.norm(self.v, axis=-1, keepdims=True)
        return prm.gain[:, None] * u + prm.bias - prm.drag[:, None] * speed * self.v

    def step(self, u: np.ndarray) -> None:
        u = clip_accel(u, self.a_max)
        a = self.accel(u)
        self.v = self.v + a * self.dt
        self.p = self.p + self.v * self.dt

    def observe_velocity(self) -> np.ndarray:
        if self.v_noise <= 0.0:
            return self.v.copy()
        return self.v + self.rng.normal(0.0, self.v_noise, self.v.shape)
