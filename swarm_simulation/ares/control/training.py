"""Training pipeline for the adaptive controller (numpy only).

Stage A  collect closed-loop data on randomised vehicles, driven by the
         oracle (true-parameter) controller plus exploration noise.
Stage B  fit encoder E and forward-dynamics head f jointly on
         L_dyn = || f(v_t, u_t, E(history_t)) - a_{t+1} ||^2 - the
         self-supervised objective from SPEC-02 section 3.2. z is grounded
         in the physics only because it must predict how the vehicle moves.
Stage C  freeze E; behaviour-clone the oracle's action into a residual
         policy on top of the nominal PD law, conditioned on z.
Stage D  DAgger: roll out the learned policy, relabel visited states with
         the oracle, retrain, so the policy is trained on its own state
         distribution rather than only the oracle's.
"""
from __future__ import annotations

import dataclasses
import time

import numpy as np

from ..dynamics import PointMassQuad, VehicleParams, ParamRanges, TRAIN, clip_accel
from .adaptive_controller import (
    AdaptiveModel, AdaptivePolicy, ControlConfig, HistoryBuffer, OraclePolicy,
    pd_accel, A_SCALE, U_SCALE, V_SCALE, E_SCALE,
)
from .mlp import Adam

ARENA_LO = np.array([-80.0, -80.0, -10.0])
ARENA_HI = np.array([80.0, 80.0, 20.0])
GOAL_PERIOD_STEPS = 400      # new random goal every 8 s (long enough to reach cruise speed)
V_MAX_RANGE = (3.0, 12.0)    # per-vehicle cruise-speed limit sampled in training, m/s
AUX_WEIGHT = 1.0             # weight of the z -> parameters auxiliary loss
V_NOISE = 0.003              # velocity measurement noise, m/s


def _uniform_points(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(ARENA_LO, ARENA_HI, (n, 3))


class Dataset:
    KEYS = ("win", "v", "u", "a_next", "e", "a_des", "u_oracle", "theta")

    def __init__(self):
        self._chunks: dict[str, list[np.ndarray]] = {k: [] for k in self.KEYS}

    def add(self, **arrays: np.ndarray) -> None:
        for k in self.KEYS:
            self._chunks[k].append(arrays[k])

    def concat(self) -> dict[str, np.ndarray]:
        return {k: np.concatenate(v, axis=0) for k, v in self._chunks.items() if v}

    def __len__(self) -> int:
        return sum(len(c) for c in self._chunks["v"])


def collect(params: VehicleParams, policy, n_steps: int, rng: np.random.Generator,
            cfg: ControlConfig, act_noise: float = 0.0, sample_p: float = 0.04,
            v_max_range: tuple[float, float] | None = V_MAX_RANGE) -> dict[str, np.ndarray]:
    """Roll `policy` out on a batch of vehicles and record supervised samples
    labelled by the oracle. Returns concatenated arrays (see Dataset.KEYS)."""
    n = params.n
    if v_max_range is not None:
        # every vehicle gets its own cruise-speed limit so the policy is
        # trained across the speeds the simulator will ask for
        cfg = dataclasses.replace(cfg, v_max=rng.uniform(*v_max_range, (n, 1)))
    policy.cfg = cfg
    plant = PointMassQuad(params, cfg.dt, cfg.a_max, V_NOISE, rng)
    plant.reset(p=_uniform_points(n, rng), v=rng.normal(0.0, 0.5, (n, 3)))
    oracle = OraclePolicy(params, cfg)
    policy.reset(n)
    hist = HistoryBuffer(n, cfg.dt)
    goal = _uniform_points(n, rng)
    ds = Dataset()

    for t in range(n_steps):
        if t % GOAL_PERIOD_STEPS == 0:
            goal = _uniform_points(n, rng)
        v_obs = plant.observe_velocity()
        hist.observe(v_obs)
        u, _ = policy.act(plant.p, v_obs, goal)
        if act_noise > 0.0:
            u = clip_accel(u + rng.normal(0.0, act_noise, u.shape), cfg.a_max)

        mask = hist.full & (rng.random(n) < sample_p)
        if mask.any():
            win = hist.window()[mask].copy()
            e = (goal - plant.p)[mask]
            vo = v_obs[mask]
            cfg_m = dataclasses.replace(cfg, v_max=cfg.v_max[mask]) if np.ndim(cfg.v_max) else cfg
            pend = dict(
                win=win, v=vo, u=u[mask], e=e,
                a_des=pd_accel(e, vo, cfg_m),
                u_oracle=clip_accel(oracle.target_accel(plant.p, v_obs, goal)[mask], cfg.a_max),
                theta=params.theta()[mask],
            )
        else:
            pend = None

        hist.record_action(u)
        plant.step(u)
        if pend is not None:
            v_next = plant.observe_velocity()[mask]
            pend["a_next"] = (v_next - pend["v"]) / cfg.dt
            ds.add(**pend)
    return ds.concat()


def _merge(*datasets: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: np.concatenate([d[k] for d in datasets], axis=0) for k in datasets[0]}


def train_dynamics(model: AdaptiveModel, data: dict[str, np.ndarray], rng: np.random.Generator,
                   epochs: int = 40, batch: int = 512, lr: float = 2e-3,
                   aux_weight: float = AUX_WEIGHT, log=print) -> tuple[float, float]:
    """Joint fit of encoder + dynamics head on L_dyn (SPEC-02 3.2), plus an
    auxiliary readout head z -> true parameters (aux_weight * L_aux, using the
    simulator's privileged labels) so that z is decodable, not just useful.
    Returns final train (dynamics MSE in (m/s^2)^2, aux MSE on standardised
    parameters)."""
    theta = data["theta"]
    model.theta_mu, model.theta_sd = theta.mean(0), theta.std(0) + 1e-9
    theta_n = (theta - model.theta_mu) / model.theta_sd
    opt = Adam(model.encoder.params() + model.dynamics.params() + model.param_head.params(), lr)
    n = len(data["v"])
    mse = aux = float("nan")
    for ep in range(epochs):
        opt.lr = lr * (0.5 * (1 + np.cos(np.pi * ep / epochs)) * 0.95 + 0.05)
        perm = rng.permutation(n)
        tot_dyn = tot_aux = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            z = model.encode(data["win"][idx])
            a_pred = model.predict_accel(data["v"][idx], data["u"][idx], z)
            diff = a_pred - data["a_next"][idx]
            tot_dyn += float((diff ** 2).sum())
            g_in = model.dynamics.backward(2.0 * diff * A_SCALE / diff.size)
            g_z = g_in[:, 6:]
            if aux_weight > 0.0:
                d_aux = model.param_head.forward(z) - theta_n[idx]
                tot_aux += float((d_aux ** 2).sum())
                g_z = g_z + model.param_head.backward(2.0 * aux_weight * d_aux / d_aux.size)
            else:
                model.param_head.gW = [np.zeros_like(w) for w in model.param_head.W]
                model.param_head.gb = [np.zeros_like(b) for b in model.param_head.b]
            model.encoder.backward(g_z * (1.0 - z ** 2))
            opt.step(model.encoder.grads() + model.dynamics.grads() + model.param_head.grads())
        mse, aux = tot_dyn / (n * 3), tot_aux / (n * theta.shape[1])
        if ep % 10 == 0 or ep == epochs - 1:
            log(f"  dyn epoch {ep:3d}  mse={mse:.4f}  aux={aux:.4f}")
    return mse, aux


def train_policy(model: AdaptiveModel, data: dict[str, np.ndarray], rng: np.random.Generator,
                 epochs: int = 40, batch: int = 512, lr: float = 2e-3, log=print) -> float:
    """Behaviour-clone the oracle's residual over the nominal PD law."""
    z_all = model.encode(data["win"])
    target = (data["u_oracle"] - data["a_des"]) / U_SCALE
    x_all = np.concatenate([data["e"] / E_SCALE, data["v"] / V_SCALE, data["a_des"] / U_SCALE, z_all], axis=1)
    opt = Adam(model.policy.params(), lr)
    n = len(x_all)
    loss = float("nan")
    for ep in range(epochs):
        opt.lr = lr * (0.5 * (1 + np.cos(np.pi * ep / epochs)) * 0.95 + 0.05)
        perm = rng.permutation(n)
        total = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            diff = model.policy.forward(x_all[idx]) - target[idx]
            total += float((diff ** 2).sum())
            model.policy.backward(2.0 * diff / diff.size)
            opt.step(model.policy.grads())
        loss = total / (n * 3)
        if ep % 10 == 0 or ep == epochs - 1:
            log(f"  policy epoch {ep:3d}  mse={loss:.5f}")
    return loss


def train(seed: int = 0, n_envs: int = 1500, n_steps: int = 1600, dyn_epochs: int = 40,
          policy_epochs: int = 40, dagger_rounds: int = 3, dagger_envs: int = 400,
          dagger_steps: int = 600, ranges: ParamRanges = TRAIN, log=print) -> tuple[AdaptiveModel, dict]:
    rng = np.random.default_rng(seed)
    cfg = ControlConfig()
    model = AdaptiveModel(rng=rng)
    t0 = time.time()

    log(f"[A] collecting {n_envs} envs x {n_steps} steps (oracle + noise)")
    params = VehicleParams.sample(n_envs, rng, ranges)
    data = collect(params, OraclePolicy(params, cfg), n_steps, rng, cfg, act_noise=1.0)
    log(f"    {len(data['v'])} samples ({time.time() - t0:.0f}s)")

    log("[B] encoder + forward-dynamics head")
    dyn_mse, aux_mse = train_dynamics(model, data, rng, epochs=dyn_epochs, log=log)

    log("[C] behaviour-cloning residual policy")
    pol_mse = train_policy(model, data, rng, epochs=policy_epochs, log=log)

    for r in range(dagger_rounds):
        log(f"[D] DAgger round {r + 1}/{dagger_rounds}")
        p2 = VehicleParams.sample(dagger_envs, rng, ranges)
        new = collect(p2, AdaptivePolicy(model, cfg), dagger_steps, rng, cfg, act_noise=0.0, sample_p=0.06)
        data = _merge(data, new)
        pol_mse = train_policy(model, data, rng, epochs=max(10, policy_epochs // 2), lr=1e-3, log=log)

    meta = {"seed": seed, "n_envs": n_envs, "n_steps": n_steps, "samples": int(len(data["v"])),
            "dyn_train_mse": dyn_mse, "aux_train_mse": aux_mse, "policy_train_mse": pol_mse,
            "dagger_rounds": dagger_rounds, "train_seconds": round(time.time() - t0, 1)}
    return model, meta
