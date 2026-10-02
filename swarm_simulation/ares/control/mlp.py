"""Minimal numpy MLP with hand-written backprop and Adam.

Same reasoning as ares/link_predictor.py: the networks here are tiny
(a few thousand parameters), so numpy keeps the runtime dependency-free
(no torch) and the training reproducible from a seed. tanh hidden layers,
linear output.
"""
from __future__ import annotations

import numpy as np


class MLP:
    def __init__(self, sizes: list[int], rng: np.random.Generator | None = None,
                 weights: list[tuple[list, list]] | None = None):
        self.sizes = list(sizes)
        if weights is not None:
            self.W = [np.array(w, float) for w, _ in weights]
            self.b = [np.array(b, float) for _, b in weights]
        else:
            rng = rng or np.random.default_rng(0)
            self.W = [rng.normal(0.0, np.sqrt(1.0 / i), (i, o)) for i, o in zip(sizes[:-1], sizes[1:])]
            self.b = [np.zeros(o) for o in sizes[1:]]
        self._acts: list[np.ndarray] = []
        self.gW = [np.zeros_like(w) for w in self.W]
        self.gb = [np.zeros_like(b) for b in self.b]

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._acts = [x]
        h = x
        last = len(self.W) - 1
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            h = h @ w + b
            if i < last:
                h = np.tanh(h)
            self._acts.append(h)
        return h

    def backward(self, g_out: np.ndarray) -> np.ndarray:
        """Accumulates parameter grads (from the last `forward`) and returns
        the gradient w.r.t. the input."""
        g = g_out
        for i in range(len(self.W) - 1, -1, -1):
            a_in = self._acts[i]
            self.gW[i] = a_in.T @ g
            self.gb[i] = g.sum(axis=0)
            g = g @ self.W[i].T
            if i > 0:
                g = g * (1.0 - a_in ** 2)
        return g

    def params(self) -> list[np.ndarray]:
        return self.W + self.b

    def grads(self) -> list[np.ndarray]:
        return self.gW + self.gb

    def to_list(self) -> list[tuple[list, list]]:
        return [(w.tolist(), b.tolist()) for w, b in zip(self.W, self.b)]


class Adam:
    def __init__(self, params: list[np.ndarray], lr: float = 1e-3,
                 b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads: list[np.ndarray]) -> None:
        self.t += 1
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            mh = m / (1 - self.b1 ** self.t)
            vh = v / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)
