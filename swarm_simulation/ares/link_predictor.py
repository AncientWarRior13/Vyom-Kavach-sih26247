"""Learned link-quality predictor (Phase B1, PROJECT_CONTEXT.md section 34
Option B: "predict near-future communication quality from position,
relative velocity, and history").

Predicts a link's PDR `horizon` ticks in the future from features
available now (current distance, its recent rate of change, comm range,
endpoint battery/role) - see scripts/generate_link_training_data.py for
why this has to be a genuine extrapolation task, not a re-derivation of
the deterministic current-distance PDR formula in ares/network.py.

Model: ridge-regularized polynomial regression (closed-form via numpy's
least-squares solver, not a full NN). This is a deliberate choice, not a
cut corner: a closed-form fit is exactly reproducible, has no training
instability or convergence question to argue about, and the feature set
here (7 inputs, low-order polynomial expansion) doesn't have enough
signal or data complexity to need a deep network. numpy is the one
non-stdlib dependency this module needs; the rest of ares/ deliberately
has none.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

FEATURE_NAMES = [
    "distance_now", "distance_rate", "comm_range",
    "battery_a", "battery_b", "a_is_relay", "b_is_relay",
]


def _expand_features(X: np.ndarray) -> np.ndarray:
    """Degree-2 polynomial expansion of the core distance/rate features
    (the ones with genuine nonlinear relationship to PDR - see
    ares/network.py's quadratic falloff), plus the raw linear features for
    everything else. No bias/intercept column here - that's added
    separately, unstandardized, in fit()/predict() (a constant column
    would have its mean subtracted to exactly zero by standardization,
    silently deleting the intercept from the model - this bit LinkQualityPredictor
    once already, see docs/link_predictor_analysis.md)."""
    distance_now = X[:, 0]
    distance_rate = X[:, 1]
    comm_range = X[:, 2]
    rest = X[:, 3:]

    # project forward: naive linear extrapolation of distance as a strong
    # engineered feature, since that's the actual mechanism generating the
    # label (distance changes roughly linearly over short horizons)
    ratio = np.divide(distance_now, np.maximum(comm_range, 1e-6))

    cols = [
        distance_now,
        distance_rate,
        distance_now ** 2,
        distance_rate ** 2,
        distance_now * distance_rate,
        ratio,
        ratio ** 2,
        comm_range,
    ]
    return np.column_stack(cols + [rest])


@dataclass
class LinkQualityPredictor:
    weights: np.ndarray | None = field(default=None, repr=False)
    feature_mean: np.ndarray | None = field(default=None, repr=False)
    feature_std: np.ndarray | None = field(default=None, repr=False)
    horizon: int = 5
    train_mse: float = float("nan")
    test_mse: float = float("nan")
    baseline_test_mse: float = float("nan")

    def fit(self, X: np.ndarray, y: np.ndarray, ridge_lambda: float = 1.0) -> None:
        Phi = _expand_features(X)
        # Standardize before ridge - without this, a uniform L2 penalty
        # shrinks large-scale columns (raw distance, ~0-200) and
        # small-scale ones (the [0,1] ratio features) by very different
        # effective amounts, which was silently producing a badly
        # regularized fit (confirmed: unstandardized fits scored worse
        # than the naive "distance unchanged" baseline on held-out data -
        # see docs/link_predictor_analysis.md). The bias/intercept term is
        # added separately below, not standardized.
        self.feature_mean = Phi.mean(axis=0)
        self.feature_std = Phi.std(axis=0)
        self.feature_std[self.feature_std < 1e-8] = 1.0
        Phi_std = (Phi - self.feature_mean) / self.feature_std
        Phi_std = np.column_stack([np.ones(len(Phi_std)), Phi_std])

        n_features = Phi_std.shape[1]
        reg = ridge_lambda * np.eye(n_features)
        reg[0, 0] = 0.0  # never regularize the intercept
        A = Phi_std.T @ Phi_std + reg
        b = Phi_std.T @ y
        self.weights = np.linalg.solve(A, b)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("predictor has not been fit or loaded yet")
        Phi = _expand_features(X)
        Phi_std = (Phi - self.feature_mean) / self.feature_std
        Phi_std = np.column_stack([np.ones(len(Phi_std)), Phi_std])
        return np.clip(Phi_std @ self.weights, 0.0, 1.0)

    def predict_one(self, distance_now: float, distance_rate: float, comm_range: float,
                     battery_a: float = 100.0, battery_b: float = 100.0,
                     a_is_relay: float = 0.0, b_is_relay: float = 0.0) -> float:
        X = np.array([[distance_now, distance_rate, comm_range, battery_a, battery_b, a_is_relay, b_is_relay]])
        return float(self.predict(X)[0])

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({
                "weights": self.weights.tolist(),
                "feature_mean": self.feature_mean.tolist(),
                "feature_std": self.feature_std.tolist(),
                "horizon": self.horizon,
                "train_mse": self.train_mse,
                "test_mse": self.test_mse,
                "baseline_test_mse": self.baseline_test_mse,
                "feature_names": FEATURE_NAMES,
            }, f, indent=2)

    @staticmethod
    def load(path: str) -> "LinkQualityPredictor":
        with open(path) as f:
            d = json.load(f)
        p = LinkQualityPredictor(
            weights=np.array(d["weights"]),
            feature_mean=np.array(d["feature_mean"]), feature_std=np.array(d["feature_std"]),
            horizon=d["horizon"],
            train_mse=d["train_mse"], test_mse=d["test_mse"], baseline_test_mse=d["baseline_test_mse"],
        )
        return p
