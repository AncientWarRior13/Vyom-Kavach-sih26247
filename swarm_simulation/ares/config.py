"""Loads config/*.yaml into the weight/threshold objects used across
ares/. Keeping this in one place means every tunable number the spec
calls out (sections 5, 8, 13, 19, 21) lives in a YAML file, not a
hardcoded constant, per section 37's "keep every important parameter
configurable" rule.
"""
from __future__ import annotations

import os
import yaml

CONFIG_DIR = os.environ.get(
    "ARES_CONFIG_DIR", os.path.join(os.path.dirname(__file__), "..", "config")
)


def _load(name: str) -> dict:
    path = os.path.join(CONFIG_DIR, name)
    with open(path) as f:
        return yaml.safe_load(f)


def load_role_weights():
    from .roles import RoleWeights

    d = _load("planner.yaml")["role_weights"]
    return RoleWeights(**d)


def load_resilience_weights():
    from .resilience import ResilienceWeights

    d = _load("planner.yaml")["resilience_weights"]
    return ResilienceWeights(**d)


def load_proactive_battery_margin() -> float:
    return _load("planner.yaml")["proactive_battery_margin"]


def load_max_relay_fraction() -> float:
    return _load("planner.yaml")["max_relay_fraction"]


def load_network_config() -> dict:
    return _load("network.yaml")


def load_safety_config() -> dict:
    return _load("safety.yaml")


def load_score_weights() -> dict:
    return _load("score.yaml")
