"""Implements the two arms of the scenario-sequencing (contextual
interference) experiment described in research/PROTOCOLS.md.

The contextual-interference literature (blocked vs. randomized practice
order) is mixed in applied settings and has never been tested for
threat-recognition/engagement-decision tasks specifically — this is the
open question `sequencing` lets this trainer actually answer, using data
it already collects.

- "blocked": difficulty increases monotonically with attempt number
  (1,1,2,2,3,3,4,4,5,5,...) — the conventional "ramp up" design.
- "randomized": difficulty for each attempt is drawn i.i.d. from 1-5,
  independent of attempt number — the contextual-interference condition.
"""
from __future__ import annotations

import random


def next_difficulty(arm: str, attempt_number: int, seed: int | None = None) -> int:
    if arm == "blocked":
        return min(5, max(1, (attempt_number - 1) // 2 + 1))
    if arm == "randomized":
        rng = random.Random(seed) if seed is not None else random
        return rng.randint(1, 5)
    raise ValueError(f"unknown sequencing arm: {arm!r}")
