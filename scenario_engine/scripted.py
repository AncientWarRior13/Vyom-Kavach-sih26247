"""Scripted (instructor-authored, fixed) scenarios — the complement to
`generator.py`'s procedural generation. Each file under `scripted/` is a
full, hand-selected scenario (fixed contact count/types/paths/environment),
not re-randomized on every load; this is what the problem statement means
by "scripted scenarios" as distinct from "procedurally generated" ones.

Loading a drill reuses the same scenario/sanitize/score pipeline as a
procedural scenario — only the source of the contact list differs — so the
rest of the system (scoring, AAR, ML classifier, swarm sim) needs no
special-casing for scripted vs. procedural.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

SCRIPTED_DIR = Path(__file__).parent / "scripted"


def list_scripted() -> list[str]:
    return sorted(p.stem for p in SCRIPTED_DIR.glob("*.json"))


def load_scripted(name: str) -> dict:
    path = SCRIPTED_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"no scripted scenario named {name!r}")
    scenario = json.loads(path.read_text())
    scenario["scenario_id"] = str(uuid.uuid4())  # fresh session id per load; content stays fixed
    return scenario
