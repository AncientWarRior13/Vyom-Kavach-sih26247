"""Procedural scenario generator for the drone/swarm threat trainer.

Produces a scenario: a timed set of radar/visual contacts (single drones,
swarm members, decoys, birds) moving through a volume, with parameters for
time-of-day, terrain and sensor degradation that affect how noisy/late the
trainee's sensor picture is versus the hidden ground truth.

No external deps beyond the stdlib so this runs anywhere.
"""
from __future__ import annotations

import math
import random
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum


class ContactType(str, Enum):
    HOSTILE_DRONE = "hostile_drone"
    HOSTILE_SWARM_MEMBER = "hostile_swarm_member"
    FRIENDLY_DRONE = "friendly_drone"
    DECOY = "decoy"
    BIRD = "bird"


class ThreatLevel(str, Enum):
    HIGH = "high"       # engage
    LOW = "low"          # monitor, do not engage
    NONE = "none"        # no engagement, non-military object


# ground-truth threat doctrine: what the *correct* engagement decision is
THREAT_DOCTRINE = {
    ContactType.HOSTILE_DRONE: ThreatLevel.HIGH,
    ContactType.HOSTILE_SWARM_MEMBER: ThreatLevel.HIGH,
    ContactType.FRIENDLY_DRONE: ThreatLevel.NONE,
    ContactType.DECOY: ThreatLevel.LOW,
    ContactType.BIRD: ThreatLevel.NONE,
}

TERRAINS = ["urban", "rural", "coastal", "mountain"]
TIMES_OF_DAY = ["day", "night", "dusk"]


@dataclass
class Waypoint:
    t: float
    x: float
    y: float
    z: float


@dataclass
class Contact:
    contact_id: str
    contact_type: ContactType
    true_threat_level: ThreatLevel
    swarm_id: str | None
    spawn_t: float
    despawn_t: float
    path: list[Waypoint] = field(default_factory=list)

    def position_at(self, t: float) -> Waypoint | None:
        if t < self.spawn_t or t > self.despawn_t:
            return None
        pts = self.path
        if t <= pts[0].t:
            return pts[0]
        if t >= pts[-1].t:
            return pts[-1]
        for a, b in zip(pts, pts[1:]):
            if a.t <= t <= b.t:
                f = 0 if b.t == a.t else (t - a.t) / (b.t - a.t)
                return Waypoint(
                    t=t,
                    x=a.x + (b.x - a.x) * f,
                    y=a.y + (b.y - a.y) * f,
                    z=a.z + (b.z - a.z) * f,
                )
        return pts[-1]


@dataclass
class ScenarioParams:
    difficulty: int = 1          # 1 (easy) .. 5 (hard)
    time_of_day: str = "day"
    terrain: str = "rural"
    sensor_degradation: float = 0.0   # 0..1, adds noise/dropout to trainee's picture
    swarm_attack: bool = False
    seed: int | None = None


def _make_linear_path(duration: float, speed: float, start=None, heading=None) -> list[Waypoint]:
    start = start or (random.uniform(-2000, 2000), random.uniform(-2000, 2000), random.uniform(50, 400))
    heading = heading if heading is not None else random.uniform(0, 2 * math.pi)
    vx, vy = math.cos(heading) * speed, math.sin(heading) * speed
    steps = max(2, int(duration // 5))
    return [
        Waypoint(t=i * duration / steps, x=start[0] + vx * (i * duration / steps),
                  y=start[1] + vy * (i * duration / steps), z=start[2])
        for i in range(steps + 1)
    ]


def _spawn_single_contact(contact_type: ContactType, window: tuple[float, float]) -> Contact:
    spawn_t = random.uniform(*window)
    duration = random.uniform(20, 60)
    speed = {
        ContactType.HOSTILE_DRONE: random.uniform(8, 20),
        ContactType.HOSTILE_SWARM_MEMBER: random.uniform(8, 20),
        ContactType.FRIENDLY_DRONE: random.uniform(5, 15),
        ContactType.DECOY: random.uniform(3, 10),
        ContactType.BIRD: random.uniform(2, 8),
    }[contact_type]
    path = _make_linear_path(duration, speed)
    return Contact(
        contact_id=str(uuid.uuid4())[:8],
        contact_type=contact_type,
        true_threat_level=THREAT_DOCTRINE[contact_type],
        swarm_id=None,
        spawn_t=spawn_t,
        despawn_t=spawn_t + duration,
        path=path,
    )


def _spawn_swarm(window: tuple[float, float], size: int) -> list[Contact]:
    swarm_id = str(uuid.uuid4())[:8]
    spawn_t = random.uniform(*window)
    duration = random.uniform(40, 90)
    heading = random.uniform(0, 2 * math.pi)
    base_speed = random.uniform(8, 18)
    base_start = (random.uniform(-2000, 2000), random.uniform(-2000, 2000), random.uniform(80, 300))
    contacts = []
    for i in range(size):
        jitter = (random.uniform(-80, 80), random.uniform(-80, 80), random.uniform(-20, 20))
        start = (base_start[0] + jitter[0], base_start[1] + jitter[1], base_start[2] + jitter[2])
        path = _make_linear_path(duration, base_speed + random.uniform(-1, 1), start=start, heading=heading)
        contacts.append(Contact(
            contact_id=str(uuid.uuid4())[:8],
            contact_type=ContactType.HOSTILE_SWARM_MEMBER,
            true_threat_level=ThreatLevel.HIGH,
            swarm_id=swarm_id,
            spawn_t=spawn_t,
            despawn_t=spawn_t + duration,
            path=path,
        ))
    return contacts


def generate_scenario(params: ScenarioParams | None = None, duration_s: int = 300) -> dict:
    """Generate a full scenario. Returns a plain-dict scenario (ground truth
    included) — callers that hand this to a trainee must strip
    contact_type/true_threat_level before sending it to the client; see
    `scenario_engine.sanitize_for_client`.
    """
    params = params or ScenarioParams()
    if params.seed is not None:
        random.seed(params.seed)

    difficulty = max(1, min(5, params.difficulty))
    window = (10, duration_s - 60)

    contacts: list[Contact] = []

    n_hostile = random.randint(1, difficulty)
    for _ in range(n_hostile):
        contacts.append(_spawn_single_contact(ContactType.HOSTILE_DRONE, window))

    n_decoys = random.randint(0, difficulty)
    for _ in range(n_decoys):
        contacts.append(_spawn_single_contact(ContactType.DECOY, window))

    n_birds = random.randint(0, difficulty + 1)
    for _ in range(n_birds):
        contacts.append(_spawn_single_contact(ContactType.BIRD, window))

    n_friendly = random.randint(0, 2)
    for _ in range(n_friendly):
        contacts.append(_spawn_single_contact(ContactType.FRIENDLY_DRONE, window))

    if params.swarm_attack or (difficulty >= 3 and random.random() < 0.6):
        swarm_size = random.randint(3, 3 + difficulty * 2)
        contacts.extend(_spawn_swarm(window, swarm_size))

    scenario = {
        "scenario_id": str(uuid.uuid4()),
        "params": asdict(params),
        "duration_s": duration_s,
        "contacts": [
            {
                "contact_id": c.contact_id,
                "contact_type": c.contact_type.value,
                "true_threat_level": c.true_threat_level.value,
                "swarm_id": c.swarm_id,
                "spawn_t": c.spawn_t,
                "despawn_t": c.despawn_t,
                "path": [asdict(w) for w in c.path],
            }
            for c in contacts
        ],
    }
    return scenario


def sanitize_for_client(scenario: dict) -> dict:
    """Strip ground-truth fields so the trainee only gets what a sensor
    would show: positions over time, degraded per `sensor_degradation`.
    Classification/threat level must be judged by the trainee, not handed
    to them.
    """
    degradation = scenario["params"].get("sensor_degradation", 0.0)
    out_contacts = []
    for c in scenario["contacts"]:
        if degradation and random.random() < degradation * 0.15:
            continue  # simulate a dropped/undetected contact entirely
        noisy_path = []
        for w in c["path"]:
            noise = degradation * random.uniform(-40, 40)
            noisy_path.append({"t": w["t"], "x": w["x"] + noise, "y": w["y"] + noise, "z": w["z"]})
        out_contacts.append({
            "contact_id": c["contact_id"],
            "spawn_t": c["spawn_t"],
            "despawn_t": c["despawn_t"],
            "path": noisy_path,
        })
    return {
        "scenario_id": scenario["scenario_id"],
        "params": {k: v for k, v in scenario["params"].items() if k != "seed"},
        "duration_s": scenario["duration_s"],
        "contacts": out_contacts,
    }
