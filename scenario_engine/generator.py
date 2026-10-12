"""Procedural scenario generator for the drone/swarm threat trainer.

Produces a scenario: a timed set of radar/visual contacts (single drones,
swarm members, decoys, birds) moving through a volume, with parameters for
time-of-day, terrain and sensor degradation that affect how noisy/late the
trainee's sensor picture is versus the hidden ground truth.

No external deps beyond the stdlib so this runs anywhere.
"""
from __future__ import annotations

import hashlib
import math
import random
import time
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

# additional detection difficulty contributed by environment, folded into
# `sensor_degradation` so terrain/time-of-day have a real effect on the
# sensor picture (dropout rate, position noise, and the ML classifier's
# feature noise) rather than being inert dropdowns.
TIME_OF_DAY_PENALTY = {"day": 0.0, "dusk": 0.15, "night": 0.35}
TERRAIN_PENALTY = {"rural": 0.0, "coastal": 0.05, "mountain": 0.15, "urban": 0.25}
# urban clutter and mountain terrain mask real threats behind extra decoys/birds
TERRAIN_EXTRA_CLUTTER = {"urban": 2, "mountain": 1, "coastal": 0, "rural": 0}


def effective_degradation(params: "ScenarioParams") -> float:
    env_penalty = TIME_OF_DAY_PENALTY.get(params.time_of_day, 0.0) + TERRAIN_PENALTY.get(params.terrain, 0.0)
    return max(0.0, min(1.0, params.sensor_degradation + env_penalty * 0.6))


# the point every hostile contact is implicitly converging on; also the
# sensor origin (range/azimuth in ml_bridge are already computed relative
# to this point) — made explicit here so scoring can compute time-to-impact
# and ROE zone checks against the same reference.
DEFENDED_ASSET_POS = (0.0, 0.0, 0.0)

NO_ENGAGE_ZONE_NAMES = ["hospital", "school", "civilian_housing", "marketplace"]


def _generate_no_engage_zones(difficulty: int) -> list[dict]:
    """1-2 no-engage zones (ROE feature 1): fixed areas where kinetic
    (hard_kill) engagement is prohibited regardless of threat level,
    matching real counter-UAS ROE constraints around protected sites."""
    n_zones = 1 if difficulty < 4 else 2
    names = random.sample(NO_ENGAGE_ZONE_NAMES, k=min(n_zones, len(NO_ENGAGE_ZONE_NAMES)))
    zones = []
    for name in names:
        angle = random.uniform(0, 2 * math.pi)
        dist = random.uniform(500, 2000)
        zones.append({
            "id": f"zone_{name}",
            "name": name,
            "center": [math.cos(angle) * dist, math.sin(angle) * dist],
            "radius_m": random.uniform(150, 400),
        })
    return zones


def _generate_roe_card(difficulty: int) -> dict:
    """The ROE feature's rules, generated once per scenario so they're
    fixed and inspectable (not learned/implicit) — the trainee can be shown
    this card, and scoring_engine checks engagement decisions against it
    exactly, same auditable-decision-tree philosophy as the rest of scoring."""
    return {
        "warn_before_hard_kill_required": True,
        "time_critical_override_s": 15.0,  # inside this ETA, hard_kill w/o prior warn is self-defense, not penalized
        "max_hard_kill_range_m": 800.0,
        "max_soft_kill_range_m": 1500.0,
        "no_engage_zones_block_hard_kill": True,
    }


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
    # roadmap feature 7 (adaptive red-team tactics) flags — ground truth
    # only; never sent to the client, consumed by ml_bridge.multi_sensor
    # (rf_silent) and sanitize_for_client (low_altitude_masked) and purely
    # informational for feint (the path itself already encodes the turn-away).
    rf_silent: bool = False
    low_altitude_masked: bool = False
    feint: bool = False
    # roadmap feature 13 (spoofed-signature scenarios): the contact_type
    # value whose radar signature this contact mimics (e.g. a decoy
    # spoofed_as "bird"). None = presents its own true signature.
    spoofed_as: str | None = None

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
    # adaptive-curriculum feature (research/PROTOCOLS.md-adjacent): optional
    # per-contact-type count multipliers, e.g. {"decoy": 2.0} to spawn more
    # decoys for a trainee whose AAR history shows they're weak on decoys.
    # Keys: "hostile", "decoy", "bird", "friendly". None/missing = 1.0x.
    type_weights: dict | None = field(default=None, compare=False)
    # roadmap feature 7: feints, swarm splitting, terrain-masked low
    # approaches, RF-silent drones, decoy saturation. Auto-enabled at
    # difficulty>=4 if not explicitly set, same pattern as swarm_attack.
    red_team_tactics: bool | None = None
    # roadmap feature 11 (team roles): seconds of comms delay between
    # detector -> classifier -> engagement authority. Only matters if the
    # scenario is played via /api/team/* endpoints; ignored in solo mode.
    team_comms_delay_s: float = 5.0


def _make_linear_path(t_offset: float, duration: float, speed: float, start=None, heading=None) -> list[Waypoint]:
    """Waypoint timestamps are absolute scenario time (t_offset..t_offset+duration),
    matching Contact.spawn_t/despawn_t, which are also absolute scenario time."""
    start = start or (random.uniform(-2000, 2000), random.uniform(-2000, 2000), random.uniform(50, 400))
    heading = heading if heading is not None else random.uniform(0, 2 * math.pi)
    vx, vy = math.cos(heading) * speed, math.sin(heading) * speed
    steps = max(2, int(duration // 5))
    return [
        Waypoint(t=t_offset + i * duration / steps, x=start[0] + vx * (i * duration / steps),
                  y=start[1] + vy * (i * duration / steps), z=start[2])
        for i in range(steps + 1)
    ]


def _make_feint_path(t_offset: float, duration: float, speed: float) -> list[Waypoint]:
    """Red-team tactic: approach the defended asset, then turn away before
    reaching effective engagement range — a fake attack meant to draw an
    early, wasted response."""
    start = (random.uniform(-2000, 2000), random.uniform(-2000, 2000), random.uniform(50, 400))
    heading_in = math.atan2(-start[1], -start[0])  # toward the asset at origin
    turn_point = 0.6  # fraction of the approach before peeling away
    steps = max(4, int(duration // 5))
    pts = []
    for i in range(steps + 1):
        frac = i / steps
        t = t_offset + frac * duration
        if frac <= turn_point:
            dist = speed * frac * duration
            x = start[0] + math.cos(heading_in) * dist
            y = start[1] + math.sin(heading_in) * dist
        else:
            dist_in = speed * turn_point * duration
            dist_out = speed * (frac - turn_point) * duration
            x = start[0] + math.cos(heading_in) * dist_in - math.cos(heading_in) * dist_out
            y = start[1] + math.sin(heading_in) * dist_in - math.sin(heading_in) * dist_out
        pts.append(Waypoint(t=t, x=x, y=y, z=start[2]))
    return pts


def _spawn_single_contact(contact_type: ContactType, window: tuple[float, float],
                           rf_silent: bool = False, low_altitude_masked: bool = False,
                           feint: bool = False, spoofed_as: str | None = None) -> Contact:
    spawn_t = random.uniform(*window)
    duration = random.uniform(20, 60)
    speed = {
        ContactType.HOSTILE_DRONE: random.uniform(8, 20),
        ContactType.HOSTILE_SWARM_MEMBER: random.uniform(8, 20),
        ContactType.FRIENDLY_DRONE: random.uniform(5, 15),
        ContactType.DECOY: random.uniform(3, 10),
        ContactType.BIRD: random.uniform(2, 8),
    }[contact_type]
    path = _make_feint_path(spawn_t, duration, speed) if feint else _make_linear_path(spawn_t, duration, speed)
    if low_altitude_masked:
        for w in path:
            w.z = random.uniform(5, 20)
    return Contact(
        contact_id=str(uuid.uuid4())[:8],
        contact_type=contact_type,
        true_threat_level=THREAT_DOCTRINE[contact_type],
        swarm_id=None,
        spawn_t=spawn_t,
        despawn_t=spawn_t + duration,
        path=path,
        rf_silent=rf_silent,
        low_altitude_masked=low_altitude_masked,
        feint=feint,
        spoofed_as=spoofed_as,
    )


def spawn_live_contact(contact_type_str: str, now_t: float) -> dict:
    """Roadmap feature 8 (instructor console): spawns a single contact
    starting right now, for live event injection into an already-running
    scenario — reuses the same contact generation logic as normal
    procedural spawning, just with a fixed (not randomized) spawn time.
    Returns a plain contact dict in the same schema as
    `generate_scenario`'s `contacts` entries.
    """
    contact_type = ContactType(contact_type_str)
    contact = _spawn_single_contact(contact_type, (now_t, now_t))
    return {
        "contact_id": contact.contact_id,
        "contact_type": contact.contact_type.value,
        "true_threat_level": contact.true_threat_level.value,
        "swarm_id": contact.swarm_id,
        "spawn_t": contact.spawn_t,
        "despawn_t": contact.despawn_t,
        "path": [asdict(w) for w in contact.path],
        "rf_silent": contact.rf_silent,
        "low_altitude_masked": contact.low_altitude_masked,
        "feint": contact.feint,
        "spoofed_as": contact.spoofed_as,
    }


def _spawn_swarm_ares(window: tuple[float, float], size: int, seed: int | None) -> tuple[list[Contact], dict] | None:
    """Try to drive swarm member paths from the real ares autonomy sim
    (allocator/relay/recovery) instead of straight lines. Returns
    (contacts, swarm_sim_meta) or None if ares isn't importable here, so
    the caller can fall back to the synthetic generator. `swarm_sim_meta`
    is what server.py needs later to deterministically replay-and-diverge
    the sim when the trainee actually engages a member (see
    ml_bridge.swarm_kinematics and server.py's /api/contact/engage).
    """
    try:
        from ml_bridge import generate_swarm_paths, pick_swarm_seed
    except Exception:
        return None

    spawn_t = random.uniform(*window)
    duration = random.uniform(40, 90)
    swarm_seed = pick_swarm_seed(seed)
    trajectories = generate_swarm_paths(size=size, duration_s=duration, spawn_t=spawn_t, seed=swarm_seed)
    if not trajectories:
        return None

    swarm_id = str(uuid.uuid4())[:8]
    contacts = []
    uav_to_contact = {}
    for uid, pts in trajectories.items():
        if not pts:
            continue
        contact_id = str(uuid.uuid4())[:8]
        uav_to_contact[uid] = contact_id
        path = [Waypoint(t=p["t"], x=p["x"], y=p["y"], z=p["z"]) for p in pts]
        contacts.append(Contact(
            contact_id=contact_id,
            contact_type=ContactType.HOSTILE_SWARM_MEMBER,
            true_threat_level=ThreatLevel.HIGH,
            swarm_id=swarm_id,
            spawn_t=path[0].t,
            despawn_t=path[-1].t,
            path=path,
        ))
    if not contacts:
        return None

    meta = {
        "seed": swarm_seed,
        "spawn_t": spawn_t,
        "duration_s": duration,
        "size": size,
        "uav_to_contact": uav_to_contact,
        "contact_to_uav": {v: k for k, v in uav_to_contact.items()},
        "injected_failures": [],  # list of [uav_id, local_time] applied so far
    }
    return contacts, meta


def _spawn_swarm_synthetic(window: tuple[float, float], size: int) -> list[Contact]:
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
        path = _make_linear_path(spawn_t, duration, base_speed + random.uniform(-1, 1), start=start, heading=heading)
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


def _spawn_swarm(window: tuple[float, float], size: int, seed: int | None = None) -> tuple[list[Contact], dict | None]:
    ares_result = _spawn_swarm_ares(window, size, seed)
    if ares_result is not None:
        return ares_result
    return _spawn_swarm_synthetic(window, size), None


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
    weights = params.type_weights or {}
    red_team = params.red_team_tactics if params.red_team_tactics is not None else (difficulty >= 4 and random.random() < 0.5)
    tactics_used: list[str] = []

    contacts: list[Contact] = []

    n_hostile = round(random.randint(1, difficulty) * weights.get("hostile", 1.0))
    for _ in range(max(0, n_hostile)):
        contacts.append(_spawn_single_contact(ContactType.HOSTILE_DRONE, window))

    if red_team and contacts:
        if random.random() < 0.5:
            random.choice(contacts).rf_silent = True
            tactics_used.append("rf_silent_drone")
        if random.random() < 0.5:
            random.choice(contacts).low_altitude_masked = True
            tactics_used.append("terrain_masked_low_approach")
        if random.random() < 0.4:
            contacts.append(_spawn_single_contact(ContactType.HOSTILE_DRONE, window, feint=True))
            tactics_used.append("feint")

    clutter = TERRAIN_EXTRA_CLUTTER.get(params.terrain, 0)

    decoy_saturation = red_team and difficulty >= 4
    if decoy_saturation:
        tactics_used.append("decoy_saturation")
    n_decoys = round((random.randint(0, difficulty) + clutter) * weights.get("decoy", 1.0) * (1.8 if decoy_saturation else 1.0))
    spoofed_one = False
    for _ in range(max(0, n_decoys)):
        # roadmap feature 13: one decoy per red-team scenario is crafted to
        # mimic a bird's radar signature (RadarSense's adversarial-evasion
        # framing) — its RF/visual truth is unchanged, only what the radar
        # classifier's synthesized signature presents as.
        spoof = "bird" if (red_team and not spoofed_one) else None
        if spoof:
            spoofed_one = True
            tactics_used.append("spoofed_signature")
        contacts.append(_spawn_single_contact(ContactType.DECOY, window, spoofed_as=spoof))

    n_birds = round((random.randint(0, difficulty + 1) + clutter) * weights.get("bird", 1.0))
    for _ in range(max(0, n_birds)):
        contacts.append(_spawn_single_contact(ContactType.BIRD, window))

    n_friendly = round(random.randint(0, 2) * weights.get("friendly", 1.0))
    for _ in range(max(0, n_friendly)):
        contacts.append(_spawn_single_contact(ContactType.FRIENDLY_DRONE, window))

    swarm_sim_meta = None
    swarm_sim_meta_2 = None
    if params.swarm_attack or (difficulty >= 3 and random.random() < 0.6):
        if red_team and random.random() < 0.4:
            # swarm splitting: two independent ares-driven subgroups
            # converging from different launch points instead of one
            half = max(2, (3 + difficulty * 2) // 2)
            group_a, swarm_sim_meta = _spawn_swarm(window, half, params.seed)
            seed_b = None if params.seed is None else params.seed + 9001
            group_b, swarm_sim_meta_2 = _spawn_swarm(window, half, seed_b)
            contacts.extend(group_a)
            contacts.extend(group_b)
            tactics_used.append("swarm_splitting")
        else:
            swarm_size = random.randint(3, 3 + difficulty * 2)
            swarm_contacts, swarm_sim_meta = _spawn_swarm(window, swarm_size, params.seed)
            contacts.extend(swarm_contacts)

    from ml_bridge.multi_sensor import pick_sensor_outage
    sensor_outage = pick_sensor_outage(difficulty)

    scenario_params = asdict(params)
    scenario_params["effective_sensor_degradation"] = effective_degradation(params)

    scenario = {
        "scenario_id": str(uuid.uuid4()),
        "params": scenario_params,
        "duration_s": duration_s,
        "defended_asset_position": list(DEFENDED_ASSET_POS),
        "no_engage_zones": _generate_no_engage_zones(difficulty),
        "roe_card": _generate_roe_card(difficulty),
        "sensor_outage": sensor_outage,
        "red_team_tactics_used": tactics_used,
        # roadmap feature 11 (team roles): a server-anchored start time so
        # multiple browser tabs (one per role) agree on "scenario elapsed
        # time" regardless of when each tab happened to load the page.
        "server_start_epoch_ms": time.time() * 1000,
        "team_comms_delay_s": params.team_comms_delay_s,
        "contacts": [
            {
                "contact_id": c.contact_id,
                "contact_type": c.contact_type.value,
                "true_threat_level": c.true_threat_level.value,
                "swarm_id": c.swarm_id,
                "spawn_t": c.spawn_t,
                "despawn_t": c.despawn_t,
                "path": [asdict(w) for w in c.path],
                "rf_silent": c.rf_silent,
                "low_altitude_masked": c.low_altitude_masked,
                "feint": c.feint,
                "spoofed_as": c.spoofed_as,
            }
            for c in contacts
        ],
    }
    if swarm_sim_meta is not None:
        scenario["swarm_sim_meta"] = swarm_sim_meta
    if swarm_sim_meta_2 is not None:
        scenario["swarm_sim_meta_2"] = swarm_sim_meta_2
    return scenario


def _deterministic_rng(*parts: object) -> random.Random:
    """A random.Random seeded deterministically from `parts`, independent of
    the global `random` module's state (and never mutates it). Used so
    sanitize_for_client()'s dropout/noise is a stable function of
    (scenario_id, contact_id, ...) rather than re-rolled on every call —
    see the fix note on sanitize_for_client for why that mattered.
    """
    key = "|".join(str(p) for p in parts)
    seed = int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**32)
    return random.Random(seed)


def sanitize_for_client(scenario: dict) -> dict:
    """Strip ground-truth fields so the trainee only gets what a sensor
    would show: positions over time, degraded per `sensor_degradation`.
    Classification/threat level must be judged by the trainee, not handed
    to them.

    Fix (external review finding #5): dropout and position noise used to
    come from the unseeded global `random` module, re-rolled on every call
    — including every 3s /api/scenario/{id}/refresh poll — so the same
    contact could flicker in and out of existence and its position could
    jitter discontinuously purely from being asked about again, with no
    new information. Both are now deterministic functions of
    (scenario_id, contact_id[, waypoint index]), so repeated sanitization
    of the same ground truth is stable: a contact that was visible stays
    visible (until something real changes it, e.g. an instructor's sensor
    jam), and its noisy position is a fixed offset from ground truth, not a
    moving target. This also means two team-mode tabs independently polling
    see the identical sensor picture for the same contact, as they should.
    """
    degradation = scenario["params"].get("effective_sensor_degradation", scenario["params"].get("sensor_degradation", 0.0))
    scenario_id = scenario["scenario_id"]
    out_contacts = []
    for c in scenario["contacts"]:
        # terrain-masked low approach (red-team tactic 7): ground clutter
        # makes a low flyer meaningfully more likely to be missed entirely
        dropout = degradation * (0.35 if c.get("low_altitude_masked") else 0.15)
        contact_rng = _deterministic_rng(scenario_id, c["contact_id"], "dropout")
        if degradation and contact_rng.random() < dropout:
            continue  # simulate a dropped/undetected contact entirely — but consistently so
        noisy_path = []
        for i, w in enumerate(c["path"]):
            wp_rng = _deterministic_rng(scenario_id, c["contact_id"], "noise", i)
            noise = degradation * wp_rng.uniform(-40, 40)
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
        # ROE/zones/asset/sensor-outage are information the trainee needs to
        # operate with, not ground truth about contacts — safe to send as-is.
        "defended_asset_position": scenario["defended_asset_position"],
        "no_engage_zones": scenario["no_engage_zones"],
        "roe_card": scenario["roe_card"],
        "sensor_outage": scenario.get("sensor_outage"),
        "server_start_epoch_ms": scenario.get("server_start_epoch_ms"),
        "team_comms_delay_s": scenario.get("team_comms_delay_s"),
        "contacts": out_contacts,
    }
