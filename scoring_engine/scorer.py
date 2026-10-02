"""Decision-tree scoring for a completed training session.

Grades a trainee's logged responses against a scenario's hidden ground
truth on four axes: detection time, classification accuracy, a graduated
engagement/ROE response, and time-to-impact timeliness — plus a separately
reported (not score-weighted) AI-trust-calibration metric. No ML/model here
on purpose — this has to be auditable ("why did I lose points") for an AAR,
so it's a plain, inspectable decision tree; the ROE/range gates below are
deterministic thresholds, not probability rolls, for the same reason.

trainee_log entry schema (fields beyond `contact_id` are all optional,
degrading gracefully if missing so older/simpler callers still score):
{
    "contact_id": str,
    "detected_at": float | None,
    "classification": str | None,                 # one of ContactType values
    "engagement_action": "warn"|"soft_kill"|"hard_kill"|"monitor"|"ignore"
                          | "engage" (legacy alias for "hard_kill"),
    "engagement_at": float | None,                 # when engagement_action was taken
    "warn_at": float | None,                       # when a prior "warn" step was issued, if any
    "ai_hint_requested": bool,
    "ai_hint_label": str | None,                   # one of aircraft/anomaly/bird/drone
}
"""
from __future__ import annotations

import math

DETECTION_WINDOW_S = 20.0  # full credit if detected within this many seconds of spawn

WEIGHTS = {"detection": 0.20, "classification": 0.25, "engagement": 0.35, "timeliness": 0.20}

CORRECT_TIER = {
    "high": "hard_kill",   # may be resolved by a successful soft_kill instead, see _score_engagement
    "low": "soft_kill",    # warn/monitor also acceptable; hard_kill is overkill on a decoy
    "none": "monitor",     # ignore also acceptable; any kill action is fratricide/waste
}

# what the real ONNX sensor model (aircraft/anomaly/bird/drone) would be
# "right" to call for each ground-truth contact type — decoys are built to
# mimic a drone signature, so "drone" is the realistic sensor call on a
# decoy too, not a sensor error; "anomaly" never counts as a correct hint.
EXPECTED_SENSOR_LABEL = {
    "hostile_drone": "drone",
    "hostile_swarm_member": "drone",
    "friendly_drone": "drone",
    "decoy": "drone",
    "bird": "bird",
}

LEGACY_ACTION_ALIAS = {"engage": "hard_kill"}


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _position_at(path: list[dict], t: float) -> tuple[float, float, float]:
    if not path:
        return (0.0, 0.0, 0.0)
    if t <= path[0]["t"]:
        p = path[0]
        return (p["x"], p["y"], p["z"])
    if t >= path[-1]["t"]:
        p = path[-1]
        return (p["x"], p["y"], p["z"])
    for a, b in zip(path, path[1:]):
        if a["t"] <= t <= b["t"]:
            f = 0.0 if b["t"] == a["t"] else (t - a["t"]) / (b["t"] - a["t"])
            return (a["x"] + (b["x"] - a["x"]) * f, a["y"] + (b["y"] - a["y"]) * f, a["z"] + (b["z"] - a["z"]) * f)
    p = path[-1]
    return (p["x"], p["y"], p["z"])


def _closing_speed(path: list[dict], t: float, asset_xy: tuple[float, float]) -> float:
    """Positive = closing on the asset, negative = moving away. Estimated
    from a small forward finite difference along the known path."""
    dt = 1.0
    x0, y0, _ = _position_at(path, t)
    x1, y1, _ = _position_at(path, t + dt)
    d0 = math.hypot(x0 - asset_xy[0], y0 - asset_xy[1])
    d1 = math.hypot(x1 - asset_xy[0], y1 - asset_xy[1])
    return (d0 - d1) / dt


def time_to_impact(contact: dict, t: float, asset_xy: tuple[float, float] = (0.0, 0.0)) -> float | None:
    """Seconds until this contact reaches the defended asset if it keeps
    closing at its current rate, or None if it isn't closing (no meaningful
    "impact" to time against)."""
    x, y, _ = _position_at(contact["path"], t)
    dist = math.hypot(x - asset_xy[0], y - asset_xy[1])
    speed = _closing_speed(contact["path"], t, asset_xy)
    if speed <= 0.01:
        return None
    return dist / speed


def _range_at(contact: dict, t: float, asset_xy: tuple[float, float]) -> float:
    x, y, _ = _position_at(contact["path"], t)
    return math.hypot(x - asset_xy[0], y - asset_xy[1])


def _in_no_engage_zone(contact: dict, t: float, zones: list[dict]) -> dict | None:
    x, y, _ = _position_at(contact["path"], t)
    for zone in zones or []:
        cx, cy = zone["center"]
        if math.hypot(x - cx, y - cy) <= zone["radius_m"]:
            return zone
    return None


def _score_detection(contact: dict, log_entry: dict | None) -> float:
    if not log_entry or log_entry.get("detected_at") is None:
        return 0.0
    delay = log_entry["detected_at"] - contact["spawn_t"]
    if delay < 0:
        return 0.0
    return _clamp(1.0 - delay / DETECTION_WINDOW_S)


def _score_classification(contact: dict, log_entry: dict | None) -> float:
    if not log_entry or not log_entry.get("classification"):
        return 0.0
    guess = log_entry["classification"]
    truth = contact["contact_type"]
    if guess == truth:
        return 1.0
    hostile_truth = truth in ("hostile_drone", "hostile_swarm_member")
    hostile_guess = guess in ("hostile_drone", "hostile_swarm_member", "hostile")
    if hostile_truth == hostile_guess:
        return 0.5
    return 0.0


def _score_engagement(contact: dict, log_entry: dict | None, asset_xy: tuple[float, float],
                       no_engage_zones: list[dict], roe_card: dict) -> tuple[float, str | None]:
    """Graduated-response + ROE scoring (roadmap features 1 and 3).

    correct resolution per true_threat_level:
      high -> must reach hard_kill eventually, UNLESS a soft_kill was issued
              within its effective range (jam/spoof counts as a valid kill)
      low  -> warn, soft_kill, or monitor; hard_kill is wasteful overkill
      none -> monitor or ignore; any kill action is fratricide/waste

    ROE overlays on top of the above, independent of threat level:
      - hard_kill while the contact is inside a no-engage zone is always a
        violation, regardless of whether the target was hostile
      - hard_kill without a prior warn, when the card requires one AND the
        contact wasn't within the time-critical self-defense window, is a
        procedural violation (partial credit, not zero — the kill may still
        have been the materially correct outcome)
    """
    log_entry = log_entry or {}
    raw_action = log_entry.get("engagement_action", "ignore")
    action = LEGACY_ACTION_ALIAS.get(raw_action, raw_action)
    t = log_entry.get("engagement_at")
    if t is None:
        t = log_entry.get("detected_at", contact["spawn_t"])

    threat = contact["true_threat_level"]
    correct = CORRECT_TIER[threat]

    if action == "hard_kill":
        zone = _in_no_engage_zone(contact, t, no_engage_zones)
        if zone is not None and roe_card.get("no_engage_zones_block_hard_kill", True):
            return 0.0, f"roe_violation_no_engage_zone:{zone['name']}"

        if (roe_card.get("warn_before_hard_kill_required") and not log_entry.get("warn_at")
                and threat == "high"):
            eta = time_to_impact(contact, t, asset_xy)
            time_critical = eta is not None and eta <= roe_card.get("time_critical_override_s", 15.0)
            if not time_critical:
                return 0.6, "roe_violation_skipped_warn"

    if threat == "high":
        if action == "hard_kill":
            return 1.0, None
        if action == "soft_kill":
            rng = _range_at(contact, t, asset_xy)
            if rng <= roe_card.get("max_soft_kill_range_m", 1500.0):
                return 1.0, None
            return 0.3, "soft_kill_out_of_range"
        return 0.0, "threat_not_engaged"

    if threat == "low":
        if action in ("warn", "soft_kill", "monitor"):
            return 1.0, None
        if action == "hard_kill":
            return 0.3, "overkill_on_low_threat"
        return 0.5, "suboptimal_response"

    # threat == "none"
    if action in ("monitor", "ignore"):
        return 1.0, None
    return 0.0, "wasted_or_fratricide_engagement"


def _score_timeliness(contact: dict, log_entry: dict | None, asset_xy: tuple[float, float]) -> tuple[float, str | None]:
    """Roadmap feature 2: score decision latency against the threat's
    actual closing window instead of a fixed clock. Only meaningful for
    hostile contacts; everything else gets full credit (no impact window
    to be late against)."""
    if contact["true_threat_level"] != "high":
        return 1.0, None

    log_entry = log_entry or {}
    detected_at = log_entry.get("detected_at")
    resolved_at = log_entry.get("engagement_at")
    action = LEGACY_ACTION_ALIAS.get(log_entry.get("engagement_action", "ignore"), log_entry.get("engagement_action", "ignore"))

    if detected_at is None or resolved_at is None or action not in ("hard_kill", "soft_kill"):
        return 0.0, None  # not engaged at all; already flagged by engagement scoring

    eta_at_detection = time_to_impact(contact, detected_at, asset_xy)
    if eta_at_detection is None or eta_at_detection <= 0:
        return 1.0, None  # not actually closing on the asset; no real window to be late against

    if resolved_at >= detected_at + eta_at_detection:
        return 0.0, "threat_reached_asset"

    used_fraction = (resolved_at - detected_at) / eta_at_detection
    return _clamp(1.0 - used_fraction), None


def _ai_trust_category(contact: dict, log_entry: dict | None) -> str | None:
    """Roadmap feature 4: automation-bias categorization. Returns one of
    hint_correct_followed / hint_correct_overridden / hint_wrong_followed /
    hint_wrong_overridden, or None if no hint was requested for this contact."""
    log_entry = log_entry or {}
    if not log_entry.get("ai_hint_requested") or not log_entry.get("ai_hint_label"):
        return None
    hint_label = log_entry["ai_hint_label"]
    hint_correct = EXPECTED_SENSOR_LABEL.get(contact["contact_type"]) == hint_label

    classification = log_entry.get("classification")
    if not classification:
        return None
    trainee_matches_hint = EXPECTED_SENSOR_LABEL.get(classification, classification) == hint_label

    if hint_correct and trainee_matches_hint:
        return "hint_correct_followed"
    if hint_correct and not trainee_matches_hint:
        return "hint_correct_overridden"
    if not hint_correct and trainee_matches_hint:
        return "hint_wrong_followed"
    return "hint_wrong_overridden"


def score_session(scenario: dict, trainee_log: list[dict]) -> dict:
    """
    scenario: full ground-truth scenario dict (from scenario_engine.generate_scenario)
    trainee_log: list of dicts matching the schema documented at module top
    """
    log_by_contact = {e["contact_id"]: e for e in trainee_log if "contact_id" in e}
    asset_xy = tuple(scenario.get("defended_asset_position", [0.0, 0.0, 0.0])[:2])
    no_engage_zones = scenario.get("no_engage_zones", [])
    roe_card = scenario.get("roe_card", {
        "warn_before_hard_kill_required": False,
        "time_critical_override_s": 15.0,
        "max_soft_kill_range_m": 1500.0,
        "no_engage_zones_block_hard_kill": True,
    })

    per_contact = []
    flags: list[str] = []
    trust_counts = {"hint_correct_followed": 0, "hint_correct_overridden": 0,
                     "hint_wrong_followed": 0, "hint_wrong_overridden": 0}

    for contact in scenario["contacts"]:
        entry = log_by_contact.get(contact["contact_id"])
        det = _score_detection(contact, entry)
        cls = _score_classification(contact, entry)
        eng, eng_flag = _score_engagement(contact, entry, asset_xy, no_engage_zones, roe_card)
        tml, tml_flag = _score_timeliness(contact, entry, asset_xy)
        for f in (eng_flag, tml_flag):
            if f:
                flags.append(f"{contact['contact_id']}:{f}")

        trust_category = _ai_trust_category(contact, entry)
        if trust_category:
            trust_counts[trust_category] += 1

        contact_total = (det * WEIGHTS["detection"] + cls * WEIGHTS["classification"]
                         + eng * WEIGHTS["engagement"] + tml * WEIGHTS["timeliness"])
        per_contact.append({
            "contact_id": contact["contact_id"],
            "true_type": contact["contact_type"],
            "true_threat_level": contact["true_threat_level"],
            "detected": entry is not None and entry.get("detected_at") is not None,
            "detection_score": round(det, 3),
            "classification_score": round(cls, 3),
            "engagement_score": round(eng, 3),
            "timeliness_score": round(tml, 3),
            "contact_score": round(contact_total * 100, 1),
            "flag": eng_flag or tml_flag,
            "ai_trust_category": trust_category,
        })

    n = len(per_contact) or 1
    detection_avg = sum(c["detection_score"] for c in per_contact) / n
    classification_avg = sum(c["classification_score"] for c in per_contact) / n
    engagement_avg = sum(c["engagement_score"] for c in per_contact) / n
    timeliness_avg = sum(c["timeliness_score"] for c in per_contact) / n
    overall = (
        detection_avg * WEIGHTS["detection"]
        + classification_avg * WEIGHTS["classification"]
        + engagement_avg * WEIGHTS["engagement"]
        + timeliness_avg * WEIGHTS["timeliness"]
    ) * 100

    hints_requested = sum(trust_counts.values())
    automation_bias_score = (
        round(trust_counts["hint_wrong_followed"] / hints_requested, 3) if hints_requested else None
    )

    return {
        "scenario_id": scenario["scenario_id"],
        "overall_score": round(overall, 1),
        "detection_score": round(detection_avg * 100, 1),
        "classification_score": round(classification_avg * 100, 1),
        "engagement_score": round(engagement_avg * 100, 1),
        "timeliness_score": round(timeliness_avg * 100, 1),
        "contacts_total": len(per_contact),
        "contacts_missed": sum(1 for c in per_contact if not c["detected"]),
        "critical_flags": [f for f in flags if "threat_not_engaged" in f or "threat_reached_asset" in f
                            or "roe_violation" in f],
        "flags": flags,
        "ai_trust": {
            **trust_counts,
            "hints_requested": hints_requested,
            "automation_bias_score": automation_bias_score,
        },
        "per_contact": per_contact,
    }
