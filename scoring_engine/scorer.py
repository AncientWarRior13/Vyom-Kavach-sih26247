"""Decision-tree scoring for a completed training session.

Grades a trainee's logged responses against a scenario's hidden ground
truth on three axes the problem statement asks for: detection time,
classification accuracy, and engagement decisions. No ML/model here on
purpose — this has to be auditable ("why did I lose points") for an AAR,
so it's a plain, inspectable decision tree.
"""
from __future__ import annotations

DETECTION_WINDOW_S = 20.0  # full credit if detected within this many seconds of spawn

WEIGHTS = {"detection": 0.25, "classification": 0.35, "engagement": 0.40}

CORRECT_ACTION = {
    "high": "engage",
    "low": "monitor",
    "none": "ignore",
}


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


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


def _score_engagement(contact: dict, log_entry: dict | None) -> tuple[float, str | None]:
    correct = CORRECT_ACTION[contact["true_threat_level"]]
    action = (log_entry or {}).get("engagement_action", "ignore")
    if action == correct:
        return 1.0, None
    if correct == "engage" and action != "engage":
        return 0.0, "threat_not_engaged"
    if correct in ("ignore", "monitor") and action == "engage":
        return 0.0, "wasted_or_fratricide_engagement"
    return 0.3, "suboptimal_response"


def score_session(scenario: dict, trainee_log: list[dict]) -> dict:
    """
    scenario: full ground-truth scenario dict (from scenario_engine.generate_scenario)
    trainee_log: list of {contact_id, detected_at, classification, engagement_action}
    """
    log_by_contact = {e["contact_id"]: e for e in trainee_log if "contact_id" in e}

    per_contact = []
    flags: list[str] = []
    for contact in scenario["contacts"]:
        entry = log_by_contact.get(contact["contact_id"])
        det = _score_detection(contact, entry)
        cls = _score_classification(contact, entry)
        eng, flag = _score_engagement(contact, entry)
        if flag:
            flags.append(f"{contact['contact_id']}:{flag}")
        contact_total = det * WEIGHTS["detection"] + cls * WEIGHTS["classification"] + eng * WEIGHTS["engagement"]
        per_contact.append({
            "contact_id": contact["contact_id"],
            "true_type": contact["contact_type"],
            "true_threat_level": contact["true_threat_level"],
            "detected": entry is not None and entry.get("detected_at") is not None,
            "detection_score": round(det, 3),
            "classification_score": round(cls, 3),
            "engagement_score": round(eng, 3),
            "contact_score": round(contact_total * 100, 1),
            "flag": flag,
        })

    n = len(per_contact) or 1
    detection_avg = sum(c["detection_score"] for c in per_contact) / n
    classification_avg = sum(c["classification_score"] for c in per_contact) / n
    engagement_avg = sum(c["engagement_score"] for c in per_contact) / n
    overall = (
        detection_avg * WEIGHTS["detection"]
        + classification_avg * WEIGHTS["classification"]
        + engagement_avg * WEIGHTS["engagement"]
    ) * 100

    return {
        "scenario_id": scenario["scenario_id"],
        "overall_score": round(overall, 1),
        "detection_score": round(detection_avg * 100, 1),
        "classification_score": round(classification_avg * 100, 1),
        "engagement_score": round(engagement_avg * 100, 1),
        "contacts_total": len(per_contact),
        "contacts_missed": sum(1 for c in per_contact if not c["detected"]),
        "critical_flags": [f for f in flags if "threat_not_engaged" in f],
        "flags": flags,
        "per_contact": per_contact,
    }
