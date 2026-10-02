"""Session persistence for after-action review (AAR).

Plain sqlite3, stdlib only. One row per completed training session, plus
the full score breakdown as JSON so the AAR UI can drill into a single
session as well as show trends across many.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).parent / "aar.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trainee_id TEXT NOT NULL,
    unit_id TEXT NOT NULL,
    scenario_id TEXT NOT NULL,
    difficulty INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    overall_score REAL NOT NULL,
    detection_score REAL NOT NULL,
    classification_score REAL NOT NULL,
    engagement_score REAL NOT NULL,
    contacts_missed INTEGER NOT NULL,
    result_json TEXT NOT NULL,
    aar_ablation_arm TEXT,
    sequencing_arm TEXT,
    attempt_number INTEGER,
    scenario_json TEXT,
    trainee_log_json TEXT,
    chain_hash TEXT
);

CREATE TABLE IF NOT EXISTS experiment_assignments (
    experiment_name TEXT NOT NULL,
    trainee_id TEXT NOT NULL,
    arm TEXT NOT NULL,
    assigned_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (experiment_name, trainee_id)
);
"""


@contextmanager
def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _conn() as conn:
        conn.executescript(SCHEMA)
        existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()}
        for col in ("scenario_json", "trainee_log_json", "chain_hash"):
            if col not in existing_cols:
                conn.execute(f"ALTER TABLE sessions ADD COLUMN {col} TEXT")


def save_session(trainee_id: str, unit_id: str, difficulty: int, result: dict,
                  aar_ablation_arm: str | None = None, sequencing_arm: str | None = None,
                  scenario: dict | None = None, trainee_log: list | None = None) -> int:
    with _conn() as conn:
        attempt_number = conn.execute(
            "SELECT COUNT(*) + 1 FROM sessions WHERE trainee_id = ?", (trainee_id,)
        ).fetchone()[0]
        result_json = json.dumps(result)

        # roadmap feature 12 (offline/audit mode): hash-chained session log —
        # each row's hash covers its own data plus the previous row's hash,
        # so editing/deleting a past session (air-gapped, no central server
        # to cross-check against) breaks every hash after it, detectably.
        prev = conn.execute("SELECT chain_hash FROM sessions ORDER BY id DESC LIMIT 1").fetchone()
        prev_hash = prev["chain_hash"] if prev and prev["chain_hash"] else "GENESIS"
        # every denormalized column, not just result_json, so tampering with
        # any of them (not only the JSON blob) breaks the chain
        chain_payload = "|".join(str(v) for v in (
            prev_hash, trainee_id, unit_id, result["scenario_id"], difficulty, attempt_number,
            result["overall_score"], result["detection_score"], result["classification_score"],
            result["engagement_score"], result["contacts_missed"], result_json,
        ))
        chain_hash = hashlib.sha256(chain_payload.encode()).hexdigest()

        cur = conn.execute(
            """INSERT INTO sessions
               (trainee_id, unit_id, scenario_id, difficulty, overall_score,
                detection_score, classification_score, engagement_score,
                contacts_missed, result_json, aar_ablation_arm, sequencing_arm, attempt_number,
                scenario_json, trainee_log_json, chain_hash)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                trainee_id, unit_id, result["scenario_id"], difficulty,
                result["overall_score"], result["detection_score"],
                result["classification_score"], result["engagement_score"],
                result["contacts_missed"], result_json,
                aar_ablation_arm, sequencing_arm, attempt_number,
                json.dumps(scenario) if scenario is not None else None,
                json.dumps(trainee_log) if trainee_log is not None else None,
                chain_hash,
            ),
        )
        return cur.lastrowid


def verify_audit_chain() -> dict:
    """Walks every session in insertion order and recomputes each row's
    hash from scratch, confirming it matches what's stored and that it
    correctly chains from the previous row — detects any row that was
    edited, deleted-and-reinserted, or reordered after the fact."""
    with _conn() as conn:
        rows = conn.execute(
            """SELECT id, trainee_id, unit_id, scenario_id, difficulty, attempt_number,
                      overall_score, detection_score, classification_score, engagement_score,
                      contacts_missed, result_json, chain_hash
               FROM sessions ORDER BY id ASC"""
        ).fetchall()

    prev_hash = "GENESIS"
    for row in rows:
        if row["chain_hash"] is None:
            continue  # saved before this feature shipped; not part of the chain
        expected_payload = "|".join(str(v) for v in (
            prev_hash, row["trainee_id"], row["unit_id"], row["scenario_id"], row["difficulty"],
            row["attempt_number"], row["overall_score"], row["detection_score"],
            row["classification_score"], row["engagement_score"], row["contacts_missed"], row["result_json"],
        ))
        expected_hash = hashlib.sha256(expected_payload.encode()).hexdigest()
        if expected_hash != row["chain_hash"]:
            return {"valid": False, "broken_at_session_id": row["id"], "sessions_checked": len(rows)}
        prev_hash = row["chain_hash"]
    return {"valid": True, "broken_at_session_id": None, "sessions_checked": len(rows)}


def get_or_assign_arm(experiment_name: str, trainee_id: str, arms: list[str]) -> str:
    """Deterministic, persistent arm assignment: a trainee who returns for
    another session stays in the same experimental condition. Assignment is
    a stable hash of (experiment_name, trainee_id) over `arms`, recorded on
    first use so it's auditable later.
    """
    with _conn() as conn:
        row = conn.execute(
            "SELECT arm FROM experiment_assignments WHERE experiment_name = ? AND trainee_id = ?",
            (experiment_name, trainee_id),
        ).fetchone()
        if row:
            return row["arm"]
        import hashlib
        h = int(hashlib.sha256(f"{experiment_name}:{trainee_id}".encode()).hexdigest(), 16)
        arm = arms[h % len(arms)]
        conn.execute(
            "INSERT INTO experiment_assignments (experiment_name, trainee_id, arm) VALUES (?,?,?)",
            (experiment_name, trainee_id, arm),
        )
        return arm


def experiment_results(experiment_name: str, arm_column: str) -> list[dict]:
    """Raw per-session rows for a given experiment's arm column, for
    offline analysis (e.g. comparing overall_score across arms with a
    t-test/ANOVA outside this app)."""
    with _conn() as conn:
        rows = conn.execute(
            f"""SELECT trainee_id, unit_id, {arm_column} as arm, attempt_number,
                       overall_score, detection_score, classification_score,
                       engagement_score, difficulty, created_at
                FROM sessions WHERE {arm_column} IS NOT NULL
                ORDER BY trainee_id, attempt_number""",
        ).fetchall()
    return [dict(r) for r in rows]


_LIST_COLUMNS = """id, trainee_id, unit_id, scenario_id, difficulty, created_at,
                   overall_score, detection_score, classification_score, engagement_score,
                   contacts_missed, result_json, aar_ablation_arm, sequencing_arm, attempt_number"""


def list_sessions(trainee_id: str | None = None, unit_id: str | None = None, limit: int = 100) -> list[dict]:
    # excludes scenario_json/trainee_log_json (replay-only payloads) so a
    # routine AAR list stays small; use get_session_replay for those.
    query = f"SELECT {_LIST_COLUMNS} FROM sessions"
    clauses, params = [], []
    if trainee_id:
        clauses.append("trainee_id = ?")
        params.append(trainee_id)
    if unit_id:
        clauses.append("unit_id = ?")
        params.append(unit_id)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    with _conn() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def trainee_trend(trainee_id: str) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            """SELECT created_at, overall_score, detection_score,
                      classification_score, engagement_score, difficulty
               FROM sessions WHERE trainee_id = ? ORDER BY created_at ASC""",
            (trainee_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def unit_trend(unit_id: str) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            """SELECT trainee_id, AVG(overall_score) as avg_score, COUNT(*) as sessions
               FROM sessions WHERE unit_id = ? GROUP BY trainee_id ORDER BY avg_score DESC""",
            (unit_id,),
        ).fetchall()
    return [dict(r) for r in rows]


# roadmap feature 5: adaptive curriculum — maps contact_type (as scoring_engine
# names them) to the generator's type_weights key
_TYPE_WEIGHT_KEY = {
    "decoy": "decoy",
    "bird": "bird",
    "friendly_drone": "friendly",
    "hostile_drone": "hostile",
    "hostile_swarm_member": "hostile",
}


def weakness_profile(trainee_id: str, lookback: int = 10) -> dict:
    """Aggregates this trainee's last `lookback` sessions' per-contact
    scores by true contact type, and returns generator-ready type_weights:
    a type the trainee scores poorly on gets a multiplier > 1 so the next
    scenario spawns more of it (targeted practice instead of random
    repetition) — the approach roadmap feature 5 asks for, built on data
    this app already stores.
    """
    with _conn() as conn:
        rows = conn.execute(
            "SELECT result_json FROM sessions WHERE trainee_id = ? ORDER BY created_at DESC LIMIT ?",
            (trainee_id, lookback),
        ).fetchall()

    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in rows:
        result = json.loads(row["result_json"])
        for c in result.get("per_contact", []):
            key = _TYPE_WEIGHT_KEY.get(c.get("true_type"))
            if key is None:
                continue
            totals[key] = totals.get(key, 0.0) + c["contact_score"]
            counts[key] = counts.get(key, 0) + 1

    weights = {}
    for key, total in totals.items():
        avg_score_0_1 = (total / counts[key]) / 100.0
        # worse average -> higher weight, capped so one weak spot doesn't
        # crowd out everything else; no history for a type -> default 1.0
        weights[key] = round(1.0 + (1.0 - avg_score_0_1) * 1.5, 2)
    return weights


# roadmap feature 9: AAR replay with ghost track, heatmap, error taxonomy, export

def get_session_replay(session_id: int) -> dict | None:
    """Full ground-truth scenario + the trainee's raw logged actions for a
    past session, for timeline/ghost-track replay. Only sessions saved
    after this feature shipped have scenario_json/trainee_log_json —
    older rows return those as None, and the caller should say so rather
    than fabricate a replay for them.
    """
    with _conn() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["result"] = json.loads(d.pop("result_json"))
    d["scenario"] = json.loads(d["scenario_json"]) if d.get("scenario_json") else None
    d["trainee_log"] = json.loads(d["trainee_log_json"]) if d.get("trainee_log_json") else None
    d.pop("scenario_json", None)
    d.pop("trainee_log_json", None)
    return d


def error_taxonomy(trainee_id: str | None = None, unit_id: str | None = None, limit: int = 200) -> dict:
    """Counts of each scoring flag type across matching sessions (stripping
    the per-contact/zone-name suffix so e.g. all
    `roe_violation_no_engage_zone:hospital` entries bucket together)."""
    clauses, params = [], []
    if trainee_id:
        clauses.append("trainee_id = ?")
        params.append(trainee_id)
    if unit_id:
        clauses.append("unit_id = ?")
        params.append(unit_id)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with _conn() as conn:
        rows = conn.execute(
            f"SELECT result_json FROM sessions {where} ORDER BY created_at DESC LIMIT ?",
            (*params, limit),
        ).fetchall()

    counts: dict[str, int] = {}
    for row in rows:
        result = json.loads(row["result_json"])
        for flag in result.get("flags", []):
            flag_type = flag.split(":", 1)[-1].split(":")[0] if ":" in flag else flag
            counts[flag_type] = counts.get(flag_type, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def missed_positions(trainee_id: str | None = None, unit_id: str | None = None, limit: int = 50) -> list[dict]:
    """(x, y) of every missed contact at its spawn position, across recent
    sessions that have scenario_json saved — feeds the AAR heatmap of
    missed regions."""
    clauses, params = [], []
    if trainee_id:
        clauses.append("trainee_id = ?")
        params.append(trainee_id)
    if unit_id:
        clauses.append("unit_id = ?")
        params.append(unit_id)
    clauses.append("scenario_json IS NOT NULL")
    where = f"WHERE {' AND '.join(clauses)}"
    with _conn() as conn:
        rows = conn.execute(
            f"SELECT result_json, scenario_json FROM sessions {where} ORDER BY created_at DESC LIMIT ?",
            (*params, limit),
        ).fetchall()

    points = []
    for row in rows:
        result = json.loads(row["result_json"])
        scenario = json.loads(row["scenario_json"])
        contacts_by_id = {c["contact_id"]: c for c in scenario.get("contacts", [])}
        for c in result.get("per_contact", []):
            if c["detected"]:
                continue
            contact = contacts_by_id.get(c["contact_id"])
            if not contact or not contact.get("path"):
                continue
            p = contact["path"][0]
            points.append({"x": p["x"], "y": p["y"], "true_type": c["true_type"]})
    return points


def certification_status(trainee_id: str, threshold: float = 80.0, window: int = 5) -> dict:
    """Roadmap feature 10: pass/fail readiness grade — certified if the
    average overall_score across the trainee's last `window` sessions
    meets `threshold`."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT overall_score FROM sessions WHERE trainee_id = ? ORDER BY created_at DESC LIMIT ?",
            (trainee_id, window),
        ).fetchall()
    scores = [r["overall_score"] for r in rows]
    avg = round(sum(scores) / len(scores), 1) if scores else None
    return {
        "trainee_id": trainee_id,
        "sessions_counted": len(scores),
        "window": window,
        "threshold": threshold,
        "avg_score": avg,
        "certified": avg is not None and len(scores) >= window and avg >= threshold,
    }


def unit_certification_summary(unit_id: str, threshold: float = 80.0, window: int = 5) -> dict:
    with _conn() as conn:
        trainees = conn.execute(
            "SELECT DISTINCT trainee_id FROM sessions WHERE unit_id = ?", (unit_id,)
        ).fetchall()
    per_trainee = [certification_status(r["trainee_id"], threshold, window) for r in trainees]
    certified_count = sum(1 for t in per_trainee if t["certified"])
    return {
        "unit_id": unit_id,
        "trainees": per_trainee,
        "certified_count": certified_count,
        "total_trainees": len(per_trainee),
        "certification_rate": round(certified_count / len(per_trainee), 2) if per_trainee else None,
    }
