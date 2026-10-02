"""Session persistence for after-action review (AAR).

Plain sqlite3, stdlib only. One row per completed training session, plus
the full score breakdown as JSON so the AAR UI can drill into a single
session as well as show trends across many.
"""
from __future__ import annotations

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
    attempt_number INTEGER
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


def save_session(trainee_id: str, unit_id: str, difficulty: int, result: dict,
                  aar_ablation_arm: str | None = None, sequencing_arm: str | None = None) -> int:
    with _conn() as conn:
        attempt_number = conn.execute(
            "SELECT COUNT(*) + 1 FROM sessions WHERE trainee_id = ?", (trainee_id,)
        ).fetchone()[0]
        cur = conn.execute(
            """INSERT INTO sessions
               (trainee_id, unit_id, scenario_id, difficulty, overall_score,
                detection_score, classification_score, engagement_score,
                contacts_missed, result_json, aar_ablation_arm, sequencing_arm, attempt_number)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                trainee_id, unit_id, result["scenario_id"], difficulty,
                result["overall_score"], result["detection_score"],
                result["classification_score"], result["engagement_score"],
                result["contacts_missed"], json.dumps(result),
                aar_ablation_arm, sequencing_arm, attempt_number,
            ),
        )
        return cur.lastrowid


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


def list_sessions(trainee_id: str | None = None, unit_id: str | None = None, limit: int = 100) -> list[dict]:
    query = "SELECT * FROM sessions"
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
