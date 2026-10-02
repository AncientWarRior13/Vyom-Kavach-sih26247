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
    result_json TEXT NOT NULL
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
        conn.execute(SCHEMA)


def save_session(trainee_id: str, unit_id: str, difficulty: int, result: dict) -> int:
    with _conn() as conn:
        cur = conn.execute(
            """INSERT INTO sessions
               (trainee_id, unit_id, scenario_id, difficulty, overall_score,
                detection_score, classification_score, engagement_score,
                contacts_missed, result_json)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                trainee_id, unit_id, result["scenario_id"], difficulty,
                result["overall_score"], result["detection_score"],
                result["classification_score"], result["engagement_score"],
                result["contacts_missed"], json.dumps(result),
            ),
        )
        return cur.lastrowid


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
