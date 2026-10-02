"""Structured, explainable decision log (section 33)."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class LogEntry:
    timestamp: float
    event_type: str
    source: str
    trigger: str
    old_state: str
    new_state: str
    reason: str
    confidence: Optional[float] = None


class EventLog:
    def __init__(self) -> None:
        self.entries: list[LogEntry] = []

    def emit(
        self,
        timestamp: float,
        event_type: str,
        source: str,
        trigger: str,
        old_state: str,
        new_state: str,
        reason: str,
        confidence: Optional[float] = None,
    ) -> None:
        self.entries.append(
            LogEntry(timestamp, event_type, source, trigger, old_state, new_state, reason, confidence)
        )

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump([asdict(e) for e in self.entries], f, indent=2)

    def to_csv(self, path: str) -> None:
        if not self.entries:
            with open(path, "w") as f:
                f.write("")
            return
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(self.entries[0]).keys()))
            writer.writeheader()
            for e in self.entries:
                writer.writerow(asdict(e))
