"""Append-only decision log (SQLite). Every AI decision gets an ID; every human
action references it. The lab also keeps a separate oracle table with the true
labels — production systems never have this; the lab uses it to grade estimators."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from itertools import count

SCHEMA = """
CREATE TABLE decision_events (
    seq               INTEGER PRIMARY KEY,
    decision_id       TEXT UNIQUE NOT NULL,
    kind              TEXT NOT NULL CHECK (kind IN ('ai_draft','human_view','human_accept','human_override')),
    draft_decision_id TEXT,              -- human events point at the AI draft they act on
    subject_id        TEXT NOT NULL,     -- the document
    tenant_id         INTEGER NOT NULL,
    filename          TEXT NOT NULL,
    label             TEXT,
    confidence        REAL,
    held              INTEGER,
    hold_reason       TEXT,
    action            TEXT,              -- corrected | repick | bulk_set | accepted | view
    actor             TEXT
);
CREATE INDEX ix_events_kind ON decision_events(kind);
CREATE INDEX ix_events_tenant_file ON decision_events(tenant_id, filename);
CREATE INDEX ix_events_draft ON decision_events(draft_decision_id);
CREATE TRIGGER no_update BEFORE UPDATE ON decision_events
  BEGIN SELECT RAISE(ABORT, 'decision_events is append-only'); END;
CREATE TRIGGER no_delete BEFORE DELETE ON decision_events
  BEGIN SELECT RAISE(ABORT, 'decision_events is append-only'); END;

CREATE TABLE oracle_truth (subject_id TEXT PRIMARY KEY, true_label TEXT NOT NULL, is_twin INTEGER NOT NULL);
"""


@dataclass
class EventLog:
    path: str = ":memory:"
    conn: sqlite3.Connection = field(init=False)
    _ids: count = field(init=False, default_factory=lambda: count(1))

    def __post_init__(self) -> None:
        self.conn = sqlite3.connect(self.path)
        self.conn.executescript(SCHEMA)

    def __del__(self) -> None:
        try:
            self.conn.close()
        except Exception:  # noqa: BLE001 — interpreter shutdown
            pass

    def _next_id(self) -> str:
        return f"d_{next(self._ids):07d}"

    def append(self, kind: str, *, subject_id: str, tenant_id: int, filename: str,
               label: str | None = None, confidence: float | None = None,
               held: bool | None = None, hold_reason: str | None = None,
               draft_decision_id: str | None = None, action: str | None = None,
               actor: str | None = None) -> str:
        decision_id = self._next_id()
        self.conn.execute(
            "INSERT INTO decision_events (decision_id, kind, draft_decision_id, subject_id, tenant_id,"
            " filename, label, confidence, held, hold_reason, action, actor)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (decision_id, kind, draft_decision_id, subject_id, tenant_id, filename, label,
             confidence, None if held is None else int(held), hold_reason, action, actor),
        )
        return decision_id

    def record_truth(self, subject_id: str, true_label: str, is_twin: bool) -> None:
        self.conn.execute("INSERT INTO oracle_truth VALUES (?,?,?)", (subject_id, true_label, int(is_twin)))

    def q(self, sql: str, params: tuple = ()) -> list[tuple]:
        return self.conn.execute(sql, params).fetchall()

    def one(self, sql: str, params: tuple = ()):
        return self.conn.execute(sql, params).fetchone()[0]
