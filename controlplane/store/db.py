"""SQLite schema + DAO. Deliberately boring: no external service to stand up,
which means a governance reviewer can open the file with any SQLite client.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Any, Optional

from ..config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
  request_id TEXT PRIMARY KEY,
  ts REAL, tenant TEXT, use_case TEXT, geo TEXT, data_class TEXT,
  model TEXT, action TEXT, risk REAL, reward REAL,
  usd REAL, total_latency_ms INTEGER, check_latency_ms INTEGER,
  policy_version TEXT, degraded INTEGER, payload TEXT
);
CREATE INDEX IF NOT EXISTS idx_traces_ts ON traces(ts);
CREATE INDEX IF NOT EXISTS idx_traces_uc ON traces(use_case);

CREATE TABLE IF NOT EXISTS labels (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL, request_id TEXT, use_case TEXT, risk REAL, action TEXT,
  human_says_unsafe INTEGER, true_quality REAL
);
CREATE INDEX IF NOT EXISTS idx_labels_uc ON labels(use_case);

CREATE TABLE IF NOT EXISTS audit (
  idx INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL, request_id TEXT, prev_hash TEXT, row_hash TEXT, body TEXT
);
"""


class Database:
    def __init__(self, path: str | None = None):
        self.path = path or settings.db_path
        self.lock = threading.Lock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        with self.lock:
            self.conn.executescript(SCHEMA)
            # WAL keeps the dashboard readable while the gateway is writing.
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.commit()

    # ---------------- traces ----------------
    def save_trace(self, t: dict[str, Any]) -> None:
        """Stores flat columns for fast aggregation PLUS the full JSON payload,
        which is what makes counterfactual replay possible later."""
        req = t["request"]
        dec = t.get("decision") or {}
        risk = t.get("risk") or {}
        comp = t.get("completion") or {}
        with self.lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO traces VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    req["request_id"],
                    time.time(),
                    req["tenant"],
                    req["use_case"],
                    req["geo"],
                    req["data_class"],
                    t.get("chosen_model", ""),
                    dec.get("action", ""),
                    float(risk.get("fused", 0.0)),
                    float(t.get("reward") or 0.0),
                    float(comp.get("usd", 0.0)),
                    int(t.get("total_latency_ms", 0)),
                    int(t.get("check_latency_ms", 0)),
                    t.get("policy_version", ""),
                    int(bool(risk.get("degraded"))),
                    json.dumps(t),
                ),
            )
            self.conn.commit()

    def get_trace(self, request_id: str) -> Optional[sqlite3.Row]:
        cur = self.conn.execute(
            "SELECT * FROM traces WHERE request_id=?", (request_id,)
        )
        return cur.fetchone()

    def recent(self, limit: int = 200, use_case: str | None = None) -> list[dict]:
        q = "SELECT * FROM traces"
        args: tuple = ()
        if use_case:
            q += " WHERE use_case=?"
            args = (use_case,)
        q += " ORDER BY ts DESC LIMIT ?"
        cur = self.conn.execute(q, args + (limit,))
        return [dict(r) for r in cur.fetchall()]

    # ---------------- labels ----------------
    def add_label(
        self,
        request_id: str,
        use_case: str,
        risk: float,
        action: str,
        human_says_unsafe: bool,
        true_quality: float,
    ) -> None:
        with self.lock:
            self.conn.execute(
                """INSERT INTO labels
                   (ts, request_id, use_case, risk, action, human_says_unsafe, true_quality)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    time.time(),
                    request_id,
                    use_case,
                    risk,
                    action,
                    int(human_says_unsafe),
                    true_quality,
                ),
            )
            self.conn.commit()

    def labelled_rows(self, use_case: str | None = None) -> list[dict]:
        q = "SELECT * FROM labels"
        args: tuple = ()
        if use_case:
            q += " WHERE use_case=?"
            args = (use_case,)
        cur = self.conn.execute(q, args)
        return [
            {**dict(r), "human_says_unsafe": bool(r["human_says_unsafe"])}
            for r in cur.fetchall()
        ]


DB = Database()
