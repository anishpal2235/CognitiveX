"""Hash-chained, tamper-evident audit log.

Governance is only real if the trail is hard to quietly rewrite. Each row stores
SHA-256(previous_hash + row_json), so every entry commits to the entire history
before it. Editing any past decision invalidates every hash after it, and
verify_chain() names the exact first broken index.

Append-only guarantees on plain SQLite: no blockchain, no extra infrastructure,
and extremely persuasive in a governance conversation.
"""
from __future__ import annotations

import hashlib
import json
import time

from ..store.db import DB


def _h(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def append(trace: dict) -> str:
    """The record deliberately stores the EVIDENCE from each detector (spans,
    similarity scores, cluster counts) plus the policy version -- exactly what
    is needed to answer "why was this blocked?" months later, when the model,
    the thresholds and the staff have all changed.
    """
    risk = trace.get("risk") or {}
    dec = trace.get("decision") or {}
    row = {
        "request_id": trace["request"]["request_id"],
        "ts": time.time(),
        "use_case": trace["request"]["use_case"],
        "geo": trace["request"]["geo"],
        "data_class": trace["request"]["data_class"],
        "policy_version": trace.get("policy_version"),
        "eligible_models": trace.get("eligible_models"),
        "chosen_model": trace.get("chosen_model"),
        "route_reason": trace.get("route_reason"),
        "signals": [
            {
                "detector": s["detector"],
                "dim": s["dim"],
                "score": s["score"],
                "verdict": s["verdict"],
                "evidence": s["evidence"],
            }
            for s in risk.get("signals", [])
        ],
        "fused_risk": risk.get("fused"),
        "action": dec.get("action"),
        "reason": dec.get("reason"),
        "reward": trace.get("reward"),
    }

    body = json.dumps(row, sort_keys=True)
    cur = DB.conn.execute("SELECT row_hash FROM audit ORDER BY idx DESC LIMIT 1")
    last = cur.fetchone()
    prev = last["row_hash"] if last else "GENESIS"
    rh = _h(prev + body)

    with DB.lock:
        DB.conn.execute(
            "INSERT INTO audit (ts, request_id, prev_hash, row_hash, body) "
            "VALUES (?,?,?,?,?)",
            (row["ts"], row["request_id"], prev, rh, body),
        )
        DB.conn.commit()
    return rh


def verify_chain() -> dict:
    """Returns the first broken index, or ok=True.

    Run it live in the dashboard: an integrity check an auditor can watch
    execute is worth more than a promise in a policy document.
    """
    prev = "GENESIS"
    cur = DB.conn.execute(
        "SELECT idx, prev_hash, row_hash, body FROM audit ORDER BY idx"
    )
    n = 0
    for r in cur.fetchall():
        n += 1
        if r["prev_hash"] != prev or _h(prev + r["body"]) != r["row_hash"]:
            return {"ok": False, "broken_at": r["idx"], "checked": n}
        prev = r["row_hash"]
    return {"ok": True, "records": n, "head": prev[:16]}
