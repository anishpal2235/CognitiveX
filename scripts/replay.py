"""Counterfactual policy evaluation -- ZERO model calls.

The governance question nobody can usually answer: "if we tighten the support-bot
threshold from 0.60 to 0.45, what actually happens?"

Because every trace stores the full risk vector, we can re-run the DECISION
logic against a modified ladder offline and report exactly how many responses
change rung, plus the effect on FPR/FNR where labels exist. Policy changes stop
being a leap of faith and become a measured decision -- and it costs nothing.

Usage:
    python -m scripts.replay --use-case support_bot --scale 0.75
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from controlplane.act.decision import _from_ladder          # noqa: E402
from controlplane.policy.engine import resolve              # noqa: E402
from controlplane.store.db import DB                        # noqa: E402

FLAG = ("repair", "escalate", "block")


def main(use_case: str, scale: float, limit: int) -> None:
    rows = DB.recent(limit=limit, use_case=use_case)
    if not rows:
        print("no traces -- run scripts.simulate first")
        return

    pol = resolve(use_case, "IN", "confidential")

    # The counterfactual ladder: same shape, thresholds scaled.
    # scale < 1 tightens (flags more), scale > 1 loosens.
    new_ladder = [
        {"action": r["action"], "max_risk": min(1.01, float(r["max_risk"]) * scale)}
        for r in pol.ladder
    ]

    labels = {r["request_id"]: r["human_says_unsafe"] for r in DB.labelled_rows(use_case)}

    before, after, changed = Counter(), Counter(), 0
    cm = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    cm_new = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    for r in rows:
        trace = json.loads(r["payload"])
        risk = float(trace["risk"]["fused"])
        old_action = trace["decision"]["action"]
        new_action, _ = _from_ladder(new_ladder, risk)
        new_action = new_action.value

        before[old_action] += 1
        after[new_action] += 1
        if new_action != old_action:
            changed += 1

        rid = trace["request"]["request_id"]
        if rid in labels:
            unsafe = labels[rid]
            for bucket, act in ((cm, old_action), (cm_new, new_action)):
                flagged = act in FLAG
                if flagged and unsafe:
                    bucket["tp"] += 1
                elif flagged and not unsafe:
                    bucket["fp"] += 1
                elif not flagged and unsafe:
                    bucket["fn"] += 1
                else:
                    bucket["tn"] += 1

    def rates(c: dict) -> dict:
        return {
            "FPR": round(c["fp"] / max(c["fp"] + c["tn"], 1), 4),
            "FNR": round(c["fn"] / max(c["fn"] + c["tp"], 1), 4),
        }

    print(f"use case      {use_case}")
    print(f"traces        {len(rows)}")
    print(f"threshold x   {scale}")
    print(f"changed       {changed} ({100 * changed / len(rows):.1f}%)\n")
    print(f"before        {dict(before)}")
    print(f"after         {dict(after)}\n")

    if sum(cm.values()):
        print(f"labels        {sum(cm.values())}")
        print(f"before rates  {rates(cm)}  {cm}")
        print(f"after rates   {rates(cm_new)}  {cm_new}")
        print(
            "\nRead this as the cost of the change: tightening buys lower FNR "
            "and pays for it in FPR and reviewer load."
        )
    else:
        print("no labels for this use case -- rerun simulate with --label")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-case", default="support_bot")
    ap.add_argument("--scale", type=float, default=0.75)
    ap.add_argument("--limit", type=int, default=1000)
    a = ap.parse_args()
    main(a.use_case, a.scale, a.limit)
