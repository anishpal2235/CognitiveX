"""The two feedback channels that make the system improve with use.

A. IMPLICIT (every request): Check scores -> reward -> LinUCB update.
   Free and immediate, but proxy-based.
B. EXPLICIT (human overrides and labels): a reviewer's "this flag was wrong" is
   gold-standard supervision. It (i) re-updates the bandit with the true label
   and (ii) accumulates into the threshold tuner below.

Crucially, the tuner PROPOSES thresholds and never applies them. A guardrail
system that silently retunes its own risk appetite is ungovernable.
"""
from __future__ import annotations

import json

import numpy as np

from ..route.features import featurize
from ..route.router import router
from ..schemas import InterceptedRequest
from ..store.db import DB


class Learner:
    def apply_human_label(
        self,
        request_id: str,
        was_correct_flag: bool,
        true_quality: float | None = None,
    ) -> dict:
        row = DB.get_trace(request_id)
        if not row:
            return {"ok": False, "reason": "unknown_request_id"}

        trace = json.loads(row["payload"])
        req = InterceptedRequest(**trace["request"])
        x = featurize(req)
        model = trace.get("chosen_model", "")
        prior_reward = float(trace.get("reward") or 0.0)
        action = (trace.get("decision") or {}).get("action", "")

        # A human label is worth more than a proxy, so it dominates the blend.
        if true_quality is None:
            true_quality = 1.0 if was_correct_flag else 0.0
        corrected = float(
            np.clip(0.35 * prior_reward + 0.65 * (true_quality * 2 - 1), -1.0, 1.0)
        )

        if model:
            router.bandit.update(model, x, corrected)
            router.bandit.save(router.state_path)

        # Translate "was the flag correct?" into "was the output actually unsafe?",
        # which is the label the metrics module needs. The mapping depends on what
        # the system DID: agreeing with an allow means safe, agreeing with a block
        # means unsafe.
        if action in ("allow", "annotate"):
            human_says_unsafe = not was_correct_flag
        else:
            human_says_unsafe = was_correct_flag

        DB.add_label(
            request_id=request_id,
            use_case=req.use_case.value,
            risk=float((trace.get("risk") or {}).get("fused", 0.0)),
            action=action,
            human_says_unsafe=human_says_unsafe,
            true_quality=float(true_quality),
        )
        return {
            "ok": True,
            "corrected_reward": round(corrected, 3),
            "model": model,
            "human_says_unsafe": human_says_unsafe,
        }

    def suggest_thresholds(self, use_case: str, target_fpr: float = 0.05) -> dict:
        """Data-driven retuning of the ladder.

        We sweep the flag boundary over every human-labelled trace and return the
        threshold with the lowest false-negative rate that still respects the
        target false positive rate. This turns "tune the over/under-flagging
        tradeoff" from a vibe into a reproducible number a governance board can
        sign off on.

        Deliberately advisory: the recommendation is applied by a human editing
        configs/policies.yaml, which keeps the change reviewable and versioned.
        """
        rows = DB.labelled_rows(use_case)
        if len(rows) < 20:
            return {"ok": False, "reason": f"need >=20 labels, have {len(rows)}"}

        best = None
        curve = []
        for thr in [i / 100 for i in range(5, 96, 5)]:
            fp = sum(1 for r in rows if r["risk"] >= thr and not r["human_says_unsafe"])
            tn = sum(1 for r in rows if r["risk"] < thr and not r["human_says_unsafe"])
            fn = sum(1 for r in rows if r["risk"] < thr and r["human_says_unsafe"])
            tp = sum(1 for r in rows if r["risk"] >= thr and r["human_says_unsafe"])
            fpr = fp / max(fp + tn, 1)
            fnr = fn / max(fn + tp, 1)
            curve.append({"thr": thr, "fpr": round(fpr, 3), "fnr": round(fnr, 3)})
            if fpr <= target_fpr and (best is None or fnr < best["fnr"]):
                best = {"threshold": thr, "fpr": round(fpr, 3), "fnr": round(fnr, 3)}

        return {
            "ok": bool(best),
            "n_labels": len(rows),
            "target_fpr": target_fpr,
            "recommended": best,
            "operating_curve": curve,
            "note": "advisory only -- apply by editing the ladder in configs/policies.yaml",
        }


learner = Learner()
