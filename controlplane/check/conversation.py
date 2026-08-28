"""Compounding risk across turns and agent actions.

This answers the brief's point that multi-turn conversations and acting agents
introduce risk that single-response checking cannot see. One 0.4-risk turn is
fine. Six consecutive 0.4-risk turns feeding an agent that takes irreversible
actions is not -- and no per-response checker would ever notice.

Risk accumulates as 1 - prod(1 - r_i), the standard "at least one failure" form,
then is amplified for agentic and irreversible traffic.
"""
from __future__ import annotations

from collections import defaultdict, deque

from ..schemas import Completion, InterceptedRequest, RiskDim, Signal, Verdict
from .base import timed

# session_id -> rolling window of past fused risk scores.
# In production this belongs in Redis; see the limitations table in the doc.
_HISTORY: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))

# A turn must be at least this risky to count toward compounding at all.
_RISK_FLOOR = 0.45
# Per-turn recency decay, so a bad turn ten turns ago barely matters now.
_DECAY = 0.7


def record_session_risk(session_id: str, fused: float) -> None:
    """Called by the pipeline AFTER fusion, closing the loop for the next turn."""
    _HISTORY[session_id].append(fused)


def reset_sessions() -> None:
    _HISTORY.clear()


class ConversationDetector:
    name, dim, tier, est_ms = "conversation", RiskDim.COMPOUNDING, 1, 2

    async def run(self, req: InterceptedRequest, comp: Completion) -> Signal:
        with timed() as t:
            hist = list(_HISTORY.get(req.session_id, []))

            # Only turns that were THEMSELVES elevated contribute, and older
            # turns decay.
            #
            # The naive form -- accumulating every turn with no decay --
            # saturates almost immediately: at a typical per-turn risk of 0.5,
            # six benign turns give 1 - 0.5^6 = 0.98 even when nothing was ever
            # wrong. Measured on 109 labelled traces it sat at 0.357 on false
            # positives vs 0.331 on true negatives: a discrimination of 0.003,
            # i.e. a near-constant risk floor carrying no information, in the
            # same failure class as the old constant bias score.
            #
            # Compounding risk should mean "this conversation has REPEATEDLY
            # been risky recently", not "this conversation has existed".
            acc = 1.0
            contributing = 0
            for age, r in enumerate(reversed(hist)):
                if r < _RISK_FLOOR:
                    continue
                contributing += 1
                decay = _DECAY ** age
                acc *= 1 - min(max(r, 0.0), 0.95) * decay
            compounded = 1 - acc

            # A long conversation has more surface area for a bad claim to have
            # already shaped a downstream decision.
            depth_factor = min(len(req.messages) / 12, 1.0)
            score = compounded * (0.5 + 0.5 * depth_factor)

            if req.is_agentic:
                score = min(1.0, score * 1.35)   # output becomes an action
            if not req.reversible:
                score = min(1.0, score * 1.5)    # and it cannot be undone

        return Signal(
            detector=self.name,
            dim=self.dim,
            score=round(score, 3),
            verdict=(
                Verdict.FAIL if score > 0.7
                else Verdict.SUSPECT if score > 0.35
                else Verdict.PASS
            ),
            confidence=0.5,
            latency_ms=t.ms,
            evidence={
                "turns_tracked": len(hist),
                "turns_contributing": contributing,
                "risk_floor": _RISK_FLOOR,
                "compounded": round(compounded, 3),
                "agentic": req.is_agentic,
                "reversible": req.reversible,
            },
        )
