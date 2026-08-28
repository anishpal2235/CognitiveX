"""The 'cost' dimension from the pitch, made concrete.

It answers one question: would a cheaper model plausibly have sufficed? The
signal feeds the router's reward so the system learns to stop over-serving easy
prompts.

Note its fusion weight is only 0.2. Cost should shape ROUTING, never censor a
user -- an expensive answer is not an unsafe answer.
"""
from __future__ import annotations

from ..schemas import Completion, InterceptedRequest, RiskDim, Signal, Verdict
from .base import timed


class CostDetector:
    name, dim, tier, est_ms = "cost", RiskDim.COST, 1, 1

    def __init__(self, cheapest_usd_per_call: float = 0.0002):
        self.floor = cheapest_usd_per_call

    async def run(self, req: InterceptedRequest, comp: Completion) -> Signal:
        with timed() as t:
            ratio = comp.usd / max(self.floor, 1e-9)
            simple = (
                len(req.prompt) < 240 and "?" in req.prompt and not req.is_agentic
            )
            score = 0.0
            if simple and ratio > 4:
                # Easy question answered by an expensive model: pure overspend.
                score = min(1.0, 0.25 + 0.1 * (ratio - 4))
            elif ratio > 12:
                score = min(1.0, 0.2 + 0.05 * (ratio - 12))

        return Signal(
            detector=self.name,
            dim=self.dim,
            score=round(score, 3),
            verdict=Verdict.SUSPECT if score > 0.5 else Verdict.PASS,
            confidence=0.6,
            latency_ms=t.ms,
            evidence={
                "usd": round(comp.usd, 6),
                "vs_cheapest": round(ratio, 2),
                "prompt_looked_simple": simple,
            },
        )
