"""Parallel, tiered, deadline-aware check execution.

This is the piece that makes an INLINE gate viable at all. Three latency
defences, all worth naming in the pitch:

1. PARALLELISM -- detectors within a tier run concurrently, so a tier costs
   max(latency), not sum(latency).
2. TIERING -- expensive detectors (judge, semantic entropy) run only when cheap
   ones are already suspicious. On clean traffic, which is the overwhelming
   majority, we pay only the tier-1 cost.
3. DEADLINE -- a hard wall-clock budget. On expiry we do NOT fail open: the
   RiskVector is marked `degraded`, which fusion treats as a reason to be MORE
   conservative, not less. "We didn't finish checking" must never render as
   "this is safe".
"""
from __future__ import annotations

import asyncio
import time

from ..policy.engine import ResolvedPolicy
from ..schemas import (
    Completion,
    InterceptedRequest,
    RiskDim,
    RiskVector,
    Signal,
    Verdict,
)
from .bias import BiasDetector
from .conversation import ConversationDetector
from .cost import CostDetector
from .grounding import GroundingDetector
from .judge import JudgeDetector
from .performance import SelfConsistencyDetector, SemanticEntropyDetector
from .privacy import PIIDetector

REGISTRY = {
    "pii": PIIDetector,
    "bias": BiasDetector,
    "grounding": GroundingDetector,
    "selfconsistency": SelfConsistencyDetector,
    "semantic_entropy": SemanticEntropyDetector,
    "conversation": ConversationDetector,
    "cost": CostDetector,
    "judge": JudgeDetector,
}

# Tier-2 detectors fire only above this tier-1 max score.
SUSPICION_GATE = 0.30


class CheckOrchestrator:
    def __init__(self, judge_provider=None, judge_spec=None):
        self.judge_provider, self.judge_spec = judge_provider, judge_spec

    def _build(self, names: list[str]) -> list:
        out = []
        seen = set()
        for n in names:
            if n in seen:
                continue
            seen.add(n)
            cls = REGISTRY.get(n)
            if cls is None:
                continue   # unknown detector in YAML: ignore, do not crash
            out.append(
                JudgeDetector(self.judge_provider, self.judge_spec)
                if n == "judge"
                else cls()
            )
        return out

    async def _run_one(self, d, req: InterceptedRequest, comp: Completion) -> Signal:
        """Execute one detector in a WORKER THREAD.

        Detectors are CPU-bound: regex, numpy, TF-IDF. Awaiting them directly on
        the event loop means they never hit an await point, so asyncio's timeout
        can never preempt them and the "hard deadline" silently never fires --
        `degraded` stays False no matter how far over budget a check runs.
        Handing each detector to a thread keeps the loop responsive, which is
        what makes the deadline actually enforceable.

        Honest caveat: cancelling the wrapper task stops us WAITING on the
        thread, it does not kill the thread. The late result is discarded and
        recorded as a timeout. Bounding the work itself (smaller KB, fewer
        samples) is still the real fix; this bounds the LATENCY WE SERVE.
        """
        return await asyncio.to_thread(asyncio.run, d.run(req, comp))

    async def _gather(
        self, dets: list, req: InterceptedRequest, comp: Completion, budget_s: float
    ) -> list[Signal]:
        if not dets:
            return []

        tasks = {asyncio.create_task(self._run_one(d, req, comp)): d for d in dets}
        done, pending = await asyncio.wait(tasks.keys(), timeout=max(budget_s, 0.01))

        sigs: list[Signal] = []
        for task in done:
            d = tasks[task]
            try:
                sigs.append(task.result())
            except Exception as exc:
                # A crashed detector yields a mid-risk TIMEOUT signal, never a
                # PASS. Failing open is how guardrail systems get breached.
                sigs.append(
                    Signal(
                        detector=d.name,
                        dim=getattr(d, "dim", RiskDim.SAFETY),
                        verdict=Verdict.TIMEOUT,
                        score=0.4,
                        confidence=0.1,
                        evidence={"error": str(exc)[:200]},
                    )
                )

        for task in pending:
            task.cancel()
            d = tasks[task]
            sigs.append(
                Signal(
                    detector=d.name,
                    dim=getattr(d, "dim", RiskDim.SAFETY),
                    verdict=Verdict.TIMEOUT,
                    score=0.4,
                    confidence=0.1,
                    evidence={"reason": "deadline_exceeded",
                              "budget_ms": int(budget_s * 1000)},
                )
            )
        return sigs

    async def run(
        self, req: InterceptedRequest, comp: Completion, pol: ResolvedPolicy
    ) -> RiskVector:
        t0 = time.perf_counter()

        # Whatever is left of the use case's latency budget after generation.
        total_s = max(pol.latency_budget_ms - comp.latency_ms, 150) / 1000.0

        dets = self._build(pol.detectors + ["cost"])
        t1 = [d for d in dets if d.tier == 1]
        t2 = [d for d in dets if d.tier == 2]

        # Reserve 60% of the budget for tier 1, keeping 40% for conditional tier 2.
        sigs = await self._gather(t1, req, comp, total_s * 0.6)
        max_t1 = max([s.score for s in sigs], default=0.0)
        abstained = [s.detector for s in sigs if s.verdict == Verdict.ABSTAIN]

        # Escalate to tier 2 if suspicious, if anything abstained, or if policy
        # mandates it for this use case / data class.
        must = pol.require_grounding or req.data_class == "regulated"
        if t2 and (max_t1 >= SUSPICION_GATE or abstained or must):
            elapsed = time.perf_counter() - t0
            sigs += await self._gather(t2, req, comp, max(total_s - elapsed, 0.05))

        return RiskVector(
            signals=sigs,
            abstained=[s.detector for s in sigs if s.verdict == Verdict.ABSTAIN],
            degraded=any(s.verdict == Verdict.TIMEOUT for s in sigs),
        )
