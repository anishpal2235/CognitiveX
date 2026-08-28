from __future__ import annotations

import time
from typing import Protocol

from ..schemas import Completion, InterceptedRequest, RiskDim, Signal


class Detector(Protocol):
    """One tiny protocol, so adding a new regulation-driven check is a 30-line
    file rather than a refactor.
    """

    name: str
    dim: RiskDim
    tier: int    # 1 = always run (cheap); 2 = run only if tier-1 is suspicious
    est_ms: int  # for deadline planning

    async def run(self, req: InterceptedRequest, comp: Completion) -> Signal:
        ...


class timed:
    """Stamps detector latency onto a Signal.

    Latency is itself a governance metric: a detector that repeatedly blows its
    budget gets demoted to tier 2 or removed from a latency-tight use case.
    """

    ms: int = 0

    def __enter__(self) -> "timed":
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> bool:
        self.ms = int((time.perf_counter() - self.t0) * 1000)
        return False

    @property
    def elapsed_ms(self) -> int:
        """Readable mid-block, unlike `ms` which is only set on exit."""
        return int((time.perf_counter() - self.t0) * 1000)
