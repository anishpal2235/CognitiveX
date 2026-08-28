"""Online knapsack via Lagrangian relaxation.

We want:  max sum(quality)  subject to  sum(cost) <= B  over a window.
Relaxed:  max sum(quality - lambda * cost),   lambda >= 0

lambda is updated by dual gradient ascent on the pacing error:

    lambda <- max(0, lambda + lr * (spend_rate / target_rate - 1))

Interpretation: lambda is the SHADOW PRICE OF A DOLLAR. When we overspend it
rises and the router quietly prefers cheaper arms; when we underspend it decays
and quality is bought back. No hard cutoffs, no cliff at the end of the month,
no "budget exhausted, service degraded" incident.
"""
from __future__ import annotations

import time


class BudgetController:
    def __init__(self, weekly_usd: float, window_hours: int = 168, lr: float = 0.05):
        self.budget = weekly_usd
        self.window_s = window_hours * 3600
        self.lr = lr
        self.lmbda = 1.0
        self.spend = 0.0
        self.t0 = time.time()

    def elapsed_frac(self) -> float:
        return min(max((time.time() - self.t0) / self.window_s, 1e-4), 1.0)

    def record(self, usd: float) -> None:
        self.spend += usd
        expected = self.budget * self.elapsed_frac()
        err = (self.spend / max(expected, 1e-9)) - 1.0
        self.lmbda = max(0.0, min(self.lmbda + self.lr * err, 50.0))

    def penalty(self, usd_estimate: float) -> float:
        """Scaled so lambda ~1 makes a $0.01 call cost ~0.1 utility points, i.e.
        cost trades against quality on a comparable 0..1 scale. Without this
        normalisation the two terms are not commensurable and the router either
        ignores cost entirely or refuses to ever spend.
        """
        return self.lmbda * usd_estimate * 10.0

    def status(self) -> dict[str, float]:
        expected = self.budget * self.elapsed_frac()
        return {
            "spend": round(self.spend, 6),
            "budget": self.budget,
            "lambda": round(self.lmbda, 4),
            "pace": round(self.spend / max(expected, 1e-9), 4),
        }
