"""Check's scores become Route's reward. This file IS the product thesis.

reward = quality - risk penalty - shadow-priced cost - latency penalty

Because the same signals that protect the user also teach the router, safety and
savings stop being in tension: a cheap model that leaks PII earns a NEGATIVE
reward and loses traffic automatically. Nobody has to write a rule saying
"don't use the cheap model for sensitive questions" -- the system discovers it.
"""
from __future__ import annotations

from ..schemas import Action, Completion, Decision, RiskVector

W_RISK, W_COST, W_LAT = 1.10, 0.45, 0.15


def compute_reward(
    comp: Completion,
    rv: RiskVector,
    dec: Decision,
    lmbda: float,
    latency_budget_ms: int,
    human_label: float | None = None,
) -> float:
    """quality: 1 - epistemic risk, i.e. how trustworthy the answer was. A human
              label replaces the proxy whenever one exists -- labels dominate.
    risk:    the fused responsibility score, weighted above 1.0 so safety
             outranks cost in the objective rather than merely competing with it.
    cost:    multiplied by the LIVE budget multiplier lambda, so the router's
             cost sensitivity tightens automatically when spend is off-pace.
    latency: only penalised once the use case's budget is actually exceeded.
    """
    epistemic = max(
        rv.per_dim.get("hallucination", 0.0),
        rv.per_dim.get("grounding", 0.0),
    )
    quality = human_label if human_label is not None else (1.0 - epistemic)

    risk_pen = W_RISK * rv.fused
    if dec.action == Action.BLOCK:
        risk_pen += 0.5      # a blocked answer served nobody
    elif dec.action == Action.ESCALATE:
        risk_pen += 0.25     # consumed scarce human attention

    cost_pen = W_COST * lmbda * comp.usd * 10.0
    lat_pen = W_LAT * max(0.0, comp.latency_ms / max(latency_budget_ms, 1) - 1.0)

    return float(max(-1.0, min(1.0, quality - risk_pen - cost_pen - lat_pen)))
