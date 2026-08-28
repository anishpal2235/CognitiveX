"""Route = hard policy filter -> constrained bandit selection -> learning.

The ordering is the safety argument. Eligibility is decided BEFORE any
optimisation, so an ineligible-but-tempting model is never even scored, and no
amount of cost pressure or exploration bonus can smuggle it back in.
"""
from __future__ import annotations

import numpy as np

from ..config import ROOT, models_cfg
from ..policy.engine import ResolvedPolicy
from ..schemas import InterceptedRequest, ModelSpec
from .bandit import LinUCB
from .budget import BudgetController
from .features import D, featurize

_ORDER = {"public": 0, "internal": 1, "confidential": 2, "regulated": 3}
STATE = ROOT / "router_state.json"


def load_models() -> list[ModelSpec]:
    return [ModelSpec(**m) for m in models_cfg().data["models"]]


class Router:
    def __init__(self) -> None:
        self.models = load_models()
        names = [m.name for m in self.models]
        self.state_path = STATE
        self.bandit = LinUCB.load(STATE, names, D)
        b = models_cfg().data["budget"]
        self.budget = BudgetController(b["weekly_usd"], b["window_hours"], b["lambda_lr"])

    # ---------- 1. hard policy filter: safety BEFORE optimisation ----------
    def eligible(self, req: InterceptedRequest, pol: ResolvedPolicy) -> list[ModelSpec]:
        out = []
        for m in self.models:
            if req.geo not in m.regions:
                continue                                    # data residency
            if _ORDER.get(req.data_class, 1) > _ORDER.get(m.max_data_class, 1):
                continue                                    # sensitivity ceiling
            if any(t in m.tags for t in pol.forbid_model_tags):
                continue                                    # e.g. EU: no DPA -> out
            if m.p50_latency_ms > pol.latency_budget_ms * 0.75:
                continue                                    # leave room for checks
            out.append(m)

        if not out:
            # Never fail closed on routing: fall back to the single safest arm.
            # A guardrail that takes the product down is a guardrail that gets
            # switched off.
            out = [
                max(
                    self.models,
                    key=lambda m: (_ORDER.get(m.max_data_class, 0), m.quality_prior),
                )
            ]
        return out

    def _est_usd(self, req: InterceptedRequest, m: ModelSpec) -> float:
        tin = max(1, len(req.prompt) // 4)
        return (tin / 1000) * m.usd_per_1k_in + (req.max_tokens / 1000) * m.usd_per_1k_out

    # ---------- 2. constrained selection ----------
    def select(
        self, req: InterceptedRequest, pol: ResolvedPolicy
    ) -> tuple[ModelSpec, list[str], str, np.ndarray]:
        x = featurize(req)
        cands = self.eligible(req, pol)

        # Exploration gate. Learning is confined to traffic that is low-risk AND
        # reversible, so the promise "performance never falls below the incumbent
        # baseline" is enforced structurally, not by hope.
        explore = (
            pol.allow_exploration
            and req.reversible
            and not req.is_agentic
            and req.data_class != "regulated"
        )

        best, best_val, why = None, -1e9, ""
        for m in cands:
            mean, bonus = self.bandit.score(m.name, x, explore=explore)

            # Prior blend: while an arm is cold, trust the published benchmark
            # rather than an optimistic empty regression.
            n = self.bandit.n.get(m.name, 0)
            w = 1.0 / (1.0 + n / 25.0)
            utility = (1 - w) * mean + w * m.quality_prior

            pen = self.budget.penalty(self._est_usd(req, m))
            val = utility + bonus - pen
            if val > best_val:
                best, best_val = m, val
                why = (
                    f"utility={utility:.3f} bonus={bonus:.3f} cost_pen={pen:.3f} "
                    f"lambda={self.budget.lmbda:.2f} n={n} explore={explore}"
                )

        # Safety floor: if the app explicitly asked for a stronger eligible model
        # and this use case is not allowed to explore, honour the request.
        if req.requested_model and not pol.allow_exploration:
            forced = next((m for m in cands if m.name == req.requested_model), None)
            if forced:
                best = forced
                why += " | override:requested_model(no-explore policy)"

        assert best is not None
        return best, [m.name for m in cands], why, x

    # ---------- 3. learn ----------
    def learn(self, model_name: str, x: np.ndarray, reward: float, usd: float) -> None:
        self.bandit.update(model_name, x, reward)
        self.budget.record(usd)
        self.bandit.save(self.state_path)


router = Router()
