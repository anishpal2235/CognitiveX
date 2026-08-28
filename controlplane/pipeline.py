"""Intercept -> Route -> Check -> Act -> Learn, in one readable function.

This is the whole thesis in ~90 lines. Read it top to bottom and you have the
system.
"""
from __future__ import annotations

import time

from .act.decision import decide
from .act.fusion import fuse
from .check.conversation import record_session_risk
from .check.orchestrator import CheckOrchestrator
from .config import settings
from .feedback.reward import compute_reward
from .observability import audit
from .policy.engine import resolve
from .providers.mock import MockProvider
from .providers.openai_compat import OpenAICompatProvider
from .route.router import router
from .schemas import Completion, InterceptedRequest, Trace
from .store.db import DB

PROVIDER = MockProvider() if settings.provider_mode == "mock" else OpenAICompatProvider()


def _cheapest_usd() -> float:
    m = min(router.models, key=lambda x: x.usd_per_1k_out)
    return (m.usd_per_1k_in + m.usd_per_1k_out) * 0.3


async def handle(req: InterceptedRequest) -> Trace:
    t_start = time.perf_counter()

    # ---- 1. POLICY: resolved per request, never hard-coded. -----------------
    # Use case + geography + data class are composed into one effective policy,
    # so a new regulation is a YAML edit and a version bump, not a release.
    pol = resolve(req.use_case.value, req.geo, req.data_class)

    # ---- 2. ROUTE: eligibility first, then constrained optimisation. --------
    spec, eligible, why, x = router.select(req, pol)

    # ---- 3. GENERATE ------------------------------------------------------
    # Extra samples are requested ONLY when the policy's latency budget can
    # absorb them -- they are the raw material for semantic entropy, and they
    # are also the main cost lever, so they are budgeted, not assumed.
    n = settings.n_samples if pol.latency_budget_ms >= 1500 else 2
    comp: Completion = await PROVIDER.generate(req, spec, n_samples=n)

    # ---- 4. CHECK: parallel, tiered, deadline-bounded. ---------------------
    t_check = time.perf_counter()
    orch = CheckOrchestrator(judge_provider=PROVIDER, judge_spec=spec)
    rv = await orch.run(req, comp, pol)
    rv = fuse(rv, pol)
    check_ms = int((time.perf_counter() - t_check) * 1000)

    # ---- 5. ACT: graded ladder + hard rules + alert budget. -----------------
    dec = decide(req, comp, rv, pol)

    # ---- 6. LEARN: Check's scores become Route's reward. --------------------
    # This is the flywheel. No separate labelling pipeline is required for the
    # system to start improving.
    reward = compute_reward(
        comp, rv, dec, router.budget.lmbda, pol.latency_budget_ms
    )
    router.learn(spec.name, x, reward, comp.usd)

    # Feed this turn's risk forward so the next turn can see compounding risk.
    record_session_risk(req.session_id, rv.fused)

    trace = Trace(
        request=req,
        policy_version=pol.version,
        eligible_models=eligible,
        chosen_model=spec.name,
        route_reason=why,
        completion=comp,
        risk=rv,
        decision=dec,
        reward=round(reward, 4),
        total_latency_ms=int((time.perf_counter() - t_start) * 1000),
        check_latency_ms=check_ms,
        budget=router.budget.status(),
    )

    # ---- 7. PERSIST + AUDIT ------------------------------------------------
    payload = trace.model_dump(mode="json")
    DB.save_trace(payload)
    audit.append(payload)
    return trace
