"""Full pipeline tests: Intercept -> Route -> Check -> Act -> Learn."""
from __future__ import annotations

import pytest

from controlplane.check.conversation import record_session_risk, reset_sessions
from controlplane.feedback.learner import learner
from controlplane.observability import audit, metrics
from controlplane.pipeline import handle
from controlplane.policy.engine import resolve
from controlplane.schemas import InterceptedRequest, Turn, UseCase
from controlplane.store.db import DB


def _req(**kw) -> InterceptedRequest:
    return InterceptedRequest(
        messages=[Turn(role="user", content="How long do refunds take?")], **kw
    )


@pytest.mark.asyncio
async def test_pipeline_produces_complete_trace():
    t = await handle(_req(use_case=UseCase.SUPPORT_BOT, data_class="confidential"))
    assert t.chosen_model
    assert t.decision is not None and t.decision.repaired_text
    assert 0.0 <= t.risk.fused <= 1.0
    assert t.reward is not None
    assert t.policy_version
    assert t.risk.signals, "at least one detector must have run"


@pytest.mark.asyncio
async def test_trace_is_persisted_and_auditable():
    t = await handle(_req())
    assert DB.get_trace(t.request.request_id) is not None
    assert audit.verify_chain()["ok"] is True


@pytest.mark.asyncio
async def test_check_overhead_respects_latency_budget():
    """Inline checking is only viable if it stays inside the budget. Tiering plus
    parallelism plus a hard deadline is what makes that true."""
    pol = resolve("support_bot", "IN", "confidential")
    t = await handle(_req(use_case=UseCase.SUPPORT_BOT, data_class="confidential"))
    assert t.check_latency_ms < pol.latency_budget_ms


@pytest.mark.asyncio
async def test_regulated_flow_is_stricter_than_support_flow():
    """Same question, different context -> different treatment. This is the whole
    argument against one-size-fits-all checking."""
    soft = resolve("support_bot", "IN", "confidential")
    hard = resolve("credit_decision", "IN", "regulated")
    assert hard.ladder[0]["max_risk"] < soft.ladder[0]["max_risk"]
    assert hard.risk_multiplier >= soft.risk_multiplier
    assert "judge" in hard.detectors


@pytest.mark.asyncio
async def test_compounding_risk_accumulates_across_turns():
    """Six mediocre turns are not the same as one mediocre turn."""
    reset_sessions()
    from controlplane.check.conversation import ConversationDetector
    from controlplane.schemas import Completion

    req = _req(session_id="s-compound")
    comp = Completion(model="m", text="ok")
    first = await ConversationDetector().run(req, comp)
    for _ in range(6):
        record_session_risk("s-compound", 0.45)
    later = await ConversationDetector().run(req, comp)
    assert later.score > first.score
    reset_sessions()


@pytest.mark.asyncio
async def test_human_label_updates_router_and_metrics():
    """The explicit feedback channel: a reviewer's judgement is gold-standard
    supervision and must reach both the bandit and the trust report."""
    t = await handle(_req())
    out = learner.apply_human_label(t.request.request_id, was_correct_flag=True)
    assert out["ok"] is True
    rep = metrics.trust_report()
    assert rep["detection"]["n"] >= 1


@pytest.mark.asyncio
async def test_unknown_request_id_is_rejected():
    assert learner.apply_human_label("does-not-exist", True)["ok"] is False


@pytest.mark.asyncio
async def test_metrics_report_has_all_four_sections():
    await handle(_req())
    rep = metrics.trust_report()
    for key in ("operations", "detection", "calibration", "per_use_case"):
        assert key in rep
