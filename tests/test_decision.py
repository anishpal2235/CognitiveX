"""Fusion and ladder tests -- the over/under-flagging tradeoff, pinned down."""
from __future__ import annotations

from controlplane.act.decision import ALERTS, decide
from controlplane.act.fusion import fuse
from controlplane.act.repair import auto_repair, redact_pii
from controlplane.policy.engine import resolve
from controlplane.schemas import (
    Action,
    Completion,
    InterceptedRequest,
    RiskDim,
    RiskVector,
    Signal,
    Turn,
    UseCase,
    Verdict,
)


def _rv(*sigs: Signal) -> RiskVector:
    return RiskVector(
        signals=list(sigs),
        abstained=[s.detector for s in sigs if s.verdict == Verdict.ABSTAIN],
    )


def _sig(det, dim, score, conf=0.8, verdict=Verdict.SUSPECT) -> Signal:
    return Signal(detector=det, dim=dim, score=score, confidence=conf, verdict=verdict)


def _req(**kw) -> InterceptedRequest:
    return InterceptedRequest(messages=[Turn(role="user", content="q")], **kw)


def _comp(text="Refunds take 30 days.") -> Completion:
    return Completion(model="m", text=text, usd=0.001, latency_ms=100)


def test_high_confidence_signal_is_not_diluted():
    """Confidence weighting: a strong PII hit must survive alongside weak noise."""
    pol = resolve("support_bot", "IN", "confidential")
    rv = fuse(
        _rv(
            _sig("pii", RiskDim.PRIVACY, 0.95, conf=0.9),
            _sig("bias", RiskDim.BIAS, 0.0, conf=0.2),
            _sig("cost", RiskDim.COST, 0.0, conf=0.6),
        ),
        pol,
    )
    assert rv.fused > 0.4


def test_overlapping_risk_is_superadditive():
    """A fabricated detail about a real person is worse than either risk alone --
    exactly the overlap case in the brief."""
    pol = resolve("support_bot", "IN", "confidential")
    only_h = fuse(_rv(_sig("se", RiskDim.HALLUCINATION, 0.6)), pol).fused
    both = fuse(
        _rv(
            _sig("se", RiskDim.HALLUCINATION, 0.6),
            _sig("pii", RiskDim.PRIVACY, 0.6),
        ),
        pol,
    ).fused
    assert both > only_h


def test_abstention_carries_partial_risk_not_zero():
    """'I could not verify this' must never score as safe."""
    pol = resolve("support_bot", "IN", "confidential")
    rv = fuse(
        _rv(_sig("grounding", RiskDim.GROUNDING, 0.8, verdict=Verdict.ABSTAIN)), pol
    )
    assert rv.fused > 0.1


def test_degraded_checks_increase_risk():
    """Timeouts must fail CLOSED, not open."""
    pol = resolve("support_bot", "IN", "confidential")
    base = _rv(_sig("bias", RiskDim.BIAS, 0.3))
    clean = fuse(base.model_copy(deep=True), pol).fused
    degraded_rv = base.model_copy(deep=True)
    degraded_rv.degraded = True
    assert fuse(degraded_rv, pol).fused > clean


def test_eu_overlay_amplifies_privacy():
    eu = fuse(_rv(_sig("pii", RiskDim.PRIVACY, 0.5)), resolve("support_bot", "EU", "confidential")).fused
    us = fuse(_rv(_sig("pii", RiskDim.PRIVACY, 0.5)), resolve("support_bot", "US", "confidential")).fused
    assert eu > us


def test_low_risk_allows():
    pol = resolve("support_bot", "IN", "confidential")
    rv = fuse(_rv(_sig("cost", RiskDim.COST, 0.0, verdict=Verdict.PASS)), pol)
    assert decide(_req(), _comp(), rv, pol).action == Action.ALLOW


def test_hard_rule_can_only_escalate_severity():
    """A compliance rule is a FLOOR, never a way to relax the ladder."""
    pol = resolve("support_bot", "IN", "confidential")
    rv = fuse(_rv(_sig("pii", RiskDim.PRIVACY, 0.85, conf=0.9)), pol)
    dec = decide(_req(), _comp("reach them at a@b.co"), rv, pol)
    assert dec.action in (Action.REPAIR, Action.ESCALATE, Action.BLOCK)


def test_irreversible_agentic_output_is_escalated():
    """The compounding-risk guard: medium risk plus an action you cannot undo
    must never ship unreviewed."""
    pol = resolve("credit_decision", "IN", "regulated")
    rv = fuse(_rv(_sig("se", RiskDim.HALLUCINATION, 0.5)), pol)
    dec = decide(
        _req(use_case=UseCase.CREDIT_DECISION, data_class="regulated",
             is_agentic=True, reversible=False),
        _comp(),
        rv,
        pol,
    )
    assert dec.action in (Action.ESCALATE, Action.BLOCK)


def test_escalation_never_leaks_the_risky_draft():
    """The reviewer sees the draft; the end user sees a holding message."""
    pol = resolve("credit_decision", "IN", "regulated")
    rv = fuse(_rv(_sig("pii", RiskDim.PRIVACY, 0.7, conf=0.9)), pol)
    dec = decide(
        _req(use_case=UseCase.CREDIT_DECISION, data_class="regulated",
             is_agentic=True, reversible=False),
        _comp("Her email is priya@acme.co and the refund is guaranteed."),
        rv,
        pol,
    )
    if dec.action == Action.ESCALATE:
        assert "priya@acme.co" not in dec.repaired_text
        assert dec.human_ticket


def test_alert_budget_downgrades_borderline_cases():
    """Directly targets alert fatigue: once the review budget is spent,
    borderline cases are repaired instead of escalated -- but extreme risk
    (>=0.9) always still surfaces."""
    ALERTS.reset()
    pol = resolve("support_bot", "IN", "confidential")
    rv = fuse(_rv(_sig("se", RiskDim.HALLUCINATION, 0.72, conf=0.9)), pol)

    actions = [decide(_req(), _comp(), rv, pol).action for _ in range(60)]
    alerts = sum(1 for a in actions if a in (Action.ESCALATE, Action.BLOCK))
    assert alerts < len(actions)      # the budget bit at some point

    extreme = fuse(_rv(_sig("pii", RiskDim.PRIVACY, 1.0, conf=1.0)), pol)
    assert decide(_req(), _comp(), extreme, pol).action in (
        Action.ESCALATE,
        Action.BLOCK,
    )
    ALERTS.reset()


def test_repair_preserves_utility():
    """The core product argument: correction keeps the answer, blocking destroys it."""
    pol = resolve("support_bot", "IN", "confidential")
    rv = fuse(_rv(_sig("pii", RiskDim.PRIVACY, 0.6, conf=0.9)), pol)
    text, notes = auto_repair(
        _comp("Refunds take 30 days. Contact ravi@acme.co for help."), rv
    )
    assert "ravi@acme.co" not in text
    assert "30 days" in text          # the useful part survived
    assert notes


def test_redact_pii_masks_all_patterns():
    out, n = redact_pii("mail a@b.co and call +91 98200 11223")
    assert n >= 2
    assert "a@b.co" not in out
