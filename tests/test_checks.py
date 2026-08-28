"""Detector behaviour tests.

These encode the DESIGN CLAIMS, not just the code. If someone later "optimises"
the PII detector into a naive scanner, the leakage-vs-echo test fails and tells
them why that is wrong.
"""
from __future__ import annotations

import pytest

from controlplane.check.bias import BiasDetector
from controlplane.check.grounding import GroundingDetector
from controlplane.check.performance import SelfConsistencyDetector
from controlplane.check.privacy import PIIDetector
from controlplane.schemas import Completion, InterceptedRequest, Turn, UseCase, Verdict


def _req(prompt: str, **kw) -> InterceptedRequest:
    return InterceptedRequest(messages=[Turn(role="user", content=prompt)], **kw)


def _comp(text: str, samples: list[str] | None = None) -> Completion:
    return Completion(model="t", text=text, samples=samples or [])


@pytest.mark.asyncio
async def test_pii_flags_novel_egress():
    s = await PIIDetector().run(
        _req("who owns this account?"),
        _comp("Contact them at ravi.sharma@acme.co"),
    )
    assert s.score > 0.3
    assert s.evidence["entities"][0]["novel"] is True


@pytest.mark.asyncio
async def test_pii_discounts_echoed_data():
    """Echoing back the user's OWN email is not an egress event. This single
    distinction removes a large share of naive-scanner false positives."""
    email = "ravi.sharma@acme.co"
    novel = await PIIDetector().run(_req("hello"), _comp(f"Write to {email}"))
    echoed = await PIIDetector().run(
        _req(f"my email is {email}"), _comp(f"Write to {email}")
    )
    assert echoed.score < novel.score


@pytest.mark.asyncio
async def test_credit_card_luhn_suppresses_false_positive():
    """A random 16-digit string must NOT be reported as a card number."""
    bad = await PIIDetector().run(_req("x"), _comp("ref 1234 5678 9012 3456"))
    kinds = {e["type"] for e in bad.evidence["entities"]}
    assert "credit_card" not in kinds


@pytest.mark.asyncio
async def test_bias_ignores_bare_mention_but_flags_generalisation():
    """The precision trick: mentioning a protected attribute is not bias."""
    neutral = await BiasDetector().run(
        _req("q"), _comp("She submitted the form yesterday.")
    )
    biased = await BiasDetector().run(
        _req("q"),
        _comp("Older applicants are usually less reliable, so reject the claim."),
    )
    assert neutral.score < 0.3
    assert biased.score > 0.6


@pytest.mark.asyncio
async def test_bias_is_context_sensitive():
    """The same sentence is more dangerous in a lending decision."""
    text = _comp("Older applicants typically score lower, so decline.")
    chat = await BiasDetector().run(_req("q"), text)
    loan = await BiasDetector().run(
        _req("q", use_case=UseCase.CREDIT_DECISION), text
    )
    assert loan.score >= chat.score


@pytest.mark.asyncio
async def test_selfconsistency_detects_numeric_instability():
    s = await SelfConsistencyDetector().run(
        _req("how many days?"),
        _comp("It takes 30 days.", ["It takes 45 days.", "It takes 12 days."]),
    )
    assert s.score > 0.5


@pytest.mark.asyncio
async def test_selfconsistency_abstains_on_single_sample():
    """Cannot measure dispersion from one sample -> ABSTAIN, never PASS.
    'I could not check this' must not render as 'this is safe'."""
    s = await SelfConsistencyDetector().run(_req("q"), _comp("one answer only"))
    assert s.verdict == Verdict.ABSTAIN


@pytest.mark.asyncio
async def test_grounding_passes_when_no_checkable_claims():
    s = await GroundingDetector().run(_req("hi"), _comp("Happy to help!"))
    assert s.verdict == Verdict.PASS


@pytest.mark.asyncio
async def test_grounding_reacts_to_invented_citation():
    s = await GroundingDetector().run(
        _req("what does the policy say?"),
        _comp("Section 14(b) of the charter guarantees a 100% refund in all cases."),
    )
    assert s.score > 0.0
    assert s.evidence["n_claims"] >= 1
