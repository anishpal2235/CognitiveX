"""Router tests: the SAFETY guarantees, not just the optimisation.

The important assertions here are the negative ones -- what the router must
never do, no matter how attractive the cost or the exploration bonus.
"""
from __future__ import annotations

import numpy as np

from controlplane.policy.engine import resolve
from controlplane.route.bandit import LinUCB
from controlplane.route.budget import BudgetController
from controlplane.route.features import D, featurize
from controlplane.route.router import Router
from controlplane.schemas import InterceptedRequest, Turn, UseCase


def _req(**kw) -> InterceptedRequest:
    return InterceptedRequest(
        messages=[Turn(role="user", content="How long do refunds take?")], **kw
    )


def test_features_are_bounded_and_sized():
    x = featurize(_req())
    assert x.shape == (D,)
    assert np.all(x >= 0) and np.all(x <= 1)


def test_eligibility_respects_data_residency():
    """An EU request must never be routed to a model without an EU region."""
    r = Router()
    pol = resolve("support_bot", "EU", "confidential")
    for m in r.eligible(_req(geo="EU"), pol):
        assert "EU" in m.regions


def test_eligibility_respects_sensitivity_ceiling():
    """Regulated data must never reach a model capped below it."""
    r = Router()
    pol = resolve("credit_decision", "IN", "regulated")
    order = {"public": 0, "internal": 1, "confidential": 2, "regulated": 3}
    for m in r.eligible(_req(data_class="regulated"), pol):
        assert order[m.max_data_class] >= order["regulated"]


def test_eu_overlay_forbids_untagged_models():
    """The EU overlay forbids models tagged no_dpa -- policy beats optimisation."""
    r = Router()
    pol = resolve("support_bot", "EU", "confidential")
    for m in r.eligible(_req(geo="EU"), pol):
        assert not set(m.tags) & set(pol.forbid_model_tags)


def test_eligibility_never_returns_empty():
    """A guardrail that takes the product down gets switched off. Even with an
    impossible context we must fall back to the safest single arm."""
    r = Router()
    pol = resolve("credit_decision", "EU", "regulated")
    assert len(r.eligible(_req(geo="EU", data_class="regulated"), pol)) >= 1


def test_no_exploration_on_regulated_traffic():
    """Learning is confined to low-risk, reversible traffic. Two identical
    regulated requests must resolve deterministically."""
    r = Router()
    pol = resolve("credit_decision", "IN", "regulated")
    req = _req(use_case=UseCase.CREDIT_DECISION, data_class="regulated",
               is_agentic=True, reversible=False)
    a, _, why_a, _ = r.select(req, pol)
    b, _, _, _ = r.select(req, pol)
    assert a.name == b.name
    assert "explore=False" in why_a


def test_bandit_learns_from_reward():
    b = LinUCB(["a", "b"], 4)
    x = np.array([1.0, 0.5, 0.0, 0.2])
    for _ in range(30):
        b.update("a", x, 0.9)
        b.update("b", x, -0.5)
    mean_a, _ = b.score("a", x, explore=False)
    mean_b, _ = b.score("b", x, explore=False)
    assert mean_a > mean_b


def test_exploration_bonus_shrinks_with_evidence():
    """Uncertainty must decay as the arm is pulled, otherwise the router never
    settles and quality never stabilises."""
    b = LinUCB(["a"], 4)
    x = np.array([1.0, 0.5, 0.0, 0.2])
    _, first = b.score("a", x)
    for _ in range(50):
        b.update("a", x, 0.5)
    _, later = b.score("a", x)
    assert later < first


def test_lambda_rises_when_overspending():
    """lambda is the shadow price of a dollar: overspend -> cheaper arms."""
    bc = BudgetController(weekly_usd=1.0, window_hours=168, lr=0.2)
    start = bc.lmbda
    for _ in range(20):
        bc.record(0.5)
    assert bc.lmbda > start
    assert bc.status()["pace"] > 1.0
