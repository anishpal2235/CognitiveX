"""Policy resolution: defaults <- use_case <- geo overlay <- data class overlay.

Why policy comes before detectors: if thresholds live inside detector code you
cannot vary behaviour by use case, geography or risk appetite, and you cannot
replay history. So policy is data, versioned by content hash.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from ..config import policies
from ..schemas import Action, InterceptedRequest  # noqa: F401  (re-exported for typing)


def resolve_for(req: "InterceptedRequest") -> "ResolvedPolicy":
    """Convenience wrapper for callers that already hold a request."""
    return resolve(req.use_case.value, req.geo, req.data_class)


@dataclass
class ResolvedPolicy:
    version: str
    latency_budget_ms: int
    detectors: list[str]
    ladder: list[dict[str, Any]]
    fusion_weights: dict[str, float]
    max_weight: float
    alert_budget: float
    allow_exploration: bool
    risk_multiplier: float
    privacy_multiplier: float
    require_grounding: bool
    forbid_model_tags: list[str] = field(default_factory=list)
    disclosure_required: bool = False
    hard_rules: list[dict[str, Any]] = field(default_factory=list)


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def resolve(
    use_case: str = "internal_copilot",
    geo: str = "IN",
    data_class: str = "internal",
) -> ResolvedPolicy:
    """Layered resolution. Overlays are additive for detectors and
    multiplicative for risk, so a new regulation is usually a YAML overlay,
    not a code change.

    Takes plain strings rather than a request object so the same function can
    serve the hot path AND the /v1/policy/preview endpoint, which must answer
    "what rules would apply to this context?" without a real request.
    """
    cfg = policies()
    d = cfg.data
    merged = _deep_merge(d["defaults"], d["use_cases"].get(use_case, {}))

    geo = d.get("geo_overlays", {}).get(geo, {}) or {}
    dcl = d.get("data_class_overlays", {}).get(data_class, {}) or {}

    detectors = list(merged["detectors"])
    for overlay in (geo, dcl):
        for extra in overlay.get("detectors_add", []):
            if extra not in detectors:
                detectors.append(extra)

    return ResolvedPolicy(
        version=cfg.version,
        latency_budget_ms=merged["latency_budget_ms"],
        detectors=detectors,
        ladder=merged["ladder"],
        fusion_weights=merged["fusion"]["weights"],
        max_weight=merged["fusion"]["max_weight"],
        alert_budget=merged.get("alert_budget", 0.1),
        allow_exploration=bool(merged.get("allow_exploration", False)),
        risk_multiplier=float(dcl.get("risk_multiplier", 1.0)),
        privacy_multiplier=float(geo.get("privacy_multiplier", 1.0)),
        require_grounding=bool(merged.get("require_grounding", False)),
        forbid_model_tags=list(geo.get("forbid_model_tags", [])),
        disclosure_required=bool(geo.get("disclosure_required", False)),
        hard_rules=list(merged.get("hard_rules", [])),
    )


def eval_hard_rules(
    pol: ResolvedPolicy, text: str, per_dim: dict[str, float]
) -> Optional[tuple[Action, str]]:
    """Hard rules bypass the statistical ladder. They are the compliance
    override: cheap, explainable and auditable -- the thing a regulator can read.

    Two condition forms are supported:
      "privacy >= 0.5"       -> numeric comparison on a fused risk dimension
      "regex:(pattern|here)" -> literal pattern match on the response text
    """
    for rule in pol.hard_rules:
        cond = str(rule["when"])
        hit = False
        if cond.startswith("regex:"):
            hit = re.search(cond[6:], text, re.I) is not None
        else:
            m = re.match(r"\s*(\w+)\s*(>=|>|<=|<)\s*([0-9.]+)\s*$", cond)
            if m:
                dim, op, thr = m.group(1), m.group(2), float(m.group(3))
                val = per_dim.get(dim, 0.0)
                hit = {">=": val >= thr, ">": val > thr,
                       "<=": val <= thr, "<": val < thr}[op]
        if hit:
            return Action(rule["action"]), f"hard_rule:{rule['id']}"
    return None
