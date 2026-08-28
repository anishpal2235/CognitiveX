"""Overlapping-risk fusion: many signals -> one calibrated score.

The brief notes that bias, hallucination and privacy overlap in practice. Rather
than forcing a single label, we fuse. Four moves:

1. Per dimension, combine detectors by CONFIDENCE-WEIGHTED MEAN. A
   high-confidence PII regex must not be diluted by a low-confidence judge.
2. Blend the weighted mean across dimensions with the MAX dimension:
       fused = (1 - m) * weighted_mean + m * max
   Pure mean under-reacts to a single catastrophic signal; pure max over-flags.
   `max_weight` is the EXPLICIT KNOB for the brief's over/under-flagging
   dilemma, surfaced as a policy parameter instead of buried in code.
3. CORRELATION BUMP for the overlap case: hallucination AND privacy both
   elevated (a fabricated detail about a real person) is worse than either
   alone, so the combination is super-additive.
4. Multipliers for data class and geography, then a degradation penalty.
"""
from __future__ import annotations

from ..policy.engine import ResolvedPolicy
from ..schemas import RiskDim, RiskVector, Verdict


def fuse(rv: RiskVector, pol: ResolvedPolicy) -> RiskVector:
    by_dim: dict[str, list[tuple[float, float]]] = {}
    for s in rv.signals:
        if s.verdict == Verdict.ABSTAIN:
            # Abstention carries PARTIAL risk. It is not evidence of safety, but
            # it is weaker evidence than a positive detection.
            by_dim.setdefault(s.dim.value, []).append(
                (s.score * 0.7, s.confidence * 0.6)
            )
        else:
            by_dim.setdefault(s.dim.value, []).append(
                (s.score, max(s.confidence, 0.05))
            )

    per_dim: dict[str, float] = {}
    for dim, pairs in by_dim.items():
        wsum = sum(c for _, c in pairs) or 1.0
        per_dim[dim] = sum(sc * c for sc, c in pairs) / wsum

    # Geography: e.g. the EU overlay weights privacy harder (GDPR).
    if RiskDim.PRIVACY.value in per_dim:
        per_dim[RiskDim.PRIVACY.value] = min(
            1.0, per_dim[RiskDim.PRIVACY.value] * pol.privacy_multiplier
        )

    w = pol.fusion_weights
    num = sum(per_dim.get(d, 0.0) * w.get(d, 1.0) for d in per_dim)
    den = sum(w.get(d, 1.0) for d in per_dim) or 1.0
    weighted_mean = num / den
    mx = max(per_dim.values(), default=0.0)

    m = pol.max_weight
    fused = (1 - m) * weighted_mean + m * mx

    # The overlap case from the brief, handled explicitly.
    h = per_dim.get(RiskDim.HALLUCINATION.value, 0.0)
    g = per_dim.get(RiskDim.GROUNDING.value, 0.0)
    p = per_dim.get(RiskDim.PRIVACY.value, 0.0)
    if max(h, g) > 0.4 and p > 0.4:
        fused = min(1.0, fused + 0.12)

    fused *= pol.risk_multiplier

    if rv.degraded:
        fused = min(1.0, fused + 0.10)   # unknown != safe

    rv.per_dim = {k: round(v, 3) for k, v in per_dim.items()}
    rv.fused = round(min(max(fused, 0.0), 1.0), 3)
    return rv
