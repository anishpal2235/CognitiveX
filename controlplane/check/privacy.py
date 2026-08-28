"""PII / data-leakage detection.

Two pieces of logic matter far more than the pattern list:

1. LEAKAGE, not mere presence. If the PII already appeared in the user's own
   prompt, echoing it back is not an egress event. That single distinction
   removes a large slice of the false positives naive scanners generate -- and
   false positives are what drive users to bypass the guardrail entirely.

2. VALIDATION. Luhn-checking card numbers suppresses the random-16-digit
   false positives that make PII scanners untrustworthy.
"""
from __future__ import annotations

import re

from ..schemas import Completion, InterceptedRequest, RiskDim, Signal, Verdict
from .base import timed

# India-aware patterns alongside the usual suspects -- geography matters for a
# DPDP-Act deployment as much as for GDPR.
PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]{2,}"),
    "phone_in": re.compile(r"(?:\+91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}"),
    "pan": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    "aadhaar": re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d{4}[\s-]?){3}\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "iban": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,26}\b"),
}

# Not all personal data is equally harmful to leak. A national ID is not an IP.
SEVERITY = {
    "aadhaar": 1.0,
    "credit_card": 1.0,
    "ssn": 1.0,
    "pan": 0.9,
    "iban": 0.9,
    "phone_in": 0.6,
    "email": 0.5,
    "ipv4": 0.3,
}


def _luhn_ok(digits: str) -> bool:
    return (
        sum(
            sum(divmod(int(x) * 2, 10)) if i % 2 else int(x)
            for i, x in enumerate(reversed(digits))
        )
        % 10
        == 0
    )


class PIIDetector:
    name, dim, tier, est_ms = "pii", RiskDim.PRIVACY, 1, 8

    async def run(self, req: InterceptedRequest, comp: Completion) -> Signal:
        with timed() as t:
            prompt_blob = " ".join(m.content for m in req.messages)
            found: list[dict] = []
            score = 0.0
            for kind, pat in PATTERNS.items():
                for m in pat.finditer(comp.text):
                    val = m.group(0)
                    if kind == "credit_card":
                        digits = re.sub(r"\D", "", val)
                        if len(digits) != 16 or not _luhn_ok(digits):
                            continue
                    novel = val not in prompt_blob      # true egress?
                    sev = SEVERITY[kind] * (1.0 if novel else 0.25)
                    score = max(score, sev)
                    found.append(
                        {
                            "type": kind,
                            "span": [m.start(), m.end()],
                            "novel": novel,
                            "masked": val[:2] + "***" + val[-2:],
                        }
                    )
            # Multiple distinct entities in one response is a bulk-egress signal.
            if len(found) > 1:
                score = min(1.0, score + 0.1 * (len(found) - 1))

        return Signal(
            detector=self.name,
            dim=self.dim,
            score=round(score, 3),
            verdict=(
                Verdict.FAIL if score >= 0.9
                else Verdict.SUSPECT if score >= 0.4
                else Verdict.PASS
            ),
            confidence=0.85,   # regex + validation is high-confidence evidence
            latency_ms=t.ms,
            evidence={"entities": found[:10], "n_entities": len(found)},
        )
