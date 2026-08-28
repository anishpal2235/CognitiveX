"""Retrieval verification with honest abstention.

The brief's hardest constraint: there is often no reliable real-time ground
truth. Our answer is not to pretend otherwise. Each checkable claim gets one of
three outcomes:

    supported    -- a trusted source covers and agrees with it
    unsupported  -- a trusted source covers the topic but does not support it
    not_covered  -- NO trusted source speaks to this at all  -> ABSTAIN

Policy then decides what abstention means: fatal in a regulated flow, a soft
caveat in an internal one. Systems that collapse "not covered" into "fine" are
exactly the systems that under-flag.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..config import DATA
from ..schemas import Completion, InterceptedRequest, RiskDim, Signal, Verdict
from .base import timed
from .performance import EMB

_SENT = re.compile(r"(?<=[.!?])\s+")
_CHECKABLE = re.compile(r"\b\d|\bsection\b|\baccording to\b|\bpolicy\b|%", re.I)


class KB:
    """Tiny in-memory index over the GOVERNED document set only.

    Documents from loosely-governed sources are deliberately excluded:
    verifying against untrusted text manufactures false confidence, which is
    worse than admitting ignorance.
    """

    def __init__(self, folder: Path | None = None):
        folder = folder or (DATA / "kb")
        self.chunks: list[str] = []
        self.sources: list[str] = []
        if folder.exists():
            for f in sorted(folder.glob("*.md")):
                for para in [p.strip() for p in f.read_text(encoding="utf-8").split("\n\n")]:
                    if len(para) > 40:
                        self.chunks.append(para)
                        self.sources.append(f.name)

        # Cache the corpus matrix when the embedder has a stable vector space.
        self._cached = EMB.model is not None
        self.V = EMB.encode(self.chunks) if (self._cached and self.chunks) else None

        # TF-IDF fallback: fit the vectoriser ONCE on the corpus and reuse it.
        # The previous implementation re-fit over corpus+query for EVERY claim,
        # so verification cost grew linearly with the number of claims -- the
        # riskiest responses (more claims) were the slowest to check, which is
        # exactly backwards for an inline gate.
        self._vec = None
        self._M = None
        if not self._cached and self.chunks:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
            try:
                M = self._vec.fit_transform(self.chunks).toarray()
                norms = np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
                self._M = M / norms
            except ValueError:
                self._vec, self._M = None, None

    def top(self, q: str, k: int = 3) -> list[tuple[float, str, str]]:
        if not self.chunks:
            return []
        if self._cached and self.V is not None:
            qv = EMB.encode([q])[0]
            sims = self.V @ qv
        elif self._vec is not None and self._M is not None:
            # transform() only -- no refit. This is the hot path.
            qv = self._vec.transform([q]).toarray()[0]
            qv = qv / (np.linalg.norm(qv) + 1e-9)
            sims = self._M @ qv
        else:
            M = EMB.encode(self.chunks + [q])   # last-resort joint fit
            qv, C = M[-1], M[:-1]
            sims = C @ qv
        idx = np.argsort(-sims)[:k]
        return [(float(sims[i]), self.chunks[i], self.sources[i]) for i in idx]


KB_INDEX = KB()


class GroundingDetector:
    name, dim, tier, est_ms = "grounding", RiskDim.GROUNDING, 1, 60

    async def run(self, req: InterceptedRequest, comp: Completion) -> Signal:
        with timed() as t:
            # Only verify sentences that ASSERT something checkable. Hedged prose
            # and pleasantries are not grounding failures.
            claims = [s for s in _SENT.split(comp.text) if _CHECKABLE.search(s)]
            has_claims = bool(claims)
            if has_claims:
                # Calibrate to the embedder ACTUALLY IN USE. Char n-gram TF-IDF
                # cosines run systematically lower than sentence-transformer
                # cosines, so one fixed pair of cutoffs cannot serve both
                # backends: on the fallback it silently manufactures
                # "not_covered" verdicts for text the KB genuinely supports.
                # That is a false-positive factory, and precisely the alert
                # fatigue failure mode this system claims to prevent.
                sup_th, unsup_th = (
                    (0.55, 0.30) if EMB.backend == "sentence-transformers"
                    else (0.26, 0.14)
                )
                unsupported, not_covered, details = 0, 0, []
                for c in claims:
                    hits = KB_INDEX.top(c, k=2)
                    best = hits[0][0] if hits else 0.0
                    if best >= sup_th:
                        status = "supported"
                    elif best >= unsup_th:
                        status = "unsupported"
                        unsupported += 1
                    else:
                        status = "not_covered"
                        not_covered += 1
                    details.append(
                        {
                            "claim": c[:120],
                            "sim": round(best, 3),
                            "status": status,
                            "source": hits[0][2] if hits else None,
                        }
                    )
                n = len(claims)
                # not_covered is weighted lower than unsupported: unknown is bad,
                # contradicted-by-a-source is worse.
                score = (unsupported * 1.0 + not_covered * 0.6) / n
                verdict = (
                    Verdict.FAIL if score > 0.6
                    else Verdict.ABSTAIN if (not_covered > unsupported and score > 0.25)
                    else Verdict.SUSPECT if score > 0.25
                    else Verdict.PASS
                )

        if not has_claims:
            return Signal(
                detector=self.name,
                dim=self.dim,
                score=0.05,
                verdict=Verdict.PASS,
                confidence=0.5,
                latency_ms=t.ms,
                evidence={"checkable_claims": 0},
            )

        return Signal(
            detector=self.name,
            dim=self.dim,
            score=round(min(score, 1.0), 3),
            verdict=verdict,
            confidence=0.7,
            latency_ms=t.ms,
            evidence={
                "claims": details[:6],
                "n_claims": n,
                "unsupported": unsupported,
                "not_covered": not_covered,
                "kb_chunks": len(KB_INDEX.chunks),
                # Record WHICH calibration produced this verdict, so an auditor
                # replaying the decision knows the thresholds in force.
                "embedder": EMB.backend,
                "thresholds": {"supported": sup_th, "unsupported": unsup_th},
            },
        )
