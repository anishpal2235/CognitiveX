"""Uncertainty without ground truth.

The core idea: with no reference answer available, use DISAGREEMENT ACROSS
STOCHASTIC SAMPLES as an error proxy. Sampling k answers, clustering them by
MEANING, and taking the entropy over cluster mass gives *semantic* entropy --
it ignores harmless rewording and reacts to genuine factual instability, which
is exactly the discrimination a lexical metric like self-BLEU misses.
"""
from __future__ import annotations

import math
import re

import numpy as np

from ..config import settings
from ..schemas import Completion, InterceptedRequest, RiskDim, Signal, Verdict
from .base import timed

_NUM = re.compile(r"\b\d[\d,.]*\b")


# ---------------- embedding layer with graceful degradation ----------------
class Embedder:
    """sentence-transformers when available, TF-IDF char n-grams otherwise.

    The fallback is weaker but has zero heavy dependencies and no download, so
    the demo never breaks on conference Wi-Fi or an air-gapped server.
    """

    def __init__(self) -> None:
        self.model = None
        self.backend = "tfidf"
        if settings.embed_backend in ("auto", "st"):
            try:
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer("all-MiniLM-L6-v2")
                self.backend = "sentence-transformers"
            except Exception:
                self.model = None

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 8))
        if self.model is not None:
            v = self.model.encode(texts, normalize_embeddings=True)
            return np.asarray(v)

        from sklearn.feature_extraction.text import TfidfVectorizer

        # Char n-grams: robust to short texts and typos, and never throws on an
        # empty vocabulary the way word-level TF-IDF does.
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        try:
            m = vec.fit_transform(texts).toarray()
        except ValueError:
            return np.zeros((len(texts), 8))
        norms = np.linalg.norm(m, axis=1, keepdims=True) + 1e-9
        return m / norms


EMB = Embedder()


def _cluster(texts: list[str], thresh: float = 0.80) -> list[int]:
    """Greedy agglomerative clustering under cosine similarity -- a cheap stand-in
    for bidirectional-entailment clustering. O(k^2) on k <= 8 samples, so the
    cost is negligible next to the generation it is checking.
    """
    if len(texts) <= 1:
        return [0] * len(texts)
    V = EMB.encode(texts)
    labels = [-1] * len(texts)
    centroids: list[np.ndarray] = []
    for i, v in enumerate(V):
        placed = False
        for c, cen in enumerate(centroids):
            if float(v @ cen) >= thresh:
                labels[i] = c
                merged = cen + v
                centroids[c] = merged / (np.linalg.norm(merged) + 1e-9)
                placed = True
                break
        if not placed:
            centroids.append(v)
            labels[i] = len(centroids) - 1
    return labels


class SemanticEntropyDetector:
    """High entropy over MEANING clusters => the model is unstable on substance,
    which correlates with confabulation.

    Tier 2 because it needs the extra samples to exist.
    """

    name, dim, tier, est_ms = "semantic_entropy", RiskDim.HALLUCINATION, 2, 120

    async def run(self, req: InterceptedRequest, comp: Completion) -> Signal:
        with timed() as t:
            texts = [comp.text] + list(comp.samples)
            if len(texts) < 3:
                # Not enough samples to estimate dispersion. ABSTAIN, never PASS:
                # "I could not measure this" is not evidence of safety.
                insufficient = True
            else:
                insufficient = False
                labels = _cluster(texts)
                k = len(set(labels))
                counts = np.bincount(np.array(labels), minlength=k) / len(labels)
                H = -float(sum(p * math.log(p + 1e-12) for p in counts if p > 0))
                Hmax = math.log(len(texts))
                norm = H / Hmax if Hmax > 0 else 0.0

        if insufficient:
            return Signal(
                detector=self.name,
                dim=self.dim,
                verdict=Verdict.ABSTAIN,
                score=0.35,
                confidence=0.2,
                latency_ms=t.ms,
                evidence={"reason": "insufficient_samples", "n_samples": len(texts)},
            )

        return Signal(
            detector=self.name,
            dim=self.dim,
            score=round(min(norm * 1.15, 1.0), 3),
            verdict=(
                Verdict.FAIL if norm > 0.66
                else Verdict.SUSPECT if norm > 0.33
                else Verdict.PASS
            ),
            confidence=min(0.4 + 0.15 * len(texts), 0.9),
            latency_ms=t.ms,
            evidence={
                "clusters": k,
                "n_samples": len(texts),
                "entropy": round(H, 3),
                "normalized": round(norm, 3),
                "embedder": EMB.backend,
            },
        )


class SelfConsistencyDetector:
    """Cheaper cousin: do the CHECKABLE ATOMS (numbers, dates, amounts) agree
    across samples?

    Numeric disagreement is the highest-precision cheap hallucination signal in
    this simulation -- a model that invents a statistic rarely invents the same
    statistic twice, while genuinely known facts are stable across temperatures.
    """

    name, dim, tier, est_ms = "selfconsistency", RiskDim.HALLUCINATION, 1, 15

    async def run(self, req: InterceptedRequest, comp: Completion) -> Signal:
        with timed() as t:
            texts = [comp.text] + list(comp.samples)
            enough = len(texts) >= 2
            if enough:
                sets = [set(_NUM.findall(x)) for x in texts]
                union: set[str] = set().union(*sets)
                inter: set[str] = set.intersection(*sets)
                jac = (len(inter) / len(union)) if union else 1.0
                score = 1.0 - jac
                unstable = sorted(union - inter)

        if not enough:
            return Signal(
                detector=self.name,
                dim=self.dim,
                verdict=Verdict.ABSTAIN,
                score=0.3,
                confidence=0.2,
                latency_ms=t.ms,
                evidence={"reason": "single_sample"},
            )

        return Signal(
            detector=self.name,
            dim=self.dim,
            score=round(score, 3),
            verdict=(
                Verdict.FAIL if score > 0.6
                else Verdict.SUSPECT if score > 0.25
                else Verdict.PASS
            ),
            confidence=0.6,
            latency_ms=t.ms,
            evidence={"unstable_numbers": unstable[:8], "jaccard": round(jac, 3)},
        )
