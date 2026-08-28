"""The contracts. Every other module depends only on these types, which is what
makes detectors, providers and policies hot-swappable.

The hardest part of a guardrail system is not the detectors, it is agreeing on
what a "risk" IS, so that policy, action and audit can all reason about the same
object.
"""
from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ----------------------------- Enums -----------------------------
class UseCase(str, Enum):
    SUPPORT_BOT = "support_bot"
    INTERNAL_COPILOT = "internal_copilot"
    CREDIT_DECISION = "credit_decision"


class RiskDim(str, Enum):
    """Deliberately OVERLAPPING dimensions. A fabricated fact about a named
    person scores on BOTH hallucination and privacy; we never force a single
    label. Clean categorisation is unnecessary if you fuse instead of label."""

    HALLUCINATION = "hallucination"
    GROUNDING = "grounding"
    PRIVACY = "privacy"
    BIAS = "bias"
    SAFETY = "safety"
    COST = "cost"
    COMPOUNDING = "compounding"   # multi-turn / agentic escalation


class Action(str, Enum):
    ALLOW = "allow"
    ANNOTATE = "annotate"       # ship it, attach a visible caveat
    REPAIR = "repair"           # auto-edit: redact, hedge, strip claims
    ESCALATE = "escalate"       # human in the loop before delivery
    BLOCK = "block"


class Verdict(str, Enum):
    PASS = "pass"
    SUSPECT = "suspect"
    FAIL = "fail"
    ABSTAIN = "abstain"         # detector could not decide -> NOT a pass
    TIMEOUT = "timeout"


# ----------------------------- Request side -----------------------------
class Turn(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class InterceptedRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    session_id: str = "anon"
    tenant: str = "acme"
    use_case: UseCase = UseCase.INTERNAL_COPILOT
    geo: str = "IN"                        # drives regulatory profile
    data_class: str = "internal"           # public|internal|confidential|regulated
    messages: list[Turn]
    requested_model: Optional[str] = None   # app's hint; router may override
    max_tokens: int = 512
    stream: bool = False
    is_agentic: bool = False                # output triggers a tool/action
    reversible: bool = True                 # can the downstream effect be undone?
    created_at: float = Field(default_factory=time.time)

    @property
    def prompt(self) -> str:
        return self.messages[-1].content if self.messages else ""


# ----------------------------- Model side -----------------------------
class ModelSpec(BaseModel):
    name: str
    provider: str
    usd_per_1k_in: float
    usd_per_1k_out: float
    p50_latency_ms: int
    quality_prior: float                    # 0-1, from benchmarks / offline prefs
    max_data_class: str = "confidential"     # sensitivity ceiling
    regions: list[str] = ["IN", "EU", "US"]  # data residency allowlist
    tags: list[str] = []


class Completion(BaseModel):
    model: str
    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    usd: float = 0.0
    samples: list[str] = []                 # extra stochastic samples for entropy


# ----------------------------- Check side -----------------------------
class Signal(BaseModel):
    """One detector's output. `score` is always 0 (safe) .. 1 (dangerous)."""

    detector: str
    dim: RiskDim
    score: float = 0.0
    verdict: Verdict = Verdict.PASS
    confidence: float = 0.5                 # how much we trust this score
    latency_ms: int = 0
    usd: float = 0.0
    evidence: dict[str, Any] = {}           # spans, quotes, counts -> audit trail


class RiskVector(BaseModel):
    signals: list[Signal] = []
    fused: float = 0.0                      # calibrated 0..1 overall risk
    per_dim: dict[str, float] = {}
    abstained: list[str] = []
    degraded: bool = False                  # some detector timed out


# ----------------------------- Act side -----------------------------
class Decision(BaseModel):
    action: Action
    reason: str
    risk: float
    threshold_hit: Optional[str] = None
    policy_version: str = ""
    repaired_text: Optional[str] = None
    annotations: list[str] = []
    human_ticket: Optional[str] = None


class Trace(BaseModel):
    """The single object written to the audit log. Everything needed to replay
    the decision offline lives here -- that is what makes counterfactual policy
    evaluation possible with zero extra model calls."""

    request: InterceptedRequest
    eligible_models: list[str] = []
    chosen_model: str = ""
    route_reason: str = ""
    completion: Optional[Completion] = None
    risk: Optional[RiskVector] = None
    decision: Optional[Decision] = None
    reward: Optional[float] = None
    total_latency_ms: int = 0
    check_latency_ms: int = 0
    policy_version: str = ""
    budget: dict[str, float] = {}          # live spend / lambda snapshot
