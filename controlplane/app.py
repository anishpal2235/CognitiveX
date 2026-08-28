"""FastAPI gateway.

The request path is OpenAI-shaped on purpose: an enterprise adopts this by
changing a base URL, not by rewriting applications. Zero-friction adoption is a
product requirement for a governance layer -- if integration is painful, teams
route around it and the layer governs nothing.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import __version__
from .config import policies, reload_configs
from .intercept.middleware import TraceMiddleware
from .observability import audit, metrics
from .pipeline import handle
from .policy.engine import resolve
from .feedback.learner import learner
from .route.router import router as model_router
from .schemas import InterceptedRequest
from .store.db import DB

app = FastAPI(
    title="ControlPlane.ai",
    version=__version__,
    description="Intercept -> Route -> Check -> Act gateway for enterprise LLM traffic.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(TraceMiddleware)


# ----------------------------- inference ----------------------------------
@app.post("/v1/chat/completions")
async def chat(req: InterceptedRequest):
    """The single gate. Every governed call goes through here."""
    trace = await handle(req)
    d = trace.decision
    return {
        "id": req.request_id,
        "object": "chat.completion",
        "model": trace.chosen_model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": d.repaired_text},
                "finish_reason": "stop",
            }
        ],
        # Everything a caller needs to render the decision transparently.
        "controlplane": {
            "action": d.action.value,
            "risk": trace.risk.fused,
            "per_dim": trace.risk.per_dim,
            "annotations": d.annotations,
            "reason": d.reason,
            "human_ticket": d.human_ticket,
            "policy_version": trace.policy_version,
            "model": trace.chosen_model,
            "eligible_models": trace.eligible_models,
            "route_reason": trace.route_reason,
            "usd": round(trace.completion.usd, 6),
            "latency_ms": trace.total_latency_ms,
            "check_overhead_ms": trace.check_latency_ms,
            "degraded": trace.risk.degraded,
            "abstained": trace.risk.abstained,
            "budget": trace.budget,
        },
    }


# ----------------------------- feedback -----------------------------------
class FeedbackIn(BaseModel):
    request_id: str
    was_correct_flag: bool
    true_quality: float | None = None


@app.post("/v1/feedback")
async def feedback(f: FeedbackIn):
    """Human labels: the gold-standard signal. Closes the loop from reviewer
    judgement back into both the router and the threshold tuner."""
    out = learner.apply_human_label(f.request_id, f.was_correct_flag, f.true_quality)
    if not out["ok"]:
        raise HTTPException(404, out.get("reason", "not found"))
    return out


@app.get("/v1/thresholds/{use_case}")
async def thresholds(use_case: str, target_fpr: float = 0.05):
    """Advisory retuning of the over/under-flagging tradeoff."""
    return learner.suggest_thresholds(use_case, target_fpr)


# ----------------------------- observability -------------------------------
@app.get("/v1/traces/{request_id}")
async def get_trace(request_id: str):
    import json

    row = DB.get_trace(request_id)
    if not row:
        raise HTTPException(404, "unknown request_id")
    return json.loads(row["payload"])


@app.get("/v1/traces")
async def list_traces(limit: int = 50, use_case: str | None = None):
    return {"traces": DB.recent(limit=limit, use_case=use_case)}


@app.get("/v1/metrics")
async def get_metrics():
    """The trust report -- FPR/FNR from human labels, calibration ECE, spend,
    savings and latency overhead. Built for a skeptic."""
    return metrics.trust_report()


@app.get("/v1/audit/verify")
async def verify_audit():
    """Live tamper-evidence check over the hash chain."""
    return audit.verify_chain()


# ----------------------------- governance ---------------------------------
@app.post("/v1/policy/reload")
async def policy_reload():
    """Hot-reload policy without a deploy. Regulations change faster than
    release trains; rigid hard-coded rules age badly."""
    reload_configs()
    model_router.__init__()      # re-read the model catalogue
    return {"ok": True, "policy_version": policies().data["version"]}


@app.get("/v1/policy/preview")
async def policy_preview(
    use_case: str = "support_bot",
    geo: str = "IN",
    data_class: str = "confidential",
):
    """Shows the EFFECTIVE composed policy for a context -- the answer to
    'what rules actually apply to this request?'"""
    p = resolve(use_case, geo, data_class)
    return p.model_dump() if hasattr(p, "model_dump") else p.__dict__


@app.get("/v1/router/state")
async def router_state():
    return {
        "budget": model_router.budget.status(),
        "pulls": model_router.bandit.n,
        "models": [m.name for m in model_router.models],
    }


# ----------------------------- health -------------------------------------
@app.get("/health")
async def health():
    return {"ok": True, "version": __version__, "policy": policies().data["version"]}


@app.get("/healthz")
async def healthz():
    """Alias for k8s-style probes."""
    return await health()
