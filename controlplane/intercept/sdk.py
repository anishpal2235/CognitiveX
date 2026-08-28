"""Drop-in client for teams that prefer an SDK to a proxy.

Migration cost for an existing app is one line:

    - from openai import OpenAI
    - client = OpenAI()
    + from controlplane.intercept.sdk import ControlPlaneClient as OpenAI
    + client = OpenAI(use_case="support_bot", geo="IN")

Backwards-compatible adoption is the whole point. Nobody rewrites their
application to buy a guardrail.
"""
from __future__ import annotations

from typing import Any

import httpx


class ControlPlaneClient:
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        use_case: str = "internal_copilot",
        geo: str = "IN",
        data_class: str = "internal",
        tenant: str = "acme",
    ):
        self.base_url = base_url.rstrip("/")
        self.ctx = {
            "use_case": use_case,
            "geo": geo,
            "data_class": data_class,
            "tenant": tenant,
        }

    def chat(
        self,
        messages: list[dict[str, str]],
        session_id: str = "anon",
        model: str | None = None,
        is_agentic: bool = False,
        reversible: bool = True,
        **kw: Any,
    ) -> dict[str, Any]:
        body = {
            "messages": messages,
            "session_id": session_id,
            "requested_model": model,
            "is_agentic": is_agentic,
            "reversible": reversible,
            **self.ctx,
            **kw,
        }
        r = httpx.post(f"{self.base_url}/v1/chat/completions", json=body, timeout=120)
        r.raise_for_status()
        return r.json()
