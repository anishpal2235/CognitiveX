"""Real model calls against anything speaking the /v1/chat/completions dialect:
OpenAI, Groq, Together, Fireworks, vLLM, Ollama, LM Studio.

Vendor neutrality is a config value here, not a rewrite.
"""
from __future__ import annotations

import asyncio
import time

import httpx

from ..config import settings
from ..schemas import Completion, InterceptedRequest, ModelSpec


class OpenAICompatProvider:
    def __init__(self, base_url: str | None = None, api_key: str | None = None):
        self.base_url = (base_url or settings.openai_base_url).rstrip("/")
        self.api_key = api_key or settings.openai_api_key

    async def _one(
        self,
        client: httpx.AsyncClient,
        req: InterceptedRequest,
        spec: ModelSpec,
        temperature: float,
    ) -> tuple[str, int, int]:
        payload = {
            "model": spec.name,
            "messages": [m.model_dump() for m in req.messages],
            "max_tokens": req.max_tokens,
            "temperature": temperature,
        }
        r = await client.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=60,
        )
        r.raise_for_status()
        d = r.json()
        usage = d.get("usage", {}) or {}
        return (
            d["choices"][0]["message"]["content"],
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
        )

    async def generate(
        self,
        req: InterceptedRequest,
        spec: ModelSpec,
        n_samples: int = 1,
        temperature: float = 0.7,
    ) -> Completion:
        """Primary answer at low temperature, uncertainty samples hotter -- and
        all of them in ONE concurrent fan-out, so semantic entropy costs wall
        clock latency once rather than n times.
        """
        t0 = time.perf_counter()
        async with httpx.AsyncClient() as client:
            tasks = [self._one(client, req, spec, 0.2)]
            tasks += [
                self._one(client, req, spec, 1.0) for _ in range(max(0, n_samples - 1))
            ]
            out = await asyncio.gather(*tasks, return_exceptions=True)

        ok = [o for o in out if not isinstance(o, Exception)]
        if not ok:
            raise RuntimeError(f"all generations failed for {spec.name}: {out[0]}")

        text, tin, tout = ok[0]
        samples = [o[0] for o in ok[1:]]
        usd = (tin / 1000) * spec.usd_per_1k_in + (tout / 1000) * spec.usd_per_1k_out
        usd *= 1 + len(samples)
        return Completion(
            model=spec.name,
            text=text,
            samples=samples,
            tokens_in=tin,
            tokens_out=tout,
            latency_ms=int((time.perf_counter() - t0) * 1000),
            usd=usd,
        )
