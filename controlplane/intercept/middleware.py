from __future__ import annotations

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class TraceMiddleware(BaseHTTPMiddleware):
    """Assigns a correlation id and records the wall-clock overhead of the whole
    control plane on every response header.

    This exists to make the guardrail tax measurable rather than arguable: a
    skeptical platform owner can curl the gateway and read the cost in
    milliseconds off `x-controlplane-total-ms`.
    """

    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = rid
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["x-request-id"] = rid
        response.headers["x-controlplane-total-ms"] = str(
            int((time.perf_counter() - t0) * 1000)
        )
        return response
