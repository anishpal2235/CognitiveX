from __future__ import annotations

from typing import Protocol

from ..schemas import Completion, InterceptedRequest, ModelSpec


class Provider(Protocol):
    """The only thing the control plane needs from a model vendor.

    Keeping this surface tiny is what makes the layer vendor-neutral: adding a
    new provider is one file, not a refactor.
    """

    async def generate(
        self,
        req: InterceptedRequest,
        spec: ModelSpec,
        n_samples: int = 1,
        temperature: float = 0.7,
    ) -> Completion:
        ...
