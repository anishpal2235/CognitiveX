"""Context vector for the bandit.

Cheap, interpretable, deterministic. Every dimension is something a human can
name -- which matters when you have to explain a routing decision in an audit.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

from ..schemas import InterceptedRequest

# Keep D small: LinUCB needs O(D^2) memory per arm and converges faster.
D = 16

_CODE = re.compile(r"```|def |SELECT |import ")
_MATH = re.compile(r"\d+\s*[-+*/%]\s*\d+|calculate|how much|percent")
_PII_HINT = re.compile(r"email|phone|address|account number|aadhaar|pan\b", re.I)


def featurize(req: InterceptedRequest) -> np.ndarray:
    p = req.prompt
    n_tok = len(p) // 4
    x = np.zeros(D, dtype=float)
    x[0] = 1.0                                        # bias
    x[1] = min(n_tok / 512, 1.0)                      # length
    x[2] = 1.0 if _CODE.search(p) else 0.0            # code-like
    x[3] = 1.0 if _MATH.search(p) else 0.0            # numeric reasoning
    x[4] = 1.0 if "?" in p else 0.0                   # question vs instruction
    x[5] = 1.0 if _PII_HINT.search(p) else 0.0        # asks about personal data
    x[6] = min(len(req.messages) / 10, 1.0)           # conversation depth
    x[7] = 1.0 if req.is_agentic else 0.0             # output drives an action
    x[8] = 0.0 if req.reversible else 1.0             # irreversibility
    x[9] = {"support_bot": 1.0, "internal_copilot": 0.5,
            "credit_decision": 0.0}[req.use_case.value]
    x[10] = {"public": 0.0, "internal": 0.33,
             "confidential": 0.66, "regulated": 1.0}.get(req.data_class, 0.33)
    x[11] = 1.0 if req.geo == "EU" else 0.0

    # Hashed topic bucket: gives the bandit some domain-specialisation ability
    # without a vocabulary or an embedding model in the hot path.
    h = int(hashlib.md5(" ".join(p.lower().split()[:6]).encode()).hexdigest(), 16)
    x[12 + (h % 4)] = 1.0
    return x
