"""Disjoint LinUCB (Li et al., 2010): one ridge regression per model arm.

Why LinUCB and not a neural router:
  * Closed-form updates -- no training loop, no GPU, converges in hundreds of
    requests, which is realistic for a prototype AND for a real pilot.
  * The uncertainty term is EXPLICIT (alpha * sqrt(x' A^-1 x)), so
    "why did you try a new model here?" has a numeric answer. Essential for
    governance; a black-box router is unauditable.
  * Warm-starting from offline preference data is just pre-applying updates.
    No pretraining stage.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np


class LinUCB:
    def __init__(self, arms: list[str], d: int, alpha: float = 0.35):
        self.d, self.alpha = d, alpha
        self.lock = threading.Lock()
        self.A: dict[str, np.ndarray] = {a: np.eye(d) for a in arms}
        self.b: dict[str, np.ndarray] = {a: np.zeros(d) for a in arms}
        self.n: dict[str, int] = {a: 0 for a in arms}

    def _ensure(self, arm: str) -> None:
        if arm not in self.A:
            self.A[arm] = np.eye(self.d)
            self.b[arm] = np.zeros(self.d)
            self.n[arm] = 0

    def theta(self, arm: str) -> np.ndarray:
        self._ensure(arm)
        return np.linalg.solve(self.A[arm], self.b[arm])

    def score(self, arm: str, x: np.ndarray, explore: bool = True) -> tuple[float, float]:
        """Returns (mean, bonus) SEPARATELY, so we can log exactly why an arm won
        -- exploitation or exploration. That distinction is what a governance
        reviewer actually wants to see.
        """
        self._ensure(arm)
        Ainv = np.linalg.inv(self.A[arm])
        mean = float(self.theta(arm) @ x)
        bonus = float(self.alpha * np.sqrt(max(x @ Ainv @ x, 0.0))) if explore else 0.0
        return mean, bonus

    def update(self, arm: str, x: np.ndarray, reward: float) -> None:
        with self.lock:
            self._ensure(arm)
            self.A[arm] += np.outer(x, x)
            self.b[arm] += reward * x
            self.n[arm] += 1

    # ---------- persistence: a router that forgets on restart is a toy ----------
    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "d": self.d,
                    "alpha": self.alpha,
                    "A": {k: v.tolist() for k, v in self.A.items()},
                    "b": {k: v.tolist() for k, v in self.b.items()},
                    "n": self.n,
                }
            )
        )

    @classmethod
    def load(cls, path: str | Path, arms: list[str], d: int) -> "LinUCB":
        p = Path(path)
        if not p.exists():
            return cls(arms, d)
        try:
            raw = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            return cls(arms, d)
        if int(raw.get("d", d)) != d:
            # Feature schema changed -> old state is meaningless. Start clean
            # rather than silently mixing incompatible geometries.
            return cls(arms, d)
        m = cls(list(raw["A"].keys()) or arms, raw["d"], raw["alpha"])
        m.A = {k: np.array(v) for k, v in raw["A"].items()}
        m.b = {k: np.array(v) for k, v in raw["b"].items()}
        m.n = raw.get("n", {k: 0 for k in m.A})
        for a in arms:
            m._ensure(a)
        return m
