"""Warm-start the router from offline preference data.

Why this matters: a cold bandit explores badly for its first few hundred
requests, and "we made your system worse for a week while it learned" is not an
acceptable enterprise pitch. Offline preferences (benchmark wins, historical
thumbs-up, a one-off eval run) are replayed as bandit updates BEFORE any live
traffic, so the router starts at incumbent quality instead of at random.

Usage:  python -m scripts.seed
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from controlplane.config import DATA                       # noqa: E402
from controlplane.route.features import featurize          # noqa: E402
from controlplane.route.router import router               # noqa: E402
from controlplane.schemas import InterceptedRequest, Turn, UseCase  # noqa: E402


def main(repeats: int = 12) -> None:
    path = DATA / "seed_preferences.csv"
    if not path.exists():
        print(f"no seed file at {path}")
        return

    rows = list(csv.DictReader(path.open()))
    n = 0
    # Each preference row is replayed several times: it represents an aggregate
    # judgement over many similar prompts, not a single observation.
    for _ in range(repeats):
        for r in rows:
            req = InterceptedRequest(
                use_case=UseCase(r["use_case"]),
                geo=r.get("geo", "IN"),
                data_class=r.get("data_class", "internal"),
                messages=[Turn(role="user", content=r["prompt"])],
            )
            x = featurize(req)
            router.bandit.update(r["model"], x, float(r["reward"]))
            n += 1

    router.bandit.save(router.state_path)
    print(f"seeded {n} preference updates across {len(rows)} rows")
    print("arm pulls:", router.bandit.n)
    print(f"state written to {router.state_path}")


if __name__ == "__main__":
    main()
