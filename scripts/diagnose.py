"""Diagnose WHERE the false positives actually come from.

`metrics.confusion()` reports ONE aggregate FPR using
flag_actions=("repair", "escalate", "block"). That single number cannot tell you
which of two very different problems you have:

  (a) DEFINITION -- `repair` is counted as a positive detection even though it
      still DELIVERS an answer. A repair on safe text is a cosmetic edit, not a
      withheld response. Counting it identically to a block inflates FPR.

  (b) SCORING    -- genuinely safe responses are scored too HIGH and land on the
      repair rung. This is a real precision defect in the detectors or in the
      policy multipliers.

(a) is fixed by reporting the right metric. (b) is fixed by changing code or
thresholds. Conflating them means you cannot tell whether you have a measurement
problem or a model problem, so this script separates them.

Stdlib only -- reads controlplane.db directly, so it runs with the API down.

Usage:
    python -m scripts.diagnose
    python -m scripts.diagnose --db controlplane.db --target-fpr 0.15
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

# Actions that WITHHOLD or DEFER the answer. These are the ones that cost a
# human something, so this is the strict definition of "we raised an alarm".
WITHHOLD = ("escalate", "block")
# The definition metrics.confusion() uses by default.
ALL_FLAGS = ("repair", "escalate", "block")


def confusion(rows: list[dict], flags: tuple[str, ...]) -> dict:
    tp = sum(1 for r in rows if r["action"] in flags and r["unsafe"])
    fp = sum(1 for r in rows if r["action"] in flags and not r["unsafe"])
    fn = sum(1 for r in rows if r["action"] not in flags and r["unsafe"])
    tn = sum(1 for r in rows if r["action"] not in flags and not r["unsafe"])
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {
        "n": len(rows),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "fpr": round(fp / max(fp + tn, 1), 4),
        "fnr": round(fn / max(fn + tp, 1), 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4),
    }


def _fmt(d: dict) -> str:
    return (
        f"n={d['n']:<4} tp={d['tp']:<4} fp={d['fp']:<4} fn={d['fn']:<4} "
        f"tn={d['tn']:<4} FPR={d['fpr']:<7} FNR={d['fnr']:<7} "
        f"prec={d['precision']:<7} rec={d['recall']:<7} f1={d['f1']}"
    )


def load(db_path: str) -> list[dict]:
    if not Path(db_path).exists():
        print(f"No database at {db_path!r}. Run the simulation first.")
        return []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(
            """
        SELECT l.request_id            AS request_id,
               l.human_says_unsafe     AS unsafe,
               t.use_case              AS use_case,
               t.geo                   AS geo,
               t.data_class            AS data_class,
               t.action                AS action,
               t.risk                  AS risk,
               t.model                 AS model,
               t.payload               AS payload
        FROM labels l
        JOIN traces t ON t.request_id = l.request_id
            """
        )
    except sqlite3.OperationalError as exc:
        print(f"Could not read {db_path!r}: {exc}")
        return []
    out = []
    for r in cur.fetchall():
        d = dict(r)
        d["unsafe"] = bool(d["unsafe"])
        try:
            payload = json.loads(d.pop("payload") or "{}")
            d["per_dim"] = (payload.get("risk") or {}).get("per_dim", {}) or {}
            d["abstained"] = (payload.get("risk") or {}).get("abstained", []) or []
            d["reason"] = (payload.get("decision") or {}).get("reason", "")
        except (ValueError, TypeError):
            d["per_dim"], d["abstained"], d["reason"] = {}, [], ""
        out.append(d)
    return out


def sweep(rows: list[dict], target_fpr: float) -> None:
    """Sweep a single flag boundary over the fused risk score.

    This answers the question the brief actually asks: given a false-positive
    budget, where does the boundary go, and what does it cost in recall?
    """
    print("\n=== OPERATING CURVE (single cut on fused risk) ===")
    print(f"{'thr':>6} {'FPR':>8} {'FNR':>8} {'recall':>8} {'precision':>10}")
    best = None
    for i in range(5, 100, 5):
        thr = i / 100
        tp = sum(1 for r in rows if r["risk"] >= thr and r["unsafe"])
        fp = sum(1 for r in rows if r["risk"] >= thr and not r["unsafe"])
        fn = sum(1 for r in rows if r["risk"] < thr and r["unsafe"])
        tn = sum(1 for r in rows if r["risk"] < thr and not r["unsafe"])
        fpr = fp / max(fp + tn, 1)
        fnr = fn / max(fn + tp, 1)
        rec = tp / max(tp + fn, 1)
        prec = tp / max(tp + fp, 1)
        print(f"{thr:>6.2f} {fpr:>8.3f} {fnr:>8.3f} {rec:>8.3f} {prec:>10.3f}")
        if fpr <= target_fpr and (best is None or fnr < best[1]):
            best = (thr, fnr, rec, fpr)
    print()
    if best:
        print(
            f"--> lowest-FNR cut respecting FPR<={target_fpr}: risk>={best[0]:.2f} "
            f"(FPR={best[3]:.3f}, recall={best[2]:.3f})"
        )
    else:
        print(
            f"--> NO threshold achieves FPR<={target_fpr}. The score itself does "
            f"not separate safe from unsafe well enough; retuning the ladder "
            f"cannot fix this. Fix detector precision first."
        )


def group_table(rows: list[dict], keyfn, title: str, flags: tuple[str, ...]) -> None:
    print(f"\n=== {title} ===")
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[keyfn(r)].append(r)
    print(
        f"{'group':<34} {'n':>4} {'safe':>5} {'FP':>4} {'FPR':>7} {'meanRisk(safe)':>15}"
    )
    for k in sorted(groups, key=lambda g: -len(groups[g])):
        rs = groups[k]
        safe = [r for r in rs if not r["unsafe"]]
        fp = sum(1 for r in safe if r["action"] in flags)
        mean_safe_risk = sum(r["risk"] for r in safe) / len(safe) if safe else 0.0
        fpr = fp / len(safe) if safe else 0.0
        print(
            f"{k:<34} {len(rs):>4} {len(safe):>5} {fp:>4} {fpr:>7.3f} "
            f"{mean_safe_risk:>15.3f}"
        )


def per_dim_contrast(rows: list[dict], flags: tuple[str, ...]) -> None:
    """Which DETECTOR is driving the false positives?

    Compares mean per-dimension risk on false positives against true negatives.
    The dimension with the largest gap is the one manufacturing false alarms --
    that is where to spend your next fix.
    """
    fps = [r for r in rows if not r["unsafe"] and r["action"] in flags]
    tns = [r for r in rows if not r["unsafe"] and r["action"] not in flags]
    tps = [r for r in rows if r["unsafe"] and r["action"] in flags]
    if not fps:
        print("\n=== PER-DIMENSION CONTRAST ===\nNo false positives. Nothing to explain.")
        return

    dims = sorted({d for r in rows for d in r["per_dim"]})

    def mean(rs: list[dict], d: str) -> float:
        vals = [float(r["per_dim"].get(d, 0) or 0) for r in rs]
        return sum(vals) / len(vals) if vals else 0.0

    print("\n=== PER-DIMENSION CONTRAST (which detector causes false alarms) ===")
    print(
        f"{'dimension':<16} {'FP mean':>9} {'TN mean':>9} {'gap':>8} "
        f"{'TP mean':>9} {'discrim':>9}"
    )
    scored = []
    for d in dims:
        f, t, p = mean(fps, d), mean(tns, d), mean(tps, d)
        # gap: how much this dim inflates SAFE responses that got flagged.
        # discrim: how well it separates truly-unsafe from safe-but-flagged.
        scored.append((f - t, d, f, t, p, p - f))
    for gap, d, f, t, p, disc in sorted(scored, reverse=True):
        print(f"{d:<16} {f:>9.3f} {t:>9.3f} {gap:>8.3f} {p:>9.3f} {disc:>9.3f}")
    print(
        "\nRead it this way: a large 'gap' with a small or NEGATIVE 'discrim' means\n"
        "that dimension fires just as hard on safe text as on unsafe text. It is\n"
        "adding risk without adding information -- reduce its fusion weight or\n"
        "fix its precision."
    )


def action_mix(rows: list[dict]) -> None:
    print("\n=== ACTION x GROUND TRUTH ===")
    print(f"{'action':<12} {'safe':>6} {'unsafe':>7}")
    acts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        acts[r["action"]][1 if r["unsafe"] else 0] += 1
    for a in ("allow", "annotate", "repair", "escalate", "block"):
        if a in acts:
            print(f"{a:<12} {acts[a][0]:>6} {acts[a][1]:>7}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="controlplane.db")
    ap.add_argument("--target-fpr", type=float, default=0.15)
    a = ap.parse_args()

    rows = load(a.db)
    if not rows:
        print(
            "No labelled rows joined to traces. Run:\n"
            "    python -m scripts.simulate --n 200 --label"
        )
        return

    base = sum(1 for r in rows if r["unsafe"]) / len(rows)
    print(f"labelled rows: {len(rows)}   true unsafe base rate: {base:.3f}")

    print("\n=== DEFINITION A: flag = repair|escalate|block (what metrics.py reports) ===")
    print(_fmt(confusion(rows, ALL_FLAGS)))
    print("\n=== DEFINITION B: flag = escalate|block (answer withheld or deferred) ===")
    print(_fmt(confusion(rows, WITHHOLD)))
    print(
        "\nIf A looks bad and B looks acceptable, your problem is largely\n"
        "DEFINITIONAL: repair is being counted as a false alarm even though the\n"
        "user still got an answer. Report both, and be explicit about which\n"
        "actions you count. If BOTH look bad, the scores are genuinely inflated."
    )

    action_mix(rows)
    per_dim_contrast(rows, ALL_FLAGS)
    group_table(rows, lambda r: r["use_case"], "FALSE POSITIVES BY USE CASE", ALL_FLAGS)
    group_table(rows, lambda r: r["geo"], "FALSE POSITIVES BY GEO", ALL_FLAGS)
    group_table(
        rows, lambda r: r["data_class"], "FALSE POSITIVES BY DATA CLASS", ALL_FLAGS
    )
    group_table(
        rows,
        lambda r: f"{r['use_case']}/{r['geo']}/{r['data_class']}",
        "FALSE POSITIVES BY POLICY COMBINATION",
        ALL_FLAGS,
    )
    sweep(rows, a.target_fpr)


if __name__ == "__main__":
    main()
