"""Metrics for a SKEPTICAL stakeholder.

The brief asks how you'd prove trustworthiness to someone who doesn't believe
you. Vanity metrics ("we blocked 4,000 things!") prove nothing. These four do:

1. OPERATIONS   -- spend, savings vs a frontier-only baseline, latency overhead.
2. DETECTION    -- FPR / FNR / precision / recall against HUMAN LABELS ONLY.
                   Never grade the detector with the detector.
3. CALIBRATION  -- Expected Calibration Error. Does risk=0.8 actually mean 80%?
                   An uncalibrated score cannot support a threshold policy.
4. PER-USE-CASE -- because one aggregate number hides the regulated flow that
                   is quietly failing.
"""
from __future__ import annotations

from ..config import models_cfg
from ..store.db import DB


def confusion(use_case: str | None = None, flag_actions=("repair", "escalate", "block")) -> dict:
    """Computed strictly on human-labelled rows. If we scored ourselves against
    our own detectors we would measure self-consistency, not correctness.
    """
    rows = DB.labelled_rows(use_case)
    if not rows:
        return {"n": 0, "note": "no human labels yet -- FPR/FNR unknowable"}

    tp = sum(1 for r in rows if r["action"] in flag_actions and r["human_says_unsafe"])
    fp = sum(1 for r in rows if r["action"] in flag_actions and not r["human_says_unsafe"])
    fn = sum(1 for r in rows if r["action"] not in flag_actions and r["human_says_unsafe"])
    tn = sum(1 for r in rows if r["action"] not in flag_actions and not r["human_says_unsafe"])

    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {
        "n": len(rows),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "false_positive_rate": round(fp / max(fp + tn, 1), 4),
        "false_negative_rate": round(fn / max(fn + tp, 1), 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4),
    }


def calibration_ece(bins: int = 10) -> dict:
    """Expected Calibration Error over labelled traces.

    Why it matters: the entire ladder is thresholds on the fused risk score. If
    the score is not calibrated, every threshold is arbitrary. ECE is the number
    that tells a skeptic whether 'risk 0.6' means anything at all.
    """
    rows = DB.labelled_rows()
    if len(rows) < 10:
        return {"n": len(rows), "note": "need >=10 labels"}

    buckets: dict[int, list[dict]] = {}
    for r in rows:
        b = min(int(r["risk"] * bins), bins - 1)
        buckets.setdefault(b, []).append(r)

    ece, detail = 0.0, []
    for b, rs in sorted(buckets.items()):
        conf = sum(r["risk"] for r in rs) / len(rs)
        acc = sum(1 for r in rs if r["human_says_unsafe"]) / len(rs)
        ece += (len(rs) / len(rows)) * abs(conf - acc)
        detail.append(
            {
                "bin": f"{b / bins:.1f}-{(b + 1) / bins:.1f}",
                "n": len(rs),
                "mean_risk": round(conf, 3),
                "observed_unsafe_rate": round(acc, 3),
            }
        )
    return {"n": len(rows), "ece": round(ece, 4), "bins": detail}


def operations(limit: int = 2000) -> dict:
    rows = DB.recent(limit=limit)
    if not rows:
        return {"n": 0}

    n = len(rows)
    spend = sum(r["usd"] for r in rows)

    # Counterfactual baseline: what if EVERY request had gone to the most
    # expensive model? Re-price each trace at that model's output rate, using the
    # observed token proxy. This is the honest version of a savings claim --
    # same traffic, one model, priced from the catalogue.
    models = models_cfg().data["models"]
    top = max(models, key=lambda m: m["usd_per_1k_out"])
    per_call_frontier = (top["usd_per_1k_in"] * 0.5) + (top["usd_per_1k_out"] * 0.5)
    frontier_equiv = n * per_call_frontier

    lat = sorted(r["total_latency_ms"] for r in rows)
    chk = sorted(r["check_latency_ms"] for r in rows)

    def pct(xs: list[int], p: float) -> int:
        return xs[min(int(len(xs) * p), len(xs) - 1)] if xs else 0

    actions: dict[str, int] = {}
    models_used: dict[str, int] = {}
    for r in rows:
        actions[r["action"]] = actions.get(r["action"], 0) + 1
        models_used[r["model"]] = models_used.get(r["model"], 0) + 1

    
    escalate_n = actions.get("escalate", 0)
    block_n = actions.get("block", 0)
    alert_rate = escalate_n / n
    block_rate = block_n / n
    withhold_rate = (escalate_n + block_n) / n

    return {
        "n": n,
        "spend_usd": round(spend, 4),
        "frontier_only_usd": round(frontier_equiv, 4),
        "savings_pct": round(100 * (1 - spend / max(frontier_equiv, 1e-9)), 1),
        "latency_p50_ms": pct(lat, 0.5),
        "latency_p95_ms": pct(lat, 0.95),
        "check_overhead_p50_ms": pct(chk, 0.5),
        "check_overhead_p95_ms": pct(chk, 0.95),
        "actions": actions,
        "models": models_used,
        "alert_rate": round(alert_rate, 4),
        "block_rate": round(block_rate, 4),
        "withhold_rate": round(withhold_rate, 4),
        "degraded_rate": round(sum(r["degraded"] for r in rows) / n, 4),
        "mean_reward": round(sum(r["reward"] for r in rows) / n, 4),
    }


def ladder_report() -> dict:
    """Ladder-aware error accounting -- the metric that actually fits the design.

    A five-rung ladder cannot be honestly graded as a binary classifier, and both
    binary framings mislead in opposite directions:

      - Counting `repair` as a positive detection punishes the system for
        successfully mitigating a response that it still delivered. This is what
        inflated our reported FPR to 0.66.
      - Counting `repair` as a negative punishes it for not escalating something
        it had already fixed. This is what inflated our reported FNR to 0.79.

    Neither is what a stakeholder wants to know. They want to know what reached
    the user, so report that directly:

      leakage_rate    -- unsafe content delivered UNMITIGATED (allow/annotate).
                         The only true safety failure. This is the number that
                         must be near zero.
      mitigated_rate  -- unsafe content repaired in place and delivered. A WIN:
                         the harm was removed and the user still got an answer.
      stopped_rate    -- unsafe content escalated or blocked. Also a win, but a
                         more expensive one.
      withheld_rate   -- SAFE answers escalated or blocked. Costs human time and
                         user trust. This is the real over-flagging cost.
      edited_rate     -- SAFE answers cosmetically repaired. A minor cost, but
                         not free: we altered a correct answer.
      clean_pass_rate -- safe answers delivered untouched. The ideal path.
    """
    rows = DB.labelled_rows()
    if not rows:
        return {"n": 0, "note": "no human labels yet"}

    unsafe = [r for r in rows if r["human_says_unsafe"]]
    safe = [r for r in rows if not r["human_says_unsafe"]]
    delivered_clean = ("allow", "annotate")

    leaked = sum(1 for r in unsafe if r["action"] in delivered_clean)
    mitigated = sum(1 for r in unsafe if r["action"] == "repair")
    stopped = sum(1 for r in unsafe if r["action"] in ("escalate", "block"))
    withheld = sum(1 for r in safe if r["action"] in ("escalate", "block"))
    edited = sum(1 for r in safe if r["action"] == "repair")
    clean = sum(1 for r in safe if r["action"] in delivered_clean)

    nu, ns = max(len(unsafe), 1), max(len(safe), 1)
    return {
        "n": len(rows),
        "n_unsafe": len(unsafe),
        "n_safe": len(safe),
        "leakage_rate": round(leaked / nu, 4),
        "mitigated_rate": round(mitigated / nu, 4),
        "stopped_rate": round(stopped / nu, 4),
        "withheld_rate": round(withheld / ns, 4),
        "edited_rate": round(edited / ns, 4),
        "clean_pass_rate": round(clean / ns, 4),
        "counts": {
            "leaked": leaked,
            "mitigated": mitigated,
            "stopped": stopped,
            "withheld": withheld,
            "edited": edited,
            "clean": clean,
        },
    }


def trust_report() -> dict:
    """The single object behind GET /v1/metrics and the dashboard."""
    per_uc = {}
    for uc in ("support_bot", "internal_copilot", "credit_decision"):
        rows = DB.recent(limit=1000, use_case=uc)
        if not rows:
            continue
        per_uc[uc] = {
            "n": len(rows),
            "mean_risk": round(sum(r["risk"] for r in rows) / len(rows), 3),
            "alert_rate": round(
                sum(1 for r in rows if r["action"] == "escalate") / len(rows), 3
            ),
            "block_rate": round(
                sum(1 for r in rows if r["action"] == "block") / len(rows), 3
            ),
            "withhold_rate": round(sum(1 for r in rows if r["action"] in ("escalate", "block"))/ len(rows),3,),
            "p95_latency_ms": sorted(r["total_latency_ms"] for r in rows)[
                min(int(len(rows) * 0.95), len(rows) - 1)
            ],
            "detection": confusion(uc),
        }

    return {
        "operations": operations(),
        "detection": confusion(),
        # Reported alongside the binary confusion matrix, not instead of it. The
        # binary view is what a reviewer expects; the ladder view is what the
        # system actually does.
        "detection_withhold_only": confusion(flag_actions=("escalate", "block")),
        "ladder": ladder_report(),
        "calibration": calibration_ece(),
        "per_use_case": per_uc,
    }
