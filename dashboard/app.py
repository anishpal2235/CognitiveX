"""Streamlit governance console.

The point of this screen is not pretty charts. It is that a compliance owner --
not an engineer -- can answer four questions without asking anyone:

  1. What is this costing, and what did routing save?
  2. How often do we flag, and are the flags right? (FPR / FNR from labels)
  3. Is the risk score calibrated enough to justify the thresholds?
  4. Can I see, and verify, exactly why a specific response was blocked?

Run:  streamlit run dashboard/app.py --server.port 8501
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make the package importable when Streamlit is launched from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from controlplane.config import policies                    # noqa: E402
from controlplane.feedback.learner import learner           # noqa: E402
from controlplane.observability import audit, metrics       # noqa: E402
from controlplane.route.router import router                # noqa: E402
from controlplane.store.db import DB                        # noqa: E402

st.set_page_config(page_title="ControlPlane.ai", page_icon="🛡️", layout="wide")
st.title("🛡️ ControlPlane.ai — Governance Console")
st.caption(
    f"Policy `{policies().version}`  ·  Intercept → Route → Check → Act"
)

rep = metrics.trust_report()
ops = rep["operations"]

if not ops.get("n"):
    st.warning(
        "No traffic yet. Run `python -m scripts.simulate --n 200` to populate "
        "the trace store, then refresh."
    )
    st.stop()

# ---------------------------------------------------------------- KPIs
c = st.columns(6)
c[0].metric("Requests", ops["n"])
c[1].metric(
    "Spend",
    f"${ops['spend_usd']:.3f}",
    delta=f"-{ops['savings_pct']}% vs frontier-only",
)
c[2].metric("p95 latency", f"{ops['latency_p95_ms']} ms")
c[3].metric("Check overhead p95", f"{ops['check_overhead_p95_ms']} ms")
c[4].metric("Alert rate", f"{ops['alert_rate'] * 100:.1f}%")
c[5].metric("Mean reward", ops["mean_reward"])

tabs = st.tabs(
    ["Operations", "Detection quality", "Calibration", "Trace explorer",
     "Audit integrity", "Router"]
)

# ---------------------------------------------------------------- Operations
with tabs[0]:
    left, right = st.columns(2)
    with left:
        st.subheader("Action mix")
        st.caption(
            "Four of five rungs still deliver an answer. A healthy system lives "
            "mostly in allow/annotate/repair — heavy blocking means the "
            "thresholds are mistuned."
        )
        st.bar_chart(pd.Series(ops["actions"], name="requests"))
    with right:
        st.subheader("Traffic by model")
        st.caption(
            "The router should concentrate cheap models on easy traffic and "
            "reserve premium models for hard or sensitive requests."
        )
        st.bar_chart(pd.Series(ops["models"], name="requests"))

    st.subheader("Per use case")
    st.caption(
        "One aggregate number hides the regulated flow that is quietly failing."
    )
    rows = []
    for uc, v in rep["per_use_case"].items():
        det = v.get("detection", {})
        rows.append(
            {
                "use_case": uc,
                "n": v["n"],
                "mean_risk": v["mean_risk"],
                "alert_rate": v["alert_rate"],
                "p95_ms": v["p95_latency_ms"],
                "FPR": det.get("false_positive_rate"),
                "FNR": det.get("false_negative_rate"),
                "labels": det.get("n", 0),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    df = pd.DataFrame(DB.recent(limit=500))
    if not df.empty:
        st.subheader("Risk over time")
        st.line_chart(df.sort_values("ts")[["risk"]].reset_index(drop=True))

# ---------------------------------------------------------------- Detection
with tabs[1]:
    st.subheader("Measured against human labels only")
    st.caption(
        "Grading the detector with the detector measures self-consistency, not "
        "correctness. Every number here comes from a human label."
    )
    det = rep["detection"]
    if not det.get("n"):
        st.info(
            "No human labels yet — FPR/FNR are genuinely unknowable. Run "
            "`python -m scripts.simulate --n 200 --label` to attach oracle labels."
        )
    else:
        k = st.columns(5)
        k[0].metric("Labels", det["n"])
        k[1].metric("False positive rate", det["false_positive_rate"])
        k[2].metric("False negative rate", det["false_negative_rate"])
        k[3].metric("Precision", det["precision"])
        k[4].metric("Recall", det["recall"])
        st.json({k2: det[k2] for k2 in ("tp", "fp", "fn", "tn", "f1")})

        st.divider()
        st.subheader("Threshold advisor")
        st.caption(
            "Turns the over/under-flagging tradeoff into a reproducible number. "
            "Advisory only — a human applies it by editing policies.yaml, which "
            "keeps the change reviewable and versioned."
        )
        uc = st.selectbox(
            "Use case",
            ["support_bot", "internal_copilot", "credit_decision"],
        )
        fpr = st.slider("Target false positive rate", 0.01, 0.30, 0.05, 0.01)
        sug = learner.suggest_thresholds(uc, fpr)
        st.json(sug)
        if sug.get("operating_curve"):
            st.line_chart(
                pd.DataFrame(sug["operating_curve"]).set_index("thr")[["fpr", "fnr"]]
            )

# ---------------------------------------------------------------- Calibration
with tabs[2]:
    cal = rep["calibration"]
    st.subheader("Expected Calibration Error")
    st.caption(
        "The whole ladder is thresholds on the fused risk score. If risk=0.8 "
        "does not mean roughly 80% unsafe, every threshold is arbitrary."
    )
    if "ece" not in cal:
        st.info(cal.get("note", "not enough labels"))
    else:
        st.metric("ECE (lower is better)", cal["ece"])
        cdf = pd.DataFrame(cal["bins"]).set_index("bin")
        st.bar_chart(cdf[["mean_risk", "observed_unsafe_rate"]])
        st.dataframe(cdf, use_container_width=True)

# ---------------------------------------------------------------- Traces
with tabs[3]:
    st.subheader("Why did the system do that?")
    recent = DB.recent(limit=200)
    labels = {
        f"{r['action']:<9} risk={r['risk']:.2f}  {r['use_case']}  {r['request_id'][:8]}": r[
            "request_id"
        ]
        for r in recent
    }
    pick = st.selectbox("Trace", list(labels.keys()))
    if pick:
        row = DB.get_trace(labels[pick])
        trace = json.loads(row["payload"])

        a, b = st.columns(2)
        with a:
            st.markdown("**Prompt**")
            st.code(trace["request"]["messages"][-1]["content"])
            st.markdown("**Raw model output**")
            st.code(trace["completion"]["text"])
        with b:
            st.markdown("**Delivered to user**")
            st.code(trace["decision"]["repaired_text"])
            st.markdown("**Decision**")
            st.json(
                {
                    "action": trace["decision"]["action"],
                    "reason": trace["decision"]["reason"],
                    "risk": trace["risk"]["fused"],
                    "per_dim": trace["risk"]["per_dim"],
                    "annotations": trace["decision"]["annotations"],
                    "policy_version": trace["policy_version"],
                    "model": trace["chosen_model"],
                    "route_reason": trace["route_reason"],
                }
            )

        st.markdown("**Detector evidence**")
        st.caption(
            "Spans, similarity scores and cluster counts — the raw material "
            "behind the decision, kept so it can be re-examined months later."
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "detector": s["detector"],
                        "dim": s["dim"],
                        "score": s["score"],
                        "verdict": s["verdict"],
                        "conf": s["confidence"],
                        "ms": s["latency_ms"],
                        "evidence": json.dumps(s["evidence"])[:160],
                    }
                    for s in trace["risk"]["signals"]
                ]
            ),
            use_container_width=True,
        )

        st.divider()
        st.markdown("**Was this decision right?**")
        st.caption("Human labels feed both the router and the threshold advisor.")
        col1, col2 = st.columns(2)
        if col1.button("👍 Correct call"):
            st.json(learner.apply_human_label(trace["request"]["request_id"], True))
        if col2.button("👎 Wrong call"):
            st.json(learner.apply_human_label(trace["request"]["request_id"], False))

# ---------------------------------------------------------------- Audit
with tabs[4]:
    st.subheader("Tamper-evident audit chain")
    st.caption(
        "Each row hashes the previous one, so editing any past decision "
        "invalidates every hash after it. An integrity check an auditor can "
        "watch execute beats a promise in a policy document."
    )
    if st.button("Verify chain now"):
        res = audit.verify_chain()
        (st.success if res["ok"] else st.error)(res)
    st.dataframe(
        pd.DataFrame(
            [
                dict(r)
                for r in DB.conn.execute(
                    "SELECT idx, request_id, prev_hash, row_hash "
                    "FROM audit ORDER BY idx DESC LIMIT 25"
                ).fetchall()
            ]
        ),
        use_container_width=True,
    )

# ---------------------------------------------------------------- Router
with tabs[5]:
    st.subheader("Budget controller")
    st.caption(
        "λ is the shadow price of a dollar. It rises when spend is off-pace and "
        "the router quietly prefers cheaper arms — no cliff at month end."
    )
    st.json(router.budget.status())
    st.subheader("Arm pulls")
    st.caption("How much evidence the bandit has gathered per model.")
    st.bar_chart(pd.Series(router.bandit.n, name="pulls"))
