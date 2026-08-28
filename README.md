# 🛡️ ControlPlane.ai — Round 2 Prototype

**A vendor-neutral governance layer that sits between any application and any foundation model.**

```
App ──▶ INTERCEPT ──▶ ROUTE ──▶ [model] ──▶ CHECK ──▶ ACT ──▶ User
         (proxy/SDK)   (bandit +           (parallel,   (graded
                        budget)             tiered)      ladder)
                            ▲                  │
                            └──── reward ◀──────┘
                              Check's scores become
                              Route's reward signal
```

Enterprises run generative AI across many use cases at once — a customer-facing
support bot, an internal copilot, a regulated decision-support tool — and each
carries a different risk signature and a different latency budget. One-size-fits-all
checking fails. This prototype implements a four-stage pipeline where **policy is
data**, **checks run in parallel under a deadline**, and **the safety layer teaches
the routing layer**, so the system gets cheaper and safer the more it is used.

---

## Quick start (60 seconds, no API key)

```bash
python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
# already have a venv? just activate it instead, e.g. source .respo/bin/activate
pip install -r requirements.txt
cp .env.example .env                                    # PROVIDER_MODE=mock by default

python -m scripts.seed                                  # warm-start the router
python -m scripts.simulate --n 200 --label               # generate traffic + ground-truth labels

uvicorn controlplane.app:app --reload --port 8000        # API   → http://localhost:8000/docs
streamlit run dashboard/app.py --server.port 8501        # console → http://localhost:8501
```

Default mode needs **no API key and no network**. The mock provider injects known
failure modes (a fabricated statistic, a leaked email, a biased generalisation) so
the guardrails can be demonstrated firing against ground truth you control — and so
false-positive/negative rates are computed against a real answer key.

### One-request demo

```bash
curl -s localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"use_case":"support_bot","geo":"IN","data_class":"confidential",
       "session_id":"s-1",
       "messages":[{"role":"user","content":"How long do refunds take and who owns my account?"}]}' | jq
```

The response carries a `controlplane` block with the action taken, the fused risk,
the per-dimension breakdown, which models were *eligible*, why one was chosen,
cost, latency, and the check overhead.

---

## The five design claims

| # | Claim | Where it lives |
|---|---|---|
| 1 | **Check's scores become Route's reward.** Safety and savings stop competing: a cheap model that leaks PII earns a negative reward and loses traffic automatically. Nobody writes the rule — the system discovers it. | `feedback/reward.py`, `pipeline.py` |
| 2 | **Abstention is a first-class verdict.** When there is no ground truth, detectors return `ABSTAIN`, not `PASS`. "I could not check this" never renders as "this is safe". | `check/grounding.py`, `act/fusion.py` |
| 3 | **A graded ladder, not a kill switch.** allow → annotate → repair → escalate → block. Four of five rungs still deliver an answer, because correction preserves utility where blocking destroys it. | `act/decision.py`, `act/repair.py` |
| 4 | **Alert budget as a first-class control.** Reviewer attention is a metered resource. Once the budget is spent, borderline cases are repaired instead of escalated — but risk ≥ 0.9 always still surfaces. This directly attacks alert fatigue. | `act/decision.py` (`AlertBudget`) |
| 5 | **Counterfactual policy replay, zero model calls.** Every trace stores the full risk vector, so "what if we tighten support_bot to 0.45?" is answered offline with real FPR/FNR deltas. Policy changes stop being a leap of faith. | `scripts/replay.py` |

---

## How each stage works

### 1 · Intercept
An OpenAI-shaped endpoint plus a drop-in SDK wrapper. Enterprises adopt this by
changing a base URL, not by rewriting applications — zero-friction integration is a
*requirement* for a governance layer, because a painful gate gets routed around and
then governs nothing.

### 2 · Route
Hard policy filter **first**, optimisation second:

- **Eligibility** — data residency, sensitivity ceiling, forbidden tags, latency headroom. An ineligible model is never even scored, so no amount of cost pressure or exploration bonus can smuggle it back in.
- **Selection** — disjoint LinUCB contextual bandit over the eligible set. Closed-form updates (no GPU, no training loop), converges in hundreds of requests, and the uncertainty term is *explicit* — so "why did you try a new model here?" has a numeric answer.
- **Budget** — online knapsack via Lagrangian relaxation. λ is the **shadow price of a dollar**: it rises when spend runs ahead of pace and the router quietly prefers cheaper arms. No hard cutoff, so there is no cliff at the end of the window.
- **Conservative exploration** — learning is confined to reversible, non-agentic, non-regulated traffic. The "never below baseline" promise is enforced structurally.

### 3 · Check
Seven detectors behind one 30-line protocol, run **in parallel, tiered, deadline-bounded**:

| Detector | Signal | Tier |
|---|---|---|
| `pii` | Regex + Luhn validation. Scores **leakage, not presence** — echoing the user's own email back is not an egress event. | 1 |
| `bias` | Protected attribute **co-occurring** with a generaliser or decisional language. A bare mention is not bias. | 1 |
| `grounding` | Retrieval against *governed* sources only. Three outcomes: supported / unsupported / **not_covered → abstain**. | 1 |
| `selfconsistency` | Do the checkable atoms (numbers, dates) agree across samples? Highest-precision cheap hallucination signal. | 1 |
| `conversation` | Compounding risk across turns, amplified for agentic and irreversible output. | 1 |
| `semantic_entropy` | Entropy over **meaning clusters** of k samples — ignores harmless rewording, reacts to factual instability. | 2 |
| `judge` | LLM-as-reviewer against a rubric. Reports the **worst** dimension, never an average. | 2 |

Three latency defences make an inline gate viable: parallelism (a tier costs
`max`, not `sum`), tiering (expensive detectors fire only when cheap ones are
suspicious — clean traffic pays tier-1 cost only), and a hard deadline. On expiry
the risk vector is marked `degraded`, which makes fusion **more** conservative.
Checks fail *closed*.

### 4 · Act
**Fusion** handles the overlap problem the brief raises — a fabricated detail about
a real person is *both* hallucination and privacy. Rather than forcing one label:
confidence-weighted mean per dimension, then blended with the max
(`fused = (1-m)·mean + m·max`). `max_weight` is the explicit over/under-flagging
knob, surfaced as a **policy parameter** instead of buried in code. Hallucination
and privacy both elevated is treated as super-additive.

**Decision** applies the ladder, then hard compliance rules (which can only
*increase* severity — a rule is a floor, never an override), then the irreversible-
action guard, then the alert budget.

### 5 · Learn
Two channels. **Implicit**: every request's Check scores become a bandit reward —
free, immediate, proxy-based. **Explicit**: a reviewer's "this flag was wrong" is
gold-standard supervision that re-updates the bandit *and* feeds the threshold
advisor. The advisor **proposes and never applies** — a guardrail that silently
retunes its own risk appetite is ungovernable.

---

## Metrics for a skeptic

`GET /v1/metrics` and the console's tabs report:

- **Operations** — spend, savings vs a frontier-only counterfactual, p50/p95 latency, check overhead, alert rate.
- **Detection** — FPR / FNR / precision / recall computed **against human labels only**. Grading the detector with the detector measures self-consistency, not correctness.
- **Calibration** — Expected Calibration Error. The entire ladder is thresholds on the fused score; if `risk=0.8` doesn't mean ~80% unsafe, every threshold is arbitrary.
- **Per use case** — because one aggregate number hides the regulated flow that is quietly failing.

Plus `GET /v1/audit/verify`: the audit log is **hash-chained** (`SHA-256(prev + row)`),
so editing any past decision invalidates every hash after it and the verifier names
the first broken index. An integrity check an auditor can watch execute beats a
promise in a policy document.

---

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/chat/completions` | The single gate. OpenAI-shaped. |
| POST | `/v1/feedback` | Submit a human label; updates router + advisor. |
| GET | `/v1/thresholds/{use_case}` | Advisory threshold retuning at a target FPR. |
| GET | `/v1/traces/{request_id}` | Full decision trace with detector evidence. |
| GET | `/v1/traces` | Recent traces. |
| GET | `/v1/metrics` | The trust report. |
| GET | `/v1/audit/verify` | Verify the hash chain. |
| POST | `/v1/policy/reload` | Hot-reload policy — no deploy. |
| GET | `/v1/policy/preview` | The **effective** composed policy for a context. |
| GET | `/v1/router/state` | Budget status and arm pulls. |
| GET | `/health`, `/healthz` | Liveness. |

---

## Running on your own server

### Option A — systemd (recommended for a VM)

```bash
sudo mkdir -p /opt/controlplane /var/lib/controlplane
sudo chown -R $USER /opt/controlplane /var/lib/controlplane
cp -r . /opt/controlplane && cd /opt/controlplane

python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
cp .env.example .env
# then set DB_PATH=/var/lib/controlplane/controlplane.db in .env
./.venv/bin/python -m scripts.seed
```

`/etc/systemd/system/controlplane.service`:

```ini
[Unit]
Description=ControlPlane.ai gateway
After=network.target

[Service]
WorkingDirectory=/opt/controlplane
EnvironmentFile=/opt/controlplane/.env
ExecStart=/opt/controlplane/.venv/bin/uvicorn controlplane.app:app \
  --host 0.0.0.0 --port 8000 --workers 2
Restart=always
User=controlplane

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload && sudo systemctl enable --now controlplane
curl -s localhost:8000/health
```

> **On workers:** the bandit state and the session risk window are per-process.
> With `--workers > 1`, `router_state.json` is written by each worker and the
> compounding-risk window is not shared. For a real deployment move both to Redis
> (see *Known limitations*). For a pilot or a demo, use `--workers 1`.

### Option B — Docker

```bash
docker compose up --build          # API on :8000, console on :8501
```

### Option C — bare

```bash
./run.sh                           # seed + simulate + API + console
VENV=.respo ./run.sh               # or reuse an existing virtualenv by name
```

`run.sh` creates `.venv` by default; set `VENV` to point it at a venv you already
have. It will reuse that one instead of making a second copy of the dependencies.

### Behind nginx

```nginx
location /        { proxy_pass http://127.0.0.1:8000; }
location /console { proxy_pass http://127.0.0.1:8501; proxy_http_version 1.1;
                    proxy_set_header Upgrade $http_upgrade;
                    proxy_set_header Connection "upgrade"; }
```

The console exposes trace contents and should sit behind your SSO, not on the
public internet.

---

## Using real models

1. In `.env`: `PROVIDER_MODE=openai_compat`, set `OPENAI_BASE_URL` and `OPENAI_API_KEY`.
2. In `configs/models.yaml`: change each `provider: mock` → `openai_compat` and set `name` to the real model id (`gpt-4o-mini`, `claude-3-5-haiku`, a vLLM served name…).
3. Keep `max_data_class`, `regions` and `tags` accurate — those are enforced routing constraints, not documentation.
4. `curl -X POST localhost:8000/v1/policy/reload`

Any OpenAI-compatible endpoint works: Azure, Together, Groq, Ollama, vLLM, a
LiteLLM proxy. The gateway only ever inspects the input/output layer, which is
precisely the constraint enterprises face when consuming foundation models by API.

---

## Project layout

```
controlplane-ai/
├── README.md  requirements.txt  .env.example  pytest.ini
├── Dockerfile  docker-compose.yml  run.sh
├── configs/
│   ├── policies.yaml          # governance: ladders, overlays, hard rules
│   └── models.yaml            # catalogue + routing constraints + budget
├── data/
│   ├── kb/                    # governed sources for grounding
│   ├── seed_preferences.csv   # offline warm-start data
│   └── eval_set.jsonl         # scenario set with expectations
├── controlplane/
│   ├── schemas.py             # the contracts everything depends on
│   ├── config.py              # settings + hot-reloadable versioned YAML
│   ├── app.py  pipeline.py    # API surface + the 90-line orchestration
│   ├── intercept/             # middleware + SDK wrapper
│   ├── providers/             # mock (injects known failures) + openai_compat
│   ├── route/                 # features, LinUCB, budget controller, router
│   ├── check/                 # 7 detectors + tiered parallel orchestrator
│   ├── act/                   # fusion, repair, graded ladder
│   ├── policy/                # layered resolution + hard rules
│   ├── feedback/              # reward function + human-label learner
│   ├── observability/         # hash-chained audit + trust metrics
│   └── store/                 # SQLite DAO
├── dashboard/app.py           # Streamlit governance console
├── scripts/                   # seed · simulate (with oracle) · replay
└── tests/                     # 40+ tests pinning the design claims
```

---

## Tests

```bash
pytest -q
```

The tests encode the *design claims*, not just the code. If someone later
"optimises" the PII detector into a naive scanner, `test_pii_discounts_echoed_data`
fails and explains why that is wrong. Notable cases: eligibility never returns
empty (a guardrail that takes the product down gets switched off), escalation never
leaks the risky draft, degraded checks *increase* risk, and the alert budget
downgrades borderline cases while extreme risk always surfaces.

---

## Stated assumptions

The brief invites reasonable assumptions; these are ours.

- **Traffic** ~ tens of thousands of interactions/week across three use cases, mixed 55 / 35 / 10 (support · copilot · regulated).
- **Latency budgets** 800 ms customer-facing, 2 s internal, 8 s regulated.
- **Data sources** a small governed corpus (`data/kb/`) plus loosely-governed sources that are deliberately **excluded** from grounding — verifying against untrusted text manufactures false confidence, which is worse than admitting ignorance.
- **Model access** API-only. We inspect the input/output layer, never model internals.
- **Cost figures** illustrative catalogue prices, not a vendor quote. The savings number is a counterfactual: same traffic, re-priced at the most expensive model.

## Known limitations (and the honest fix)

| Limitation | Consequence | Fix |
|---|---|---|
| Session risk window is in-process | Compounding risk resets on restart; not shared across workers | Redis with a TTL |
| Bandit state is a JSON file | Last writer wins with multiple workers | Redis / Postgres with optimistic locking |
| TF-IDF fallback embedder | Weaker semantic clustering than real embeddings | Install `sentence-transformers` (already wired, auto-detected) |
| Bias detection is lexical | Misses implicit bias with no lexical marker | Counterfactual probing: re-run with the protected attribute swapped, compare outcomes |
| Grounding corpus is tiny | Coverage gaps inflate `not_covered` | Point `data/kb/` at the real governed document set |
| Calibration needs labels | ECE is unavailable until ~10+ human labels exist | Sample 1–5% of traffic for review; the console makes labelling one click |
| SQLite | Fine to a few hundred req/s single-node | Postgres + object storage for payloads |

None of these change the architecture — they are all swaps behind an existing
interface, which is the point of keeping `schemas.py` as the only shared dependency.
