# Multi-Agent Research Assistant

[![Live Frontend](https://img.shields.io/badge/Frontend-Vercel-black?style=flat&logo=vercel)](https://research-agent-sooty-rho.vercel.app)
[![Live Backend API](https://img.shields.io/badge/API-Render-46E3B7?style=flat&logo=render)](https://research-agent-api-49ii.onrender.com/health)
[![CI](https://github.com/AshokSingodia-Codes/research-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/AshokSingodia-Codes/research-agent/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)]()
[![Framework](https://img.shields.io/badge/LangGraph-0.2.60-orange.svg)]()
[![State Store](https://img.shields.io/badge/Redis-Durable%20Checkpoints-red.svg)]()
[![API](https://img.shields.io/badge/FastAPI-Production-teal.svg)]()

> **Production-grade multi-agent research architecture with supervisor-worker topology, durable state persistence in Redis, dynamic parallel fan-out, multi-provider LLM fallback cascade, citation verification, hard budget circuit breakers, and real-time step-level tracing.**

---

## 🌐 Live Deployments

- **Frontend (Vercel)**: [https://research-agent-sooty-rho.vercel.app](https://research-agent-sooty-rho.vercel.app)
- **Backend API (Render)**: [https://research-agent-api-49ii.onrender.com](https://research-agent-api-49ii.onrender.com)
- **Healthcheck Endpoint**: [https://research-agent-api-49ii.onrender.com/health](https://research-agent-api-49ii.onrender.com/health)
- **Prometheus Metrics**: [https://research-agent-api-49ii.onrender.com/metrics](https://research-agent-api-49ii.onrender.com/metrics)

---

## 💡 Why This Project Signals (Production AI Engineering)

Most agent demonstrations fail on process restart, hallucinate sources, loop endlessly, or burn API credits uncontrollably. This system solves each of these real-world production engineering challenges:

1. **Durable State & Crash Resumption**: Powered by LangGraph state checkpointing backed by Redis. If a worker process crashes, is killed, or restarts mid-workflow, it automatically reclaims and resumes from the exact last completed node without replaying expensive LLM or search steps.
2. **Instant Asynchronous Cancellation**: Clicking "Cancel" immediately aborts active LangGraph threads, web searches, and synthesis loops in real-time. It records a permanent `cancelled` state, releases worker concurrency slots, and prevents phantom report overwrites.
3. **Multi-Provider LLM Fallback Cascade**: If the primary LLM provider (e.g. Groq) encounters a 429 rate limit or outage, the request automatically falls back in real time through Anthropic, Gemini, OpenRouter, or Grok without failing the user's research run.
4. **Citation Existence Validation & Contradiction Detection**: Citations are verified against retrieved findings (validating citation IDs and source URLs exist). Conflicting source claims are synthesized with balance, and citation links are verified before report compilation.
5. **Deterministic Budget Guards**: Token limits, search caps, wall-clock timeouts, and strict revision limits are enforced directly at graph edges—never delegated to probabilistic LLM prompts.
6. **Step-Level Observability (`GET /research/{id}/trace`)**: Full audit trail recording specialist node execution, exact inputs/outputs, tool invocations, token costs, and per-step latencies.

---

## 🏛️ System Architecture & Workflow

```
                                      +---------------------------------------------+
                                      |             Frontend (Vercel)               |
                                      |  Minimalist Calm UI + EventSource SSE stream |
                                      +----------------------+----------------------+
                                                             |
                                           POST /research    | SSE Progress Stream
                                                             v
+----------------------------------------------------------------------------------------------------+
|                                    Backend API Server (Render)                                     |
|                                                                                                    |
|  1. Ingests topic -> Generates run_id -> Validates auth (X-API-Key) & sliding-window rate limit   |
|  2. Saves initial state to Redis -> Pushes job to Redis Stream ('research:jobs')                  |
|  3. Returns 202 Accepted immediately                                                               |
+--------------------------------------------------+-------------------------------------------------+
                                                   |
                                                   v
+----------------------------------------------------------------------------------------------------+
|                                    Stream Worker & LangGraph Engine                                |
|                                                                                                    |
|                                      +------------------------+                                    |
|                                      |      Planner Node      |                                    |
|                                      | Breaks topic into 3-5  |                                    |
|                                      | sub-questions with deps|                                    |
|                                      +-----------+------------+                                    |
|                                                  |                                                 |
|                                                  v                                                 |
|                                      +------------------------+                                    |
|                                      | Dynamic Fan-Out Branch |                                    |
|                                      +-----------+------------+                                    |
|                             +--------------------+--------------------+                            |
|                             |                    |                    |                            |
|                             v                    v                    v                            |
|                     +---------------+    +---------------+    +---------------+                    |
|                     | Researcher #1 |    | Researcher #2 |    | Researcher #3 |                    |
|                     | (Tavily Live) |    | (Tavily Live) |    | (Tavily Live) |                    |
|                     +-------+-------+    +-------+-------+    +-------+-------+                    |
|                             |                    |                    |                            |
|                             +--------------------+--------------------+                            |
|                                                  |                                                 |
|                                                  v                                                 |
|                                      +------------------------+                                    |
|                                      |    Supervisor Node     |                                    |
|                                      | Evaluates evidence,    |                                    |
|                                      | checks contradictions, |                                    |
|                                      | enforces max 1 retry   |                                    |
|                                      +-----------+------------+                                    |
|                                                  |                                                 |
|                                                  v                                                 |
|                                      +------------------------+                                    |
|                                      |      Writer Node       |                                    |
|                                      | Academic report format |                                    |
|                                      | with [n] inline claims |                                    |
|                                      +-----------+------------+                                    |
|                                                  |                                                 |
|                                                  v                                                 |
|                                      +------------------------+                                    |
|                                      |   Citation Validator   |                                    |
|                                      | Validates citation existence & URLs |                               |
|                                      +-----------+------------+                                    |
|                                                  |                                                 |
+--------------------------------------------------+-------------------------------------------------+
                                                   |
                                                   v
+----------------------------------------------------------------------------------------------------+
|                                    Redis Persistence & State Store                                 |
|                                                                                                    |
|  * Checkpointer: Saves state snapshots after EVERY node execution                                 |
|  * Stream Queue: Consumer group coordination with XAUTOCLAIM dead-letter recovery                  |
|  * Audit Log: Token accounting, USD cost metrics, step-by-step latency trace                       |
+----------------------------------------------------------------------------------------------------+
```

---

## ⚙️ How Things Work Under the Hood

### 1. Request Ingestion & Asynchronous Decoupling
- The client sends `POST /research` with `{ "question": "..." }`.
- FastAPI authenticates the caller via constant-time `hmac.compare_digest` against `API_KEY` and checks sliding-window rate limits (10 req/min).
- Initial state is persisted to Redis **before** enqueuing to prevent race conditions.
- The job is published to the Redis Stream `research:jobs` and the API responds immediately with `202 Accepted` and a `run_id`.

### 2. Autonomous Specialist Agents
- **Planner**: Deconstructs broad questions into 3–5 targeted sub-questions with explicit dependency graph routing.
- **Dynamic Researcher Fan-Out**: Dispatches parallel sub-questions across concurrent researcher coroutines. Each researcher queries the Tavily API, performs SSRF-filtered page extraction, parses readable HTML, and scores domain quality.
- **Supervisor**: Evaluates the collected evidence against adequacy criteria and detects planted contradictions between opposing sources. It triggers a maximum of 1 targeted revision pass if critical gaps exist.
- **Writer**: Synthesizes facts into structured Markdown sections with executive summaries and bracketed `[n]` citation footnotes.
- **Citation Validator**: Scans all `[n]` claims, validates that they correspond to fetched source URLs, and calculates citation coverage.

### 3. Fault-Tolerant LangGraph Checkpointing & Recovery
- At each node transition, state is serialized and persisted to Redis key `checkpoint:{run_id}:{step_index}`.
- If a server worker restarts, `resume_interrupted_runs()` inspects unfinished streams via `XAUTOCLAIM`, restores the exact checkpoint, and continues from the next pending node.

### 4. Real-Time Cancellation
- When the user clicks **Cancel**, a `POST /research/{id}/cancel` sets an atomic cancellation key `research:cancel:{run_id}` in Redis.
- Background search loops and LangGraph nodes check this flag every 500ms and break immediately.
- The state transitions to `RunStatus.CANCELLED`, worker tasks terminate cleanly, and the frontend resets its state.

---

## 🚀 API Reference

### `POST /research`
Initiates a new research run.
```bash
curl -X POST "https://research-agent-api-49ii.onrender.com/research" \
     -H "Content-Type: application/json" \
     -H "X-API-Key: $API_KEY" \
     -d '{"question": "Recent breakthroughs in solid-state battery technology"}'
```
**Response (202 Accepted):**
```json
{
  "run_id": "run_9a1f28b7c012",
  "status": "queued",
  "message": "Research run queued successfully."
}
```

### `GET /research/{id}/stream`
Real-time Server-Sent Events (SSE) progress feed:
```bash
curl -N "https://research-agent-api-49ii.onrender.com/research/run_9a1f28b7c012/stream" \
     -H "X-API-Key: $API_KEY"
```
*(For browser `EventSource` connections, acquire a 60-second single-use token via `POST /research/{id}/stream-token` with the `X-API-Key` header and connect via `?token=...`)*

### `GET /research/{id}`
Polls current run status or retrieves the completed report:
```json
{
  "run_id": "run_9a1f28b7c012",
  "status": "completed",
  "progress_percentage": 100.0,
  "completed_subquestions": 3,
  "total_subquestions": 3,
  "total_tokens": 4200,
  "total_cost_usd": 0.0042,
  "report": {
    "topic": "Solid-State Battery Breakthroughs",
    "executive_summary": "...",
    "sections": [
      {
        "heading": "Electrolyte Innovations",
        "content": "Sulfide-based solid electrolytes demonstrate ionic conductivities comparable to liquid electrolytes [1]."
      }
    ],
    "citations": [
      {
        "citation_id": "1",
        "source_url": "https://nature.com/articles/...",
        "claim_supported": "Sulfide-based solid electrolytes demonstrate ionic conductivities..."
      }
    ]
  }
}
```

### `POST /research/{id}/cancel`
Cancels a running workflow immediately.
```bash
curl -X POST "https://research-agent-api-49ii.onrender.com/research/run_9a1f28b7c012/cancel" \
     -H "X-API-Key: $API_KEY"
```

### `GET /research/{id}/trace`
Step-level execution trace with latency waterfalls and token attribution:
```json
{
  "run_id": "run_9a1f28b7c012",
  "total_steps": 5,
  "total_tokens": 4200,
  "total_latency_ms": 1420.5,
  "steps": [
    {
      "step_id": "plan_0",
      "node_name": "planner",
      "latency_ms": 180.2,
      "tokens_used": 1100,
      "input_summary": "Question: Solid-State Battery Breakthroughs",
      "output_summary": "Generated 3 subquestions"
    }
  ]
}
```

---

## 🧪 Testing & Verification

The test suite includes unit, integration, SSRF security, idempotency, and chaos recovery tests:

```bash
# Run all tests
pytest tests/ -v

# Run with test coverage
pytest --cov=app --cov-report=term-missing tests/
```

**Results:**
- CI Step 1 (Unit & Integration, FakeRedis): `129 passed, 3 skipped, 3 deselected in 18.84s`
- CI Step 2 (Real Redis, Chaos & E2E): `5 passed, 130 deselected in 7.60s`
- **Total:** `134 passed`, `0 failed`
- **Coverage:** **76% total code coverage** across 4,115 statements.

---

## 🔒 Production Deployment & Environment Configuration

### Render Configuration (`render.yaml`)

| Variable | Service | Description | Example / Default |
| :--- | :--- | :--- | :--- |
| `ENV` | API & Worker | Execution environment | `production` |
| `API_KEY` | API | Production secret key ($\ge 16$ chars) | Generated 32-char token |
| `STREAM_TOKEN_SECRET` | API | Secret for signing short-lived HMAC stream tokens | Generated 32-char token |
| `CORS_ORIGINS` | API | Allowed frontend origins (JSON array, no wildcard `*`) | `["https://research-agent-sooty-rho.vercel.app"]` |
| `REDIS_URL` | API & Worker | Real Redis connection string | `redis://multiagent-research-redis:6379/0` |
| `USE_FAKE_REDIS` | API & Worker | Persistence mode | `false` |
| `RUN_WORKER_INPROCESS`| API | Run background worker in separate process | `false` |
| `GROQ_API_KEY` | API & Worker | Groq LLM API key | Sync from Render secret |
| `ANTHROPIC_API_KEY` | API & Worker | Anthropic LLM API key | Sync from Render secret |
| `TAVILY_API_KEY` | API & Worker | Tavily web search API key | Sync from Render secret |

### Vercel Frontend Configuration
The frontend is hosted as static assets on Vercel (`static/index.html`, `static/js/app.js`, `static/css/styles.css`).
- `window.API_BASE_URL` points to the deployed Render backend (e.g. `https://research-agent-api-49ii.onrender.com`).
- **Visitor Access**: Neither `app.js` nor Vercel embeds the private `API_KEY`. Visitors can click the **🔑 API Key** button in the header or enter a key when prompted (persisted client-side in `localStorage['multiagent_api_key']`).

---

## 🛠️ Local Development

```bash
# 1. Clone repository
git clone https://github.com/AshokSingodia-Codes/research-agent.git
cd research-agent

# 2. Set up virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env

# 4. Start local development server
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
Open [http://localhost:8000](http://localhost:8000) in your browser.

---

## 📦 Docker Deployment

```bash
# Run API, Worker, and Redis with a single command:
docker compose up -d --build
```
