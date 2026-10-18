# Multi-Agent Research Assistant

[![Tests](https://img.shields.io/badge/tests-20%20passed-brightgreen.svg)]()
[![Coverage](https://img.shields.io/badge/coverage-86%25-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)]()
[![Framework](https://img.shields.io/badge/LangGraph-0.2.60-orange.svg)]()
[![State Store](https://img.shields.io/badge/Redis-Durable%20Checkpoints-red.svg)]()
[![API](https://img.shields.io/badge/FastAPI-Production-teal.svg)]()

> **Production-grade multi-agent research architecture with supervisor-worker topology, durable state persistence in Redis, hard budget circuit breakers, and step-level tracing.**

---

## 💡 Why This Project Signals

Most agent demos fail on restart, loop infinitely, or cannot explain their decisions. This project is built around the three hallmarks of production AI systems:

1. **Durable State & Resumption**: Powered by LangGraph state checkpointing in Redis. A crashed or restarted process resumes from the exact last completed node instead of replaying earlier steps.
2. **Hard Budget Enforcement**: Token limits, search caps, wall-clock ceilings, and a strict **1-revision limit** are enforced as deterministic graph edge guards—never entrusted to LLM prompt instructions.
3. **Step-Level Observability (`GET /research/{id}/trace`)**: Full audit trail recording which specialist ran, exact inputs/outputs, tool invocations, token consumption, and per-step latencies.

---

## 🏛️ System Architecture

```
User Request ──► POST /research ──► FastAPI (returns run_id immediately)
                                        │
                                        ▼ (Background Execution)
                             ┌───────────────────────┐
                             │     Planner Agent     │
                             │ (Structured Sub-Qs)   │
                             └──────────┬────────────┘
                                        │
                                        ▼
                             ┌───────────────────────┐
                     ┌───────│   Supervisor Agent    │◄──────┐
                     │       │ (Budget & Quality)    │       │
                     │       └──────────┬────────────┘       │
                     │ (Retry max 1x)   │ (Next Sub-Q)       │
                     │                  ▼                    │
                     │       ┌───────────────────────┐       │
                     └──────►│   Researcher Agent    ├───────┘
                             │ (Tavily + Page Fetch) │
                             └──────────┬────────────┘
                                        │ (All Sub-Qs Done)
                                        ▼
                             ┌───────────────────────┐
                             │     Writer Agent      │
                             │ (Synthesis & Report)  │
                             └──────────┬────────────┘
                                        │
                                        ▼
                             ┌───────────────────────┐
                             │  Citation Validator   │
                             │ (Grounding Provenance)│
                             └───────────────────────┘
                                        ▲
                                        │ Checkpoint State & Traces
                             ┌──────────┴────────────┐
                             │ Redis / Memory Saver  │
                             └───────────────────────┘
```

---

## 🚀 Quickstart

### 1. Installation
```bash
python -m venv .venv
.\.venv\Scripts\activate   # Windows
source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

### 2. Environment Configuration
Copy `.env.example` to `.env`:
```env
GROQ_API_KEY=your_groq_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
TAVILY_API_KEY=your_tavily_key_here
USE_FAKE_REDIS=true
```

#### Configuration Reference Table

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `ENV` | `development` | Deployment environment (`development`, `staging`, `production`) |
| `API_KEY` | `test-key-12345` | Static authentication key required for API endpoints |
| `CORS_ORIGINS` | `["http://localhost:8000","http://127.0.0.1:8000"]` | Allowed CORS origin whitelist |
| `CHECKPOINT_HMAC_KEY` | `""` | Optional secret for HMAC-SHA256 checkpoint tamper validation |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Global fallback Groq model |
| `ANTHROPIC_MODEL` | `claude-3-5-sonnet-20241022` | Global fallback Anthropic model |
| `PLANNER_PROVIDER` | `groq` | Provider for Planner agent (`groq`, `anthropic`, `fake`) |
| `PLANNER_MODEL` | `llama-3.3-70b-versatile` | Model requested by Planner agent |
| `RESEARCHER_PROVIDER` | `groq` | Provider for Researcher agent (`groq`, `anthropic`, `fake`) |
| `RESEARCHER_MODEL` | `llama-3.3-70b-versatile` | Model requested by Researcher agent |
| `SUPERVISOR_PROVIDER` | `groq` | Provider for Supervisor agent (`groq`, `anthropic`, `fake`) |
| `SUPERVISOR_MODEL` | `llama-3.3-70b-versatile` | Model requested by Supervisor agent |
| `WRITER_PROVIDER` | `groq` | Provider for Writer agent (`groq`, `anthropic`, `fake`) |
| `WRITER_MODEL` | `llama-3.3-70b-versatile` | Model requested by Writer agent |
| `CRITIC_PROVIDER` | `groq` | Provider for Critic agent (`groq`, `anthropic`, `fake`) |
| `CRITIC_MODEL` | `llama-3.3-70b-versatile` | Model requested by Critic agent |
| `MAX_CONCURRENT_RUNS` | `5` | Maximum concurrent research workflow jobs processed globally across workers |
| `WORKER_VISIBILITY_TIMEOUT` | `60.0` | Visibility timeout in seconds before abandoned stream messages are eligible for reclaim |
| `RUN_WORKER_INPROCESS` | `false` | When true, spawns a background stream worker in-process for tests and local dev |
| `MAX_TOTAL_TOKENS` | `50000` | Hard budget limit for total tokens consumed |
| `MAX_USD_COST` | `1.00` | Hard budget dollar spending ceiling |
| `WALL_CLOCK_TIMEOUT_SECONDS` | `240.0` | Execution time limit in seconds |
| `MAX_REVISION_LOOPS` | `1` | Strict ceiling on supervisor review retry iterations |

### 3. Run the Complete Automation Test Suite (20 Tests · 86% Coverage)
```bash
# Run all tests with code coverage & HTML report
python run_tests.py

# Filter by tier:
python run_tests.py --unit         # Unit tests (specialists & extraction tools)
python run_tests.py --budget       # Hard budget & circuit breaker tests
python run_tests.py --durability   # Redis crash & intermediate resumption tests
python run_tests.py --integration  # LangGraph & FastAPI lifecycle tests

# View generated HTML coverage report:
start coverage_html/index.html     # Windows
open coverage_html/index.html      # macOS
```

### 4. Start the Application & Visual Dashboard
```bash
.\.venv\Scripts\uvicorn.exe src.api.main:app --host 127.0.0.1 --port 8000 --reload
```
Open **`http://127.0.0.1:8000`** in your browser to access the interactive dashboard.

---

## 📡 API Reference

### `POST /research`
Initiates a new research run. Returns `202 Accepted` immediately.
```json
// Request Body
{
  "topic": "Distributed Consensus in Multi-Raft Architectures"
}

// Response (202 Accepted)
{
  "run_id": "run_9a1f28b7c012",
  "status": "queued",
  "message": "Research workflow dispatched."
}
```

### `GET /research/{id}`
Returns real-time progress percentage, current node, and the final validated report.
```json
{
  "run_id": "run_9a1f28b7c012",
  "status": "completed",
  "progress_percentage": 100.0,
  "completed_subquestions": 3,
  "total_subquestions": 3,
  "total_tokens": 4200,
  "report": {
    "topic": "Distributed Consensus in Multi-Raft Architectures",
    "executive_summary": "...",
    "sections": [...],
    "citations": [
      {
        "citation_id": "[1]",
        "source_url": "https://arxiv.org/abs/2401.0001",
        "domain": "arxiv.org",
        "claim_supported": "Multi-Raft shards data into disjoint consensus groups.",
        "is_verified": true
      }
    ]
  }
}
```

### `GET /research/{id}/trace`
Step-level execution trace with token attribution and latency waterfall.
```json
{
  "run_id": "run_9a1f28b7c012",
  "total_steps": 7,
  "total_tokens": 4200,
  "total_latency_ms": 1280.5,
  "steps": [
    {
      "step_id": "plan_1728840000",
      "node_name": "planner",
      "latency_ms": 115.4,
      "tokens_used": 1200,
      "input_summary": "Topic: Distributed Consensus...",
      "output_summary": "Generated 3 sub-questions: ['sq_1', 'sq_2', 'sq_3']",
      "tool_calls": []
    },
    {
      "step_id": "research_sq_1_1728840001",
      "node_name": "researcher",
      "latency_ms": 320.1,
      "tokens_used": 850,
      "input_summary": "SubQuestion [sq_1]...",
      "output_summary": "Collected 3 findings across ['arxiv.org', 'github.com']",
      "tool_calls": [
        {
          "tool": "search_and_extract",
          "query": "What are the core technical architectures...",
          "latency_ms": 280.0,
          "status": "success",
          "result_count": 3
        }
      ]
    }
  ]
}
```

---

## 🎯 Interview Talking Points

- **"How do you debug an agent that returned a bad answer?"**  
  *Answer:* "We expose `GET /research/{id}/trace`. Instead of guessing, I inspect the step log to determine whether the failure occurred at the Planner (vague sub-questions), the Researcher (bad search query or zero results), the Supervisor (missed deficiency), or the Writer (citation misattribution)."
- **"How do you prevent agents from burning through API budgets?"**  
  *Answer:* "Budgets are enforced in code at graph edges, not in system prompts. We cap max sub-questions, maximum searches per question, global token ceiling, and hard-limit supervisor review retries to exactly 1 pass."
- **"What happens if your worker crashes during a multi-minute research run?"**  
  *Answer:* "Because LangGraph checkpoints state after every node into Redis, restarting the task with the same `run_id` resumes execution from the latest unfinished node without replaying previous expensive search and LLM calls."
