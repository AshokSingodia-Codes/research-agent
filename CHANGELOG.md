# Changelog

All notable changes to the `research-agent` platform will be documented in this file.

## [Phase 6 / C4] - Offline Benchmark & Evaluation Harness
- Created curated dataset of 16 research benchmarks across technical deep dives, adversarial edge cases, and contradiction detection (`app/evals/datasets/core.jsonl`).
- Implemented quantitative scorers for citation validity, plan-to-report coverage, source domain diversity, budget adherence, and contradiction detection.
- Built naive single-prompt baseline runner measuring empirical statistical lift.
- Created `app/evals/cli.py` generating Markdown (`docs/eval_report.md`) and JSON (`docs/eval_report.json`) benchmark telemetry.
- Added comprehensive unit tests in `tests/unit/test_evals.py`.

## [Phase 7 / C5] - Production Packaging, Docker & Polish
- Created multi-stage production `Dockerfile` with non-root security context and built-in healthchecks.
- Created production `docker-compose.yml` orchestrating API, Redis Stream Worker, and Redis 7 container with health checks.
- Documented evaluation results, statistical lift, and deployment architecture in `README.md`.

## [Phase 4] - Durable State in Redis & Chaos Resilience
- Implemented `RedisCheckpointSaver` extending LangGraph `BaseCheckpointSaver` to persist state checkpoints after every node.
- Added automatic disk snapshot synchronization for fake Redis mode to guarantee cross-process durability.
- Implemented idempotency key manager derived from `(run_id, node, input_hash)` to prevent duplicate LLM/tool execution.
- Added distributed locking per run using Redis `SET NX PX` and configurable TTLs for finished runs.
- Built background worker with startup recovery for interrupted workflows.
- Verified with subprocess chaos test: worker process killed via `SIGKILL` mid-run, restarted in separate process, completed without replaying finished nodes.

## [Phase 3] - Live Search with Provenance & Parallel Fan-Out
- Integrated search and extraction engine with URL canonicalization and SHA-256 content deduplication.
- Built readability-style full-page extraction using `trafilatura` with token truncation cap.
- Implemented per-domain result caps (default 2 per sub-question) and `DomainCircuitBreaker`.
- Handled all failure paths with structured `ToolResult` (ok, no_results, rate_limited, paywalled, timeout, blocked, parse_error).
- Added parallel fan-out/fan-in runner respecting `depends_on` with atomic budget locks and deterministic sorting.
- Implemented `ContradictionDetector` surfacing opposing claims across sources in the writer's report.
- Added source quality heuristic scoring.

## [Phase 2] - Supervisor Graph with Hard Budgets
- Implemented `BudgetGuard` in code enforcing max sub-questions, searches, tokens, USD cost, wall-clock timeout, and repair attempts.
- Implemented LangGraph StateGraph (`plan -> dispatch -> research -> review -> (loop | write) -> critic -> finish`).
- Built `Supervisor` review specialist enforcing strict 1-retry maximum per sub-question.
- Built `Critic` specialist assessing draft report coverage against plan and ungrounded claims.
- Verified independent triggers for each budget constraint yielding graceful partial report degradation (`partial=true`).
- Verified property-based graph termination over randomized executions with zero infinite loops.

## [Phase 1] - Typed Specialists & LLM Provider Abstraction
- Built `app/llm/` provider abstraction (`LLMProvider` ABC, `AnthropicProvider`, `GroqProvider`, `FakeProvider`).
- Integrated per-provider pricing tables and accurate token/USD cost accounting.
- Implemented Pydantic v2 schemas for all hand-offs (`ResearchPlan`, `SubQuestion`, `ResearchFindings`, `Finding`, `Report`, `ReviewDecision`, `Contradiction`).
- Implemented `Planner`, `Researcher`, and `Writer` specialists with retry-with-repair (max 2 repairs).
- Verified with comprehensive test suite including rule-based quality checks across 5 technical questions.
