# Redis Schema Specification

This document details every Redis key, stream, consumer group, data type, expiration (TTL) policy, and owning module utilized across the system.

---

## Keys, Types, TTLs, and Owners

| Key Pattern | Redis Type | Default TTL | Owner Module | Description |
| :--- | :--- | :--- | :--- | :--- |
| `research:state:{run_id}` | `string` (JSON) | 24 Hours (`redis_ttl_seconds`) | `app.persistence.redis_store` | Persists full research run metadata, status enum, plan, and progress metrics. |
| `research:report:{run_id}` | `string` (JSON) | 24 Hours (`redis_ttl_seconds`) | `app.persistence.redis_store` | Persists the final synthesized and citation-grounded research report. |
| `research:trace:{run_id}` | `list` | 24 Hours (`redis_ttl_seconds`) | `app.persistence.redis_store` | Ordered list of `StepTrace` items capturing agent executions, token spend, and latency. |
| `research:running_runs` | `set` | Persistent | `app.persistence.redis_store` | Set of currently active/in-progress `run_id`s; managed dynamically via `sadd`/`srem`. |
| `research:lock:{run_id}` | `string` (Token) | Ephemeral (`timeout_seconds`, default 30–120s) | `app.persistence.redis_store` | Distributed mutual exclusion lock storing a random owner token. Released strictly via Lua compare-and-delete. |
| `research:ratelimit:{api_key}:{window}` | `string` (Integer) | 60 Seconds | `app.api.auth` | Sliding/fixed-window request counters enforcing per-key rate limits (e.g., 10 req/min). |
| `research:idempotency:{key}` | `string` (JSON) | 24 Hours | `app.persistence.idempotency` | Node-level computation memoization caching inputs to avoid redundant executions. |
| `research:idempotency_request:{api_key}:{request_hash}` | `string` (JSON) | 24 Hours | `app.api.routes` | Request-level `Idempotency-Key` tracking `{run_id, request_hash}` to ensure atomic replay or conflict detection on `POST /research`. |
| `research:checkpoint:{prefix}:{thread_id}:{ns}:{checkpoint_id}` | `string` (Envelope) | 24 Hours | `app.persistence.checkpointer` | LangGraph node checkpoints serialized via `JsonPlusSerializer` with HMAC tamper signatures. |
| `research:checkpoint:latest:{thread_id}:{ns}` | `string` | 24 Hours | `app.persistence.checkpointer` | Pointer storing the most recent checkpoint ID for a thread and namespace. |
| `research:checkpoint:writes:{thread_id}:{ns}:{checkpoint_id}:{task_id}` | `string` (Envelope) | 24 Hours | `app.persistence.checkpointer` | State write envelopes recorded by asynchronous branch executions. |
| `research:budget:{run_id}:tokens` | `string` (Integer) | 24 Hours | `app.budget.guard` | Cross-process atomic token spend counter updated via Lua scripts. |
| `research:budget:{run_id}:usd_micros` | `string` (Integer) | 24 Hours | `app.budget.guard` | Cross-process atomic USD cost counter stored as integer micro-dollars ($1 = 1,000,000 µUSD). |
| `research:budget:{run_id}:searches` | `string` (Integer) | 24 Hours | `app.budget.guard` | Cross-process atomic search tool invocation counter. |
| `research:budget:{run_id}:start_time` | `string` (Float) | 24 Hours | `app.budget.guard` | Persisted initial run wall-clock epoch timestamp preventing clock resets on run resumption. |
| `research:cancel:{run_id}` | `string` ("1") | 24 Hours | `app.api.routes`, `app.graph.nodes` | Cancellation flag checked by workers and nodes between steps to halt execution. |
| `research:worker:active_fake_worker` | `string` (Consumer) | 15 Seconds (heartbeat) | `app.worker` | Enforces that only ONE worker runs concurrently in fake mode (single-worker check). |
| `research:worker:semaphore` | `string` (Integer) | Persistent | `app.persistence.concurrency`, `app.worker` | Cross-process atomic semaphore tracking active runs against `MAX_CONCURRENT_RUNS`. |
| `research:metrics:worker_reclaims_total` | `string` (Integer) | Persistent | `app.worker`, `app.api.metrics` | Counter tracking total idle jobs reclaimed across workers. |
| `research:metrics:dlq_total` | `string` (Integer) | Persistent | `app.worker`, `app.api.metrics` | Counter tracking total poison jobs moved to DLQ after 3 delivery attempts. |
| `research:concurrency:active` | `zset` | Lease TTL | `app.persistence.concurrency` | Distributed semaphore / lease registry bounding cross-process worker concurrency. |

---

## Redis Streams and Consumer Groups

| Stream Key | Consumer Group | Payload Fields | Owner Module | Description |
| :--- | :--- | :--- | :--- | :--- |
| `research:jobs` | `research-workers` | `run_id`, `question`, `options` | `app.worker`, `app.api.routes` | Work queue for background research workflows dispatched by `POST /research`. Workers consume via `XREADGROUP` and `XAUTOCLAIM`. |
| `research:jobs:dlq` | N/A | `run_id`, `original_payload`, `error`, `attempts` | `app.worker` | Dead Letter Queue (DLQ) receiving jobs that have exceeded 3 delivery attempts without successful completion. |

---

## Concurrency and Durability Guarantees

1. **Owner-Checked Locks**:
   Distributed locks (`research:lock:{run_id}`) store a unique random UUID token upon acquisition (`SET NX PX`). Lock release is strictly executed via an atomic Lua script:
   ```lua
   if redis.call('GET', KEYS[1]) == ARGV[1] then
       return redis.call('DEL', KEYS[1])
   else
       return 0
   end
   ```
   This prevents a slow or stalled worker from accidentally unlocking a key acquired by another worker after a timeout.

2. **Atomic Budget Counters**:
   Budget reservations and spend records use integer micro-dollars (`usd_micros`) to avoid floating-point inaccuracies in Redis atomic increments.

3. **Fakeredis Local Support**:
   In local development and testing environments (`use_fake_redis=True`), state is mirrored to `.fake_redis.json` using debounced disk sync. Ephemeral keys (such as `research:lock:*`) are excluded from cross-session restoration to avoid eternal locks.
