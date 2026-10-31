# Performance Benchmarks

## Parallel Fan-Out Benchmark (Phase C1)

### Benchmark Specification
- **Test File**: `tests/benchmarks/bench_parallel.py`
- **Workload**: 4 independent SubQuestions
- **Simulated Per-Call Latency**: 0.500s (`FakeProvider.async_delay_s`)
- **Note**: Speedup was measured with FakeProvider at 0.5s simulated latency per call, not real LLM latency.
- **Target Speedup**: $\ge 2.0\times$

### Benchmark Results

| Metric | Sequential (`MAX_PARALLEL_RESEARCHERS=1`) | Parallel (`MAX_PARALLEL_RESEARCHERS=4`) | Target | Achieved Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **Duration (seconds)** | **2.281s** | **0.545s** | $\ge 2.0\times$ | **4.19x** |

### Analysis
With `MAX_PARALLEL_RESEARCHERS=1`, the 4 sub-questions execute sequentially through the fan-out wave, yielding an overall graph execution duration of ~2.28s (4 * 0.5s + planning, review, write and graph overhead).

With `MAX_PARALLEL_RESEARCHERS=4`, the 4 sub-questions are dispatched concurrently using LangGraph `Send("research_one", ...)` guarded by `asyncio.Semaphore(4)`. The tasks execute overlapping in time, reducing execution duration to 0.545s, achieving a **4.19x** empirical speedup (measured under simulated 0.5s per-call latency with FakeProvider, not real LLM API latency), substantially exceeding the 2.0x target.
