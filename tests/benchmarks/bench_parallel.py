import asyncio
import time

import pytest

from app.config import settings
from app.graph.builder import build_research_graph
from app.llm.client import llm_client
from app.llm.fake_provider import FakeProvider
from app.models.enums import RunStatus
from app.models.schemas import ResearchPlan, SubQuestion


async def run_benchmark():
    fake = FakeProvider()
    llm_client.set_override_provider(fake)
    fake.async_delay_s = 0.5

    sub_questions = [
        SubQuestion(id=f"sq_bench_{i}", text=f"Benchmark SubQuestion {i}", rationale=f"R{i}", depends_on=[])
        for i in range(4)
    ]

    # Run 1: MAX_PARALLEL_RESEARCHERS = 1
    settings.max_parallel_researchers = 1
    fake.queue_response(ResearchPlan(topic="Parallel Benchmark Test", sub_questions=sub_questions))

    graph1 = build_research_graph()
    initial_state_1 = {
        "run_id": "bench_seq_run",
        "question": "Parallel Benchmark Test",
        "start_time": time.time(),
        "total_tokens": 0,
        "total_usd_cost": 0.0,
        "step_index": 0,
    }

    t0 = time.time()
    res1 = await graph1.ainvoke(initial_state_1)
    duration_seq = time.time() - t0
    assert res1["status"] in (RunStatus.COMPLETED, RunStatus.NEEDS_REVIEW)

    # Run 2: MAX_PARALLEL_RESEARCHERS = 4
    settings.max_parallel_researchers = 4
    fake.queue_response(ResearchPlan(topic="Parallel Benchmark Test", sub_questions=sub_questions))

    graph2 = build_research_graph()
    initial_state_2 = {
        "run_id": "bench_par_run",
        "question": "Parallel Benchmark Test",
        "start_time": time.time(),
        "total_tokens": 0,
        "total_usd_cost": 0.0,
        "step_index": 0,
    }

    t0 = time.time()
    res2 = await graph2.ainvoke(initial_state_2)
    duration_par = time.time() - t0
    assert res2["status"] in (RunStatus.COMPLETED, RunStatus.NEEDS_REVIEW)

    speedup = duration_seq / max(duration_par, 0.001)

    print("\n--- Parallel Execution Benchmark ---")
    print("4 Sub-questions, 0.5s simulated latency")
    print(f"Sequential (max_parallel=1): {duration_seq:.3f}s")
    print(f"Parallel   (max_parallel=4): {duration_par:.3f}s")
    print(f"Speedup:                    {speedup:.2f}x (target >= 2.0x)")

    # Restore settings
    settings.max_parallel_researchers = 3
    llm_client.set_override_provider(None)

    return duration_seq, duration_par, speedup


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_benchmark_parallel_fanout():
    duration_seq, duration_par, speedup = await run_benchmark()
    assert speedup >= 2.0, f"Speedup {speedup:.2f}x did not meet target >= 2.0x"


if __name__ == "__main__":
    asyncio.run(run_benchmark())
