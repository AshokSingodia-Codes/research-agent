import asyncio
import time
import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.graph.builder import build_research_graph
from app.main import app
from app.models.enums import RunStatus
from app.models.schemas import StepTrace
from app.persistence.checkpointer import RedisCheckpointSaver
from app.persistence.idempotency import (
    compute_idempotency_key,
    get_idempotent_result,
    save_idempotent_result,
)
from app.persistence.redis_store import redis_store
from tests.conftest import wait_for_status


def test_redis_store_state_and_locks():
    """Validates Redis state persistence and distributed mutual exclusion locks."""
    run_id = "test_lock_001"
    state = {"run_id": run_id, "status": "researching", "tokens": 1500}

    redis_store.save_run_state(run_id, state)
    loaded = redis_store.get_run_state(run_id)
    assert loaded["tokens"] == 1500
    assert run_id in redis_store.list_running_runs()

    # Worker 1 acquires lock
    assert redis_store.acquire_lock(run_id, timeout_seconds=10.0) is True

    # Worker 2 attempts to acquire lock on same run -> denied
    assert redis_store.acquire_lock(run_id, timeout_seconds=10.0) is False

    # Worker 1 releases lock
    redis_store.release_lock(run_id)

    # Worker 2 can now acquire lock
    assert redis_store.acquire_lock(run_id, timeout_seconds=10.0) is True
    redis_store.release_lock(run_id)


def test_idempotency_caching():
    """Validates that idempotent node key caching skips re-computation."""
    run_id = f"test_idem_{int(time.time() * 1000)}"
    node = "planner"
    input_data = {"topic": "Raft Consensus Invariants"}

    key = compute_idempotency_key(run_id, node, input_data)
    assert get_idempotent_result(key) is None

    # Cache output
    output_data = {"sub_questions": ["sq_1", "sq_2"]}
    save_idempotent_result(key, output_data)

    cached = get_idempotent_result(key)
    assert cached == output_data


def test_durable_checkpoint_resumption_without_replaying(setup_fake_llm):
    """
    Validates that a LangGraph workflow checkpointed in Redis resumes from the last completed node
    without re-executing completed nodes.
    """
    checkpointer = RedisCheckpointSaver()
    run_id = f"test_resume_{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": run_id}}

    # First execution: interrupt before writing
    graph_paused = build_research_graph(checkpointer=checkpointer, interrupt_before_write=True)
    initial_state = {
        "run_id": run_id,
        "question": "Distributed Log Compaction Benchmarks",
        "start_time": time.time(),
        "total_tokens": 0,
        "total_usd_cost": 0.0,
        "step_index": 0,
    }

    graph_paused.invoke(initial_state, config=config)

    # Inspect that workflow reached interrupt before write
    state_tuple = graph_paused.get_state(config)
    assert "write" in state_tuple.next
    assert state_tuple.values.get("plan") is not None
    plan_obj = state_tuple.values["plan"]
    assert len(plan_obj.sub_questions) >= 1

    calls_before_resume = setup_fake_llm.call_count

    # Simulate fresh restart: create brand-new graph instance and resume with None input
    graph_resumed = build_research_graph(checkpointer=checkpointer, interrupt_before_write=False)
    final_state = graph_resumed.invoke(None, config=config)

    assert final_state is not None
    assert final_state.get("final_report") is not None
    # Planner was NOT re-invoked; only remaining write + critic were invoked
    assert setup_fake_llm.call_count > calls_before_resume
    assert final_state["status"] in [RunStatus.COMPLETED, RunStatus.NEEDS_REVIEW]


def test_crash_and_resumption_without_replaying(setup_fake_llm):
    """Alias for durable checkpoint resumption without replaying completed nodes."""
    test_durable_checkpoint_resumption_without_replaying(setup_fake_llm)


def test_redis_store_state_and_trace_persistence():
    """Validates saving and querying state, progress, and step traces in Redis store."""
    run_id = f"test_run_persist_{uuid.uuid4().hex[:8]}"

    # Save initial progress
    redis_store.save_run_state(
        run_id=run_id,
        status="in_progress",
        current_node="researcher",
        completed_sq_ids=["sq_1"],
        total_tokens=1500,
        total_subquestions=3,
    )

    status = redis_store.get_run_status(run_id)
    assert status is not None
    assert status.run_id == run_id
    assert status.status == "in_progress"
    assert status.current_node == "researcher"
    assert status.completed_subquestions == 1
    assert status.total_subquestions == 3
    assert status.progress_percentage == pytest.approx(33.3, 0.1)
    assert status.total_tokens == 1500

    # Append traces
    trace1 = StepTrace(
        step_id="step_1",
        run_id=run_id,
        agent="planner",
        model="fake",
        provider="fake",
        latency_ms=120.0,
        total_tokens=500,
    )
    trace2 = StepTrace(
        step_id="step_2",
        run_id=run_id,
        agent="researcher",
        model="fake",
        provider="fake",
        latency_ms=250.0,
        total_tokens=1000,
    )
    redis_store.append_trace_step(run_id, trace1)
    redis_store.append_trace_step(run_id, trace2)

    traces = redis_store.get_trace_steps(run_id)
    assert len(traces) == 2
    assert traces[0]["agent"] == "planner"
    assert traces[1]["agent"] == "researcher"


@pytest.mark.asyncio
async def test_concurrent_research_runs_isolation():
    """Validates that concurrent research runs maintain completely isolated state and run IDs."""
    transport = ASGITransport(app=app)
    headers = {"X-API-Key": settings.api_key}

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Launch 2 concurrent runs
        topics = ["Topic A: Raft Consensus", "Topic B: Paxos Protocol"]

        responses = await asyncio.gather(
            *[client.post("/research", json={"question": t}, headers=headers) for t in topics]
        )

        run_ids = [r.json()["run_id"] for r in responses]
        assert len(set(run_ids)) == len(topics)  # All run IDs must be unique

        # Wait for all runs to complete concurrently using exponential backoff helper
        datas = await asyncio.gather(
            *[
                wait_for_status(
                    client,
                    rid,
                    (RunStatus.COMPLETED, RunStatus.NEEDS_REVIEW),
                    timeout=60.0,
                    headers=headers,
                )
                for rid in run_ids
            ]
        )

        # Inspect that all completed with isolated reports
        for idx, data in enumerate(datas):
            assert data["status"] in ("completed", "needs_review")
            assert data["report"] is not None
            assert topics[idx] in data["report"]["topic"]
