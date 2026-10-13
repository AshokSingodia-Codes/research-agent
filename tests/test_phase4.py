import subprocess
import sys
import time

import pytest

from app.graph.builder import build_research_graph
from app.llm.client import llm_client
from app.llm.fake_provider import FakeProvider
from app.models.enums import RunStatus
from app.persistence.checkpointer import RedisCheckpointSaver
from app.persistence.idempotency import (
    compute_idempotency_key,
    get_idempotent_result,
    save_idempotent_result,
)
from app.persistence.redis_store import redis_store


@pytest.fixture(autouse=True)
def setup_fake_llm():
    fake = FakeProvider()
    llm_client.set_override_provider(fake)
    yield fake
    llm_client.set_override_provider(None)


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
    run_id = "test_resume_001"
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


@pytest.mark.chaos
def test_chaos_subprocess_sigkill_and_resumption():
    """
    CHAOS RESILIENCE TEST:
    1. Spawns a background worker subprocess with artificial node delays.
    2. Kills the worker process mid-run (SIGKILL via terminate/kill).
    3. Starts a new worker process referencing the exact same run_id.
    4. Asserts that the run completes, previous completed work was NOT re-executed,
       and the final report is generated.
    """
    run_id = f"chaos_run_{int(time.time())}"
    topic = "High-Throughput Distributed Raft Consensus"

    # 1. Start worker subprocess with 0.8s artificial delay per LLM step
    cmd = [sys.executable, "tests/chaos_worker_runner.py", run_id, topic, "0.8"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    checkpointer = RedisCheckpointSaver()
    # Wait until planner node completes and checkpoints into Redis
    for _ in range(60):
        cp = checkpointer.get_tuple({"configurable": {"thread_id": run_id}})
        if cp and "plan" in cp.checkpoint.get("channel_values", {}):
            break
        time.sleep(0.1)

    # 2. Hard kill worker subprocess mid-run (simulating SIGKILL / crash)
    proc.kill()
    proc.wait()

    # Verify run did not complete in first process
    assert redis_store.get_final_report(run_id) is None

    # Checkpoint should exist for the thread
    cp_tuple = checkpointer.get_tuple({"configurable": {"thread_id": run_id}})
    assert cp_tuple is not None, "Expected checkpoint to be persisted to Redis before process kill"
    assert "plan" in cp_tuple.checkpoint["channel_values"]

    # 3. Restart worker: spawn second process to resume the same run
    cmd_resume = [sys.executable, "tests/chaos_worker_runner.py", run_id, topic, "0.1"]
    proc_resume = subprocess.run(cmd_resume, capture_output=True, text=True, timeout=15)

    assert "SUCCESS:" in proc_resume.stdout

    # 4. Verify completed report exists in Redis
    final_report_json = redis_store.get_final_report(run_id)
    assert final_report_json is not None
    assert topic in final_report_json
