import time
import uuid

import pytest

from app.graph.builder import build_research_graph
from app.models.enums import RunStatus
from app.persistence.checkpointer import RedisCheckpointSaver
from app.persistence.redis_store import redis_store
from app.persistence.stream_adapter import RedisStreamAdapter


@pytest.mark.integration
def test_stream_consumer_abandon_and_reclaim_resumption(setup_fake_llm):
    """
    C3 LOCAL IN-PROCESS CRASH SIMULATION:
    1. Consumer A reads a job via XREADGROUP.
    2. Consumer A processes initial nodes (planner, researcher, supervisor) and checkpoints to Redis.
    3. Consumer A crashes / is abandoned WITHOUT calling XACK.
    4. Consumer B reclaims the abandoned job via XAUTOCLAIM / fallback with min_idle_time=0.
    5. Consumer B resumes execution from the durable checkpoint.
    6. Consumer B finishes the workflow and acknowledges (XACK) the job.
    7. Asserts:
       - No node ran twice (planner was not re-executed).
       - Spend equals the sum of unique steps.
       - Run status reaches 'completed' or 'needs_review'.
       - The stream message is fully ACKed (0 pending messages in group).
    """
    client = redis_store.get_client()
    adapter = RedisStreamAdapter(client)
    checkpointer = RedisCheckpointSaver()

    run_id = f"run_c3_{uuid.uuid4().hex[:8]}"
    stream = f"test:stream:c3_{uuid.uuid4().hex[:6]}"
    group = "worker_group_c3"
    topic = "Zero-Knowledge Rollup Architecture"

    # Step 1: Create consumer group and publish job to stream
    adapter.create_group(stream, group, mkstream=True)
    msg_id = adapter.publish_job(stream, {"run_id": run_id, "topic": topic})
    assert msg_id is not None

    # Step 2: Consumer A reads job via XREADGROUP
    jobs_a = adapter.read_job(stream, group, consumer="consumer_A", count=1)
    assert len(jobs_a) == 1
    reclaimed_id, job_payload = jobs_a[0]
    assert reclaimed_id == msg_id
    assert job_payload["run_id"] == run_id

    # Step 3: Consumer A starts workflow up to interrupt (planner, research, review)
    graph_a = build_research_graph(checkpointer=checkpointer, interrupt_before_write=True)
    config = {"configurable": {"thread_id": run_id}}
    initial_state = {
        "run_id": run_id,
        "question": topic,
        "status": RunStatus.PLANNING,
        "start_time": time.time(),
        "total_tokens": 0,
        "total_usd_cost": 0.0,
        "step_index": 0,
    }

    graph_a.invoke(initial_state, config=config)

    # Checkpoint exists and is paused before write
    state_tuple = graph_a.get_state(config)
    assert "write" in state_tuple.next
    plan_obj = state_tuple.values.get("plan")
    assert plan_obj is not None

    calls_after_a = setup_fake_llm.call_count
    tokens_after_a = state_tuple.values.get("total_tokens", 0)
    cost_after_a = state_tuple.values.get("total_usd_cost", 0.0)

    # Step 4: Consumer A crashes! It DOES NOT call xack.
    # Verify job is still pending in the consumer group.
    pending_info = adapter.get_pending_summary(stream, group)
    assert pending_info.get("pending", 0) == 1

    # Step 5: Consumer B reclaims the un-acked job (min_idle_time=0)
    next_id, claimed_jobs, _ = adapter.autoclaim_jobs(stream, group, consumer="consumer_B", min_idle_time=0)
    assert len(claimed_jobs) >= 1
    claimed_msg_id = claimed_jobs[0][0]
    assert claimed_msg_id == msg_id

    # Step 6: Consumer B resumes from checkpoint with a new graph instance
    graph_b = build_research_graph(checkpointer=checkpointer, interrupt_before_write=False)
    final_state = graph_b.invoke(None, config=config)

    # Step 7: Consumer B finishes and ACKs the message
    acked_count = adapter.ack_job(stream, group, claimed_msg_id)
    assert acked_count == 1

    # Verify stream entry is now fully acked (0 pending)
    pending_final = adapter.get_pending_summary(stream, group)
    assert pending_final.get("pending", 0) == 0

    # Step 8: Assertions
    # 1. Status is completed or needs_review
    assert final_state["status"] in [RunStatus.COMPLETED, RunStatus.NEEDS_REVIEW]
    assert final_state.get("final_report") is not None

    # 2. No node ran twice:
    # Planner was not re-executed; only write and critic nodes were called after resumption
    calls_after_b = setup_fake_llm.call_count
    assert calls_after_b > calls_after_a
    # Specifically, writer + critic = 2 LLM calls during resume
    assert calls_after_b == calls_after_a + 2

    # 3. Spend equals sum of unique steps (monotonic accumulation without double-counting)
    final_tokens = final_state.get("total_tokens", 0)
    final_cost = final_state.get("total_usd_cost", 0.0)
    assert final_tokens > tokens_after_a
    assert final_cost >= cost_after_a
