import time

import pytest
from langgraph.checkpoint.memory import MemorySaver

from src.graph.builder import build_research_graph
from src.graph.redis_store import redis_store
from src.models.schemas import StepTrace


@pytest.mark.durability
@pytest.mark.integration
def test_redis_store_state_and_trace_persistence():
    """Validates saving and querying state, progress, and step traces in Redis store."""
    run_id = "test_run_persist_001"

    # Save initial progress
    redis_store.save_run_state(
        run_id=run_id,
        status="in_progress",
        current_node="researcher",
        completed_sq_ids=["sq_1"],
        total_tokens=1500,
        total_subquestions=3
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
        node_name="planner",
        latency_ms=120.0,
        tokens_used=500,
        input_summary="Topic",
        output_summary="Generated 3 questions"
    )
    trace2 = StepTrace(
        step_id="step_2",
        node_name="researcher",
        latency_ms=250.0,
        tokens_used=1000,
        input_summary="sq_1",
        output_summary="2 findings"
    )
    redis_store.append_trace_step(run_id, trace1)
    redis_store.append_trace_step(run_id, trace2)

    traces = redis_store.get_trace_log(run_id)
    assert len(traces) == 2
    assert traces[0].node_name == "planner"
    assert traces[1].node_name == "researcher"
    assert traces[1].latency_ms == 250.0


@pytest.mark.durability
@pytest.mark.integration
def test_crash_and_resumption_without_replaying():
    """
    Validates that if a run stops mid-stream, resuming with the same thread_id
    picks up where it stopped rather than replaying completed nodes.
    """
    checkpointer = MemorySaver()
    graph = build_research_graph(checkpointer=checkpointer)

    run_id = "test_crash_resume_run"
    config = {"configurable": {"thread_id": run_id}}

    initial_state = {
        "run_id": run_id,
        "research_topic": "Fault Tolerant State Machine Replication",
        "start_time_epoch": time.time(),
        "total_tokens": 0,
        "total_searches": 0,
        "trace_log": [],
    }

    # Step 1: Run graph with a limit of 2 steps (e.g. planner and first supervisor pass)
    step_count = 0
    state_snapshot = None
    for event in graph.stream(initial_state, config=config, stream_mode="values"):
        step_count += 1
        state_snapshot = event
        if step_count >= 2:
            break  # Simulate process interruption / crash

    assert state_snapshot is not None
    # Verify planner was executed before the crash
    assert state_snapshot.get("plan") is not None
    original_plan_subquestions = len(state_snapshot["plan"].sub_questions)

    # Step 2: "Process restarted / redeployed" - resume using the same thread_id and checkpointer
    # Resumption takes NO new initial_state, it continues directly from saved checkpoint
    resumed_final = graph.invoke(None, config=config)

    assert resumed_final["status"] == "completed"
    assert resumed_final["report"] is not None
    # Plan is identical and was not regenerated
    assert len(resumed_final["plan"].sub_questions) == original_plan_subquestions
    # Report contains verified citations
    assert len(resumed_final["report"].citations) > 0

    # Redis store status must reflect completion
    final_status = redis_store.get_run_status(run_id)
    assert final_status is not None
    assert final_status.status == "completed"
    assert final_status.progress_percentage == 100.0
