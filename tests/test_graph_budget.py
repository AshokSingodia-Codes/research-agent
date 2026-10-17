import time

import pytest
from langgraph.checkpoint.memory import MemorySaver

from src.agents.supervisor import review_research_output
from src.config import settings
from src.graph.builder import build_research_graph
from src.models.schemas import (
    EvidenceRecord,
    ResearcherOutput,
    SearchStatus,
    SubQuestion,
)


@pytest.mark.budget
@pytest.mark.integration
def test_supervisor_allows_work_back_exactly_once():
    """Validates that the supervisor permits a retry exactly once and refuses subsequent retries."""
    sub_q = SubQuestion(
        id="sq_retry",
        question="What are the specific benchmarks?",
        rationale="Needs quantitative data.",
        expected_evidence="Numbers and graphs."
    )
    deficient_output = ResearcherOutput(
        sub_question_id="sq_retry",
        findings=[],  # Empty findings = deficient
        search_status=SearchStatus(status="no_results", result_count=0)
    )

    # First inspection: attempt 0 -> should allow retry exactly once
    review1, trace1 = review_research_output(
        sub_question=sub_q,
        research_output=deficient_output,
        current_retry_count=0,
        total_tokens=100,
        elapsed_seconds=10.0,
        is_last_subquestion=False
    )
    assert review1.allow_retry is True
    assert review1.next_action == "retry_subquestion"

    # Second inspection: attempt 1 -> max_revision_loops reached, retry MUST be forbidden
    review2, trace2 = review_research_output(
        sub_question=sub_q,
        research_output=deficient_output,
        current_retry_count=1,
        total_tokens=200,
        elapsed_seconds=20.0,
        is_last_subquestion=False
    )
    assert review2.allow_retry is False
    assert review2.next_action == "continue_research"  # moves on instead of infinite loop


@pytest.mark.budget
@pytest.mark.integration
def test_supervisor_enforces_wall_clock_ceiling():
    """Validates that supervisor forces termination when wall-clock ceiling is exceeded."""
    sub_q = SubQuestion(
        id="sq_timeout",
        question="Long running query",
        rationale="Timeout test",
        expected_evidence="Data"
    )
    valid_output = ResearcherOutput(
        sub_question_id="sq_timeout",
        findings=[
            EvidenceRecord(
                claim="Claim", source_url="https://example.com", domain="example.com", snippet="Snip"
            )
        ],
        search_status=SearchStatus(status="success", result_count=1)
    )

    review, trace = review_research_output(
        sub_question=sub_q,
        research_output=valid_output,
        current_retry_count=0,
        total_tokens=500,
        elapsed_seconds=settings.wall_clock_timeout_seconds + 5.0,  # exceeds ceiling
        is_last_subquestion=False
    )

    assert review.next_action == "synthesize_report"
    assert "Wall-clock budget exhausted" in review.feedback


@pytest.mark.budget
@pytest.mark.integration
def test_supervisor_enforces_token_budget_cap():
    """Validates that supervisor halts iteration when token budget is exhausted."""
    sub_q = SubQuestion(
        id="sq_token",
        question="Query",
        rationale="Token test",
        expected_evidence="Data"
    )
    valid_output = ResearcherOutput(
        sub_question_id="sq_token",
        findings=[
            EvidenceRecord(
                claim="Claim", source_url="https://example.com", domain="example.com", snippet="Snip"
            )
        ],
        search_status=SearchStatus(status="success", result_count=1)
    )

    review, trace = review_research_output(
        sub_question=sub_q,
        research_output=valid_output,
        current_retry_count=0,
        total_tokens=settings.max_total_tokens + 1000,  # exceeds token limit
        elapsed_seconds=10.0,
        is_last_subquestion=False
    )

    assert review.next_action == "synthesize_report"
    assert "Token budget exhausted" in review.feedback


@pytest.mark.integration
def test_end_to_end_graph_execution():
    """Validates full LangGraph workflow execution with state persistence and step tracing."""
    checkpointer = MemorySaver()
    graph = build_research_graph(checkpointer=checkpointer)

    run_id = "test_run_e2e_001"
    config = {"configurable": {"thread_id": run_id}}

    initial_state = {
        "run_id": run_id,
        "research_topic": "Vector Indexing Techniques in Distributed Vector Databases",
        "start_time_epoch": time.time(),
        "total_tokens": 0,
        "total_searches": 0,
        "trace_log": [],
    }

    final_state = graph.invoke(initial_state, config=config)

    assert final_state["status"] == "completed"
    assert final_state["plan"] is not None
    assert len(final_state["plan"].sub_questions) <= settings.max_subquestions
    assert len(final_state["findings"]) > 0
    assert final_state["report"] is not None
    assert final_state["report"].completion_status == "complete"
    assert len(final_state["report"].citations) > 0

    # Validate step trace log
    node_names = [t.node_name for t in final_state["trace_log"]]
    assert "planner" in node_names
    assert "researcher" in node_names
    assert "supervisor" in node_names
    assert "writer" in node_names
