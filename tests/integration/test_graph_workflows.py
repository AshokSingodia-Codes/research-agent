import time

import pytest

from app.agents.supervisor import review_findings
from app.budget.guard import BudgetGuard
from app.config import settings
from app.graph.builder import build_research_graph
from app.graph.parallel_runner import get_ready_subquestions, run_parallel_fanout
from app.llm.client import llm_client
from app.llm.fake_provider import FakeProvider
from app.models.enums import ReviewAction, RunStatus
from app.models.schemas import (
    Finding,
    ReviewDecision,
    SubQuestion,
)


@pytest.fixture(autouse=True)
def setup_fake_llm():
    fake = FakeProvider()
    llm_client.set_override_provider(fake)
    yield fake
    llm_client.set_override_provider(None)


def test_supervisor_allows_work_back_exactly_once(setup_fake_llm):
    """Validates that the supervisor permits a retry exactly once and refuses subsequent retries."""
    sub_q = SubQuestion(
        id="sq_retry",
        text="What are the specific benchmarks?",
        rationale="Needs quantitative data.",
        priority=1,
    )
    empty_findings: list[Finding] = []

    # First inspection: attempt 0 -> should allow retry exactly once
    setup_fake_llm.queue_response(
        ReviewDecision(
            sub_question_id="sq_retry",
            action=ReviewAction.RETRY_ONCE,
            reason="Findings insufficient.",
            follow_up_query="benchmarks quantitative",
        )
    )
    decision1, trace1 = review_findings(
        sub_question=sub_q,
        findings=empty_findings,
        current_retry_count=0,
    )
    assert decision1.action == ReviewAction.RETRY_ONCE

    # Second inspection: attempt 1 -> max retry loops reached, retry MUST be forbidden
    setup_fake_llm.queue_response(
        ReviewDecision(
            sub_question_id="sq_retry",
            action=ReviewAction.RETRY_ONCE,
            reason="Still insufficient.",
        )
    )
    decision2, trace2 = review_findings(
        sub_question=sub_q,
        findings=empty_findings,
        current_retry_count=1,
    )
    assert decision2.action == ReviewAction.SKIP
    assert "already used its single allowed retry" in decision2.reason


def test_supervisor_enforces_wall_clock_ceiling(setup_fake_llm):
    """Validates that supervisor forces termination when wall-clock ceiling is exceeded."""
    sub_q = SubQuestion(
        id="sq_timeout",
        text="Long running query",
        rationale="Timeout test",
        priority=1,
    )
    findings = [
        Finding(
            id="fnd_t",
            sub_question_id="sq_timeout",
            claim="Claim",
            source_url="https://example.com",
            domain="example.com",
            snippet="Snip",
            retrieval_timestamp=time.time(),
            content_hash="h1",
            confidence=0.9,
        )
    ]
    guard = BudgetGuard(wall_clock_timeout_seconds=0.01)
    guard.start_time = time.time() - 10.0  # 10s elapsed > 0.01s limit

    decision, trace = review_findings(
        sub_question=sub_q,
        findings=findings,
        current_retry_count=0,
        budget_guard=guard,
    )
    assert decision.action == ReviewAction.ABORT
    assert "Terminating research due to budget exhaustion" in decision.reason


def test_supervisor_enforces_token_budget_cap(setup_fake_llm):
    """Validates that supervisor halts iteration when token budget is exhausted."""
    sub_q = SubQuestion(
        id="sq_token",
        text="Query",
        rationale="Token test",
        priority=1,
    )
    findings = [
        Finding(
            id="fnd_tok",
            sub_question_id="sq_token",
            claim="Claim",
            source_url="https://example.com",
            domain="example.com",
            snippet="Snip",
            retrieval_timestamp=time.time(),
            content_hash="h2",
            confidence=0.9,
        )
    ]
    guard = BudgetGuard(max_total_tokens=100)
    guard.record_spend(tokens=200, usd_cost=0.01)

    decision, trace = review_findings(
        sub_question=sub_q,
        findings=findings,
        current_retry_count=0,
        budget_guard=guard,
    )
    assert decision.action == ReviewAction.ABORT
    assert "Terminating research due to budget exhaustion" in decision.reason


def test_end_to_end_graph_execution(setup_fake_llm):
    """Validates full LangGraph workflow execution with state persistence and step tracing."""
    graph = build_research_graph()
    run_id = "test_run_e2e_001"

    initial_state = {
        "run_id": run_id,
        "question": "Vector Indexing Techniques in Distributed Vector Databases",
        "start_time": time.time(),
        "total_tokens": 0,
        "total_usd_cost": 0.0,
        "step_index": 0,
    }

    final_state = graph.invoke(initial_state)

    assert final_state["status"] in (RunStatus.COMPLETED, RunStatus.NEEDS_REVIEW)
    assert final_state["plan"] is not None
    assert len(final_state["plan"].sub_questions) <= settings.max_subquestions
    assert len(final_state["findings"]) > 0
    assert final_state["final_report"] is not None
    assert len(final_state["final_report"].citations) > 0


@pytest.mark.asyncio
async def test_fanout_and_dependency_order():
    """Validates parallel fanout execution and topological sub-question resolution."""
    sub_qs = [
        SubQuestion(id="sq_root", text="Root sub-question", rationale="R1", depends_on=[]),
        SubQuestion(id="sq_child", text="Child sub-question", rationale="R2", depends_on=["sq_root"]),
    ]

    ready = get_ready_subquestions(sub_qs, completed_ids=set(), in_flight_ids=set())
    assert [sq.id for sq in ready] == ["sq_root"]

    guard = BudgetGuard()
    findings, reviews, traces = await run_parallel_fanout([sub_qs[0]], guard, "run_fanout", concurrency_limit=2)
    assert len(findings) > 0
    assert len(reviews) == 1
    assert len(traces) > 0
