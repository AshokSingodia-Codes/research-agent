import time

import pytest

from app.agents.supervisor import review_findings
from app.budget.guard import BudgetGuard
from app.config import settings
from app.graph.builder import build_research_graph
from app.graph.nodes import run_parallel_fanout
from app.graph.routing import get_ready_subquestions
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


@pytest.mark.asyncio
async def test_parallel_fanout_dependency_waves(setup_fake_llm):
    """Validates that sub-questions with dependencies wait for earlier waves to complete."""
    from app.models.schemas import ResearchPlan

    plan = ResearchPlan(
        topic="Dependency test",
        sub_questions=[
            SubQuestion(id="sq_dep_1", text="Root question 1", rationale="R1", depends_on=[]),
            SubQuestion(id="sq_dep_2", text="Child question 2", rationale="R2", depends_on=["sq_dep_1"]),
        ],
    )
    setup_fake_llm.queue_response(plan)

    graph = build_research_graph()
    initial_state = {
        "run_id": "test_run_dep_waves",
        "question": "Dependency test topic",
        "start_time": time.time(),
        "total_tokens": 0,
        "total_usd_cost": 0.0,
        "step_index": 0,
    }

    final_state = await graph.ainvoke(initial_state)
    assert final_state["status"] in (RunStatus.COMPLETED, RunStatus.NEEDS_REVIEW)
    assert "sq_dep_1" in final_state["completed_subquestion_ids"]
    assert "sq_dep_2" in final_state["completed_subquestion_ids"]


@pytest.mark.asyncio
async def test_parallel_fanout_time_overlap_and_semaphore_bound(setup_fake_llm):
    """Validates that parallel branches overlap in time and obey the max_parallel_researchers bound."""
    from app.models.schemas import ResearchPlan

    plan = ResearchPlan(
        topic="Concurrency test",
        sub_questions=[
            SubQuestion(id=f"sq_par_{i}", text=f"Parallel question {i}", rationale=f"R{i}", depends_on=[])
            for i in range(4)
        ],
    )
    setup_fake_llm.queue_response(plan)
    # Configure artificial delay on fake provider
    setup_fake_llm.async_delay_s = 0.25

    graph = build_research_graph()
    initial_state = {
        "run_id": "test_run_concurrency",
        "question": "Concurrency test topic",
        "start_time": time.time(),
        "total_tokens": 0,
        "total_usd_cost": 0.0,
        "step_index": 0,
    }

    t0 = time.time()
    final_state = await graph.ainvoke(initial_state)
    elapsed = time.time() - t0

    assert final_state["status"] in (RunStatus.COMPLETED, RunStatus.NEEDS_REVIEW)
    # 4 tasks * 0.25s sequential would take >= 1.0s. With concurrency limit 3, it should finish in < 0.8s
    assert elapsed < 0.85, f"Expected parallel overlap under 0.85s, took {elapsed:.2f}s"


@pytest.mark.asyncio
async def test_failing_branch_yields_typed_failure_while_others_finish(setup_fake_llm, monkeypatch):
    """Validates that an individual failing branch yields a typed failure while other branches finish."""
    import app.graph.nodes as nodes_mod
    from app.models.schemas import ResearchPlan

    plan = ResearchPlan(
        topic="Failure resilience test",
        sub_questions=[
            SubQuestion(id="sq_good_1", text="Good branch 1", rationale="R1", depends_on=[]),
            SubQuestion(id="sq_bad_2", text="Failing branch 2", rationale="R2", depends_on=[]),
        ],
    )
    setup_fake_llm.queue_response(plan)

    orig_research = nodes_mod.research_subquestion

    def mock_research_subquestion(sub_question, **kwargs):
        if sub_question.id == "sq_bad_2":
            raise RuntimeError("Simulated network timeout in branch sq_bad_2")
        return orig_research(sub_question, **kwargs)

    monkeypatch.setattr(nodes_mod, "research_subquestion", mock_research_subquestion)

    graph = build_research_graph()
    initial_state = {
        "run_id": "test_run_failing_branch",
        "question": "Resilience test topic",
        "start_time": time.time(),
        "total_tokens": 0,
        "total_usd_cost": 0.0,
        "step_index": 0,
    }

    final_state = await graph.ainvoke(initial_state)
    assert final_state["status"] in (RunStatus.COMPLETED, RunStatus.NEEDS_REVIEW)
    # Ensure good branch finished with findings
    assert "sq_good_1" in final_state["completed_subquestion_ids"]
    # Check trace log contains the typed error for sq_bad_2
    bad_traces = [t for t in final_state["trace_log"] if t.error and "sq_bad_2" in str(t.input_data)]
    assert len(bad_traces) > 0


@pytest.mark.asyncio
async def test_retry_cap_holds_in_parallel(setup_fake_llm):
    """Validates that retry count is capped at 1 per sub-question across parallel waves."""
    from app.models.schemas import ResearchPlan

    plan = ResearchPlan(
        topic="Retry cap test",
        sub_questions=[
            SubQuestion(id="sq_ret_1", text="Question 1 needing retry", rationale="R1", depends_on=[]),
        ],
    )
    setup_fake_llm.queue_response(plan)

    # Queue retry once twice for sq_ret_1
    setup_fake_llm.queue_response(
        ReviewDecision(
            sub_question_id="sq_ret_1",
            action=ReviewAction.RETRY_ONCE,
            reason="Findings insufficient round 1.",
        )
    )
    setup_fake_llm.queue_response(
        ReviewDecision(
            sub_question_id="sq_ret_1",
            action=ReviewAction.RETRY_ONCE,
            reason="Still insufficient round 2.",
        )
    )

    graph = build_research_graph()
    initial_state = {
        "run_id": "test_run_retry_cap",
        "question": "Retry cap test topic",
        "start_time": time.time(),
        "total_tokens": 0,
        "total_usd_cost": 0.0,
        "step_index": 0,
    }

    final_state = await graph.ainvoke(initial_state)
    assert final_state["status"] in (RunStatus.COMPLETED, RunStatus.NEEDS_REVIEW)
    assert final_state["retry_counts"].get("sq_ret_1", 0) <= 1
    assert "sq_ret_1" in final_state["completed_subquestion_ids"]

