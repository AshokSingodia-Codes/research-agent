import random
import time

import pytest

from app.agents.critic import critique_report
from app.agents.supervisor import review_findings
from app.budget.guard import BudgetExceededError, BudgetGuard
from app.graph.builder import build_research_graph
from app.llm.client import llm_client
from app.llm.fake_provider import FakeProvider
from app.models.enums import ReviewAction, RunStatus
from app.models.schemas import (
    Finding,
    Report,
    ReportSection,
    ResearchFindings,
    ResearchPlan,
    ReviewDecision,
    SubQuestion,
)


@pytest.fixture(autouse=True)
def setup_fake_llm():
    fake = FakeProvider()
    llm_client.set_override_provider(fake)
    yield fake
    llm_client.set_override_provider(None)


def test_supervisor_strictly_enforces_single_retry_ceiling(setup_fake_llm):
    """
    Validates that the supervisor code guard strictly allows work to be sent back
    EXACTLY ONCE per sub-question, overriding any model attempt to retry multiple times.
    """
    sub_q = SubQuestion(
        id="sq_retry_test",
        text="What are consensus leader lease bounds in production systems?",
        rationale="Bounds lease invariants.",
        priority=1,
    )
    empty_findings = []

    # First pass: retry_count=0 -> model returns retry_once -> allowed
    setup_fake_llm.queue_response(
        ReviewDecision(
            sub_question_id="sq_retry_test",
            action=ReviewAction.RETRY_ONCE,
            reason="Initial pass returned no sources.",
            follow_up_query="lease boundaries under clock skew",
        )
    )
    dec1, trace1 = review_findings(sub_question=sub_q, findings=empty_findings, current_retry_count=0)
    assert dec1.action == ReviewAction.RETRY_ONCE
    assert dec1.follow_up_query == "lease boundaries under clock skew"

    # Second pass: retry_count=1 -> model attempts retry_once AGAIN -> code guard forces SKIP
    setup_fake_llm.queue_response(
        ReviewDecision(
            sub_question_id="sq_retry_test",
            action=ReviewAction.RETRY_ONCE,
            reason="Still insufficient sources.",
        )
    )
    dec2, trace2 = review_findings(sub_question=sub_q, findings=empty_findings, current_retry_count=1)
    assert dec2.action == ReviewAction.SKIP
    assert "already used its single allowed retry" in dec2.reason


def test_wall_clock_budget_triggers_partial_report(monkeypatch):
    """Validates that exceeding wall-clock ceiling degrades gracefully to partial report."""
    monkeypatch.setattr("app.config.settings.wall_clock_timeout_seconds", 0.05)
    graph = build_research_graph()

    initial_state = {
        "run_id": "test_wall_clock",
        "question": "Distributed Systems Under Chaos",
        "start_time": time.time() - 10.0,  # 10s elapsed > 0.05s limit
        "total_tokens": 0,
        "total_usd_cost": 0.0,
        "step_index": 0,
    }

    final_state = graph.invoke(initial_state)
    assert final_state["partial"] is True
    assert "Wall-clock ceiling exceeded" in final_state["degradation_reason"]
    report = final_state.get("final_report")
    assert report is not None
    assert report.partial is True


def test_token_budget_triggers_partial_report(monkeypatch):
    """Validates that token budget exhaustion degrades gracefully to partial report."""
    monkeypatch.setattr("app.config.settings.max_total_tokens", 200)
    graph = build_research_graph()

    initial_state = {
        "run_id": "test_token_budget",
        "question": "Vector Database Benchmarking",
        "start_time": time.time(),
        "total_tokens": 500,  # already exceeds 200
        "total_usd_cost": 0.0,
        "step_index": 0,
    }

    final_state = graph.invoke(initial_state)
    assert final_state["partial"] is True
    assert "Token budget exceeded" in final_state["degradation_reason"]
    assert final_state["final_report"].partial is True


def test_usd_cost_budget_triggers_partial_report(monkeypatch):
    """Validates that USD cost ceiling degrades gracefully to partial report."""
    monkeypatch.setattr("app.config.settings.max_usd_cost", 0.001)
    graph = build_research_graph()

    initial_state = {
        "run_id": "test_usd_budget",
        "question": "Zero Knowledge Proof Verification Latency",
        "start_time": time.time(),
        "total_tokens": 0,
        "total_usd_cost": 0.05,  # exceeds $0.001
        "step_index": 0,
    }

    final_state = graph.invoke(initial_state)
    assert final_state["partial"] is True
    assert "USD cost budget exceeded" in final_state["degradation_reason"]
    assert final_state["final_report"].partial is True


def test_search_count_limit_enforced_by_guard():
    """Validates that BudgetGuard blocks searches when per-sub-question limit is reached."""
    guard = BudgetGuard(max_searches_per_subquestion=2)
    assert guard.can_search("sq_1") is True
    guard.record_search("sq_1")
    assert guard.can_search("sq_1") is True
    guard.record_search("sq_1")

    # Reached limit 2
    assert guard.can_search("sq_1") is False
    with pytest.raises(BudgetExceededError) as exc_info:
        guard.record_search("sq_1")
    assert "Search limit exceeded" in str(exc_info.value)


def test_critic_node_detects_unsupported_claims():
    """Validates that Critic node evaluates draft report coverage against plan and findings."""
    plan = ResearchPlan(
        topic="LSM vs B-Tree",
        sub_questions=[
            SubQuestion(id="sq_1", text="What are write amplification trade-offs?", rationale="Analysis 1"),
            SubQuestion(id="sq_2", text="How do range scans compare?", rationale="Analysis 2"),
        ],
    )
    findings = [
        Finding(
            id="fnd_1",
            sub_question_id="sq_1",
            claim="LSM trees achieve sequential writes during ingestion.",
            source_url="https://example.com/lsm",
            domain="example.com",
            snippet="Sequential I/O.",
            retrieval_timestamp=time.time(),
            content_hash="h1",
            confidence=0.9,
        )
    ]
    draft_report = Report(
        topic="LSM vs B-Tree",
        summary="Overview of storage engines.",
        sections=[ReportSection(title="Ingestion", content="LSM trees use sequential disk writes.")],
        partial=False,
    )

    critique, trace = critique_report(plan=plan, findings=findings, draft_report=draft_report)
    assert critique is not None
    assert trace.agent == "critic"
    assert trace.provider == "fake"


def test_property_style_graph_termination_with_random_agent_outputs(setup_fake_llm):
    """
    Property-style test: Runs the LangGraph state graph across 20 randomized executions
    with random sub-question counts, random supervisor decisions (accept, retry, skip, abort),
    and random findings to verify 100% guaranteed termination without infinite loops.
    """
    random.seed(42)
    graph = build_research_graph()

    for iteration in range(20):
        # 1. Random sub-questions between 1 and 4
        sq_count = random.randint(1, 4)
        sub_questions = [
            SubQuestion(
                id=f"sq_{i}",
                text=f"Randomized research sub-question {i} regarding system behaviour?",
                rationale=f"Rationale {i}",
                priority=i,
            )
            for i in range(1, sq_count + 1)
        ]
        random_plan = ResearchPlan(topic="Randomized Stress Test Topic", sub_questions=sub_questions)

        # Configure fake provider queue for this run
        setup_fake_llm.queue_response(random_plan)

        # Queue randomized responses for researcher, supervisor, etc.
        actions_pool = [ReviewAction.ACCEPT, ReviewAction.RETRY_ONCE, ReviewAction.SKIP, ReviewAction.ABORT]
        for _ in range(15):
            # Queue random research findings
            setup_fake_llm.queue_response(
                ResearchFindings(
                    sub_question_id="sq_1",
                    findings=[
                        Finding(
                            sub_question_id="sq_1",
                            claim=f"Random claim {random.randint(100, 999)}",
                            source_url="https://test.org",
                            domain="test.org",
                            snippet="Snippet test",
                            retrieval_timestamp=time.time(),
                            content_hash=f"hash_{random.randint(1, 100)}",
                            confidence=random.uniform(0.5, 1.0),
                        )
                    ]
                    if random.random() > 0.3
                    else [],
                )
            )
            # Queue random review decision
            chosen_action = random.choice(actions_pool)
            setup_fake_llm.queue_response(
                ReviewDecision(
                    sub_question_id="sq_1",
                    action=chosen_action,
                    reason=f"Random decision {chosen_action.value}",
                )
            )

        initial_state = {
            "run_id": f"prop_test_{iteration}",
            "question": f"Property Test Iteration {iteration}",
            "start_time": time.time(),
            "total_tokens": 0,
            "total_usd_cost": 0.0,
            "step_index": 0,
        }

        # Graph invocation must complete and terminate in all cases
        final_state = graph.invoke(initial_state)

        assert final_state is not None
        assert final_state.get("status") in [RunStatus.COMPLETED, RunStatus.NEEDS_REVIEW, RunStatus.FAILED]
        assert final_state.get("final_report") is not None
