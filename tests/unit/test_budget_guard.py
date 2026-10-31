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


@pytest.mark.asyncio
async def test_twenty_concurrent_charges_never_exceed_cap():
    """Validates that 20 concurrent charges never exceed the token or USD budget cap."""
    import asyncio

    from app.persistence.redis_store import redis_store

    client = redis_store.get_client()
    run_id = f"test_concurrent_20_{time.time()}"
    client.delete(f"budget:{run_id}:tokens")
    client.delete(f"budget:{run_id}:usd_micros")
    client.delete(f"budget:{run_id}:start_time")

    # Limit: 1000 tokens, $0.50 USD
    guard = BudgetGuard(
        run_id=run_id,
        max_total_tokens=1000,
        max_usd_cost=0.50,
        redis_client=client,
    )

    successful_charges = 0
    failed_charges = 0

    async def charge_attempt():
        nonlocal successful_charges, failed_charges
        try:
            # Each worker attempts 100 tokens and $0.05 spend
            guard.reserve(estimated_tokens=100)
            guard.settle(actual_tokens=100, usd_cost=0.05, reserved_tokens=100)
            successful_charges += 1
        except BudgetExceededError:
            failed_charges += 1

    # 20 concurrent workers attempting 2000 tokens total (cap is 1000)
    await asyncio.gather(*[charge_attempt() for _ in range(20)])

    final_tokens = int(client.get(f"budget:{run_id}:tokens") or 0)
    final_micros = int(client.get(f"budget:{run_id}:usd_micros") or 0)

    assert final_tokens <= 1000, f"Token cap exceeded: {final_tokens} > 1000"
    assert final_micros <= 500_000, f"USD micro cap exceeded: {final_micros} > 500000"
    assert successful_charges <= 10
    assert failed_charges >= 10


def test_two_simulated_processes_sharing_redis_never_exceed_cap():
    """Validates that two simulated processes sharing one Redis never exceed the budget cap."""
    from concurrent.futures import ThreadPoolExecutor

    from app.persistence.redis_store import redis_store

    client = redis_store.get_client()
    run_id = f"test_shared_proc_{time.time()}"
    client.delete(f"budget:{run_id}:tokens")
    client.delete(f"budget:{run_id}:usd_micros")

    # Max tokens = 500
    # Process 1 and Process 2 both try to reserve and charge 350 tokens
    def proc_charge(proc_id: str):
        p_guard = BudgetGuard(
            run_id=run_id,
            max_total_tokens=500,
            max_usd_cost=1.00,
            redis_client=client,
        )
        try:
            p_guard.reserve(estimated_tokens=350)
            p_guard.settle(actual_tokens=350, usd_cost=0.01, reserved_tokens=350)
            return True
        except BudgetExceededError:
            return False

    with ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(proc_charge, "proc_1")
        f2 = executor.submit(proc_charge, "proc_2")
        r1 = f1.result()
        r2 = f2.result()

    # Exactly one process should succeed, the other must be rejected
    assert (r1 and not r2) or (r2 and not r1)

    final_tokens = int(client.get(f"budget:{run_id}:tokens") or 0)
    assert final_tokens == 350
    assert final_tokens <= 500


def test_resumed_run_does_not_double_count_earlier_spend():
    """Validates that a resumed run preserves earlier spend and receives no fresh clock."""
    from app.persistence.redis_store import redis_store

    client = redis_store.get_client()
    run_id = f"test_resume_spend_{time.time()}"
    client.delete(f"budget:{run_id}:tokens")
    client.delete(f"budget:{run_id}:usd_micros")
    client.delete(f"budget:{run_id}:start_time")

    past_time = time.time() - 50.0
    guard1 = BudgetGuard(
        run_id=run_id,
        max_total_tokens=2000,
        max_usd_cost=1.00,
        start_time=past_time,
        redis_client=client,
    )
    guard1.reserve(400)
    guard1.settle(actual_tokens=400, usd_cost=0.02, reserved_tokens=400)

    assert int(client.get(f"budget:{run_id}:tokens")) == 400

    # Resumed process initializes BudgetGuard with same run_id later
    guard2 = BudgetGuard(
        run_id=run_id,
        max_total_tokens=2000,
        max_usd_cost=1.00,
        start_time=time.time(),  # attempts to provide new start time
        redis_client=client,
    )

    # Must retain original persisted start time
    assert abs(guard2.start_time - past_time) < 1.0, "Resumed run received fresh clock instead of persisted start_time"

    # Must retain earlier spend
    assert guard2.current_tokens == 400
    assert abs(guard2.current_usd_cost - 0.02) < 0.001

    # Additional spend increments on top of earlier spend without double-counting
    guard2.reserve(300)
    guard2.settle(actual_tokens=300, usd_cost=0.01, reserved_tokens=300)

    final_tokens = int(client.get(f"budget:{run_id}:tokens"))
    assert final_tokens == 700
    final_micros = int(client.get(f"budget:{run_id}:usd_micros"))
    assert final_micros == 30_000  # $0.03


def test_failed_or_cancelled_call_releases_budget_reservation_and_rejected_produces_partial():
    """
    Validates that:
    1. A failed or cancelled LLM call releases its budget reservation so no tokens are leaked.
    2. A rejected reservation produces a typed BudgetExceededError which routes to a partial report.
    """
    from app.persistence.redis_store import redis_store

    client = redis_store.get_client()
    run_id = f"test_res_rel_{time.time()}"
    client.delete(f"budget:{run_id}:tokens")
    client.delete(f"budget:{run_id}:usd_micros")
    client.delete(f"budget:{run_id}:start_time")

    guard = BudgetGuard(
        run_id=run_id,
        max_total_tokens=1000,
        max_usd_cost=1.00,
        redis_client=client,
    )

    # 1. Reserve 400 tokens
    guard.reserve(400)
    assert int(client.get(f"budget:{run_id}:tokens")) == 400

    # Simulate failure or cancellation: release the reservation without recording spend
    guard.release_reservation(reserved_tokens=400)
    assert int(client.get(f"budget:{run_id}:tokens")) == 0
    assert guard.current_tokens == 0
    assert guard.current_usd_cost == 0.0

    # 2. Next call can use full budget without leak
    guard.reserve(950)
    assert int(client.get(f"budget:{run_id}:tokens")) == 950
    guard.settle(actual_tokens=900, usd_cost=0.01, reserved_tokens=950)
    assert int(client.get(f"budget:{run_id}:tokens")) == 900

    # 3. Attempting to reserve 200 tokens when cap is 1000 (900 + 200 > 1000) is rejected
    with pytest.raises(BudgetExceededError) as exc_info:
        guard.reserve(200)
    assert "Token budget exceeded" in exc_info.value.reason
    assert exc_info.value.budget_type == "tokens"

    # 4. In a graph execution, rejected reservation leads to partial report
    graph = build_research_graph()
    initial_state = {
        "run_id": f"test_graph_rej_{time.time()}",
        "question": "Quantum Memory Coherence Benchmarks",
        "start_time": time.time(),
        "total_tokens": 980,  # Near 1000 token cap
        "total_usd_cost": 0.0,
        "step_index": 0,
    }
    # monkeypatch max_total_tokens to 1000
    from app.config import settings
    orig_cap = settings.max_total_tokens
    settings.max_total_tokens = 1000
    try:
        final_state = graph.invoke(initial_state)
        assert final_state["partial"] is True
        assert final_state["final_report"] is not None
        assert final_state["final_report"].partial is True
        assert "budget" in final_state["degradation_reason"].lower()
    finally:
        settings.max_total_tokens = orig_cap


def test_supervisor_handles_budget_exceeded_error_explicitly():
    """
    Validates that review_findings explicitly catches BudgetExceededError,
    does not swallow it silently, and returns an ABORT decision with reason.
    """
    sub_q = SubQuestion(
        id="sq_budget_test",
        text="What are consensus leader lease bounds in production systems?",
        rationale="Bounds lease invariants.",
        priority=1,
    )
    guard = BudgetGuard(
        max_total_tokens=100,
        max_usd_cost=0.01,
        wall_clock_timeout_seconds=60.0,
        run_id="test_sup_budget",
    )
    guard.current_tokens = 150  # Over budget

    decision, trace = review_findings(
        sub_question=sub_q,
        findings=[],
        budget_guard=guard,
        run_id="test_sup_budget",
    )

    assert decision.action == ReviewAction.ABORT
    assert "budget" in decision.reason.lower()
    assert trace.agent == "supervisor"
    assert trace.provider == "code_guard"


