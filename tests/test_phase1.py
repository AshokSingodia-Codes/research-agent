import re

import pytest

from app.agents.planner import plan_research
from app.agents.researcher import research_subquestion
from app.agents.writer import write_report
from app.llm.base import InvalidOutputError
from app.llm.client import llm_client
from app.llm.fake_provider import FakeProvider
from app.llm.pricing import calculate_cost
from app.models.schemas import (
    Contradiction,
    Finding,
    Report,
    ResearchFindings,
    ResearchPlan,
    SubQuestion,
)


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def _jaccard_overlap(a: str, b: str) -> float:
    set_a = _tokenize(a)
    set_b = _tokenize(b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a.intersection(set_b)) / len(set_a.union(set_b))


@pytest.fixture(autouse=True)
def setup_fake_provider():
    fake = FakeProvider()
    llm_client.set_override_provider(fake)
    yield fake
    llm_client.set_override_provider(None)


def test_planner_returns_valid_quality_on_5_questions(setup_fake_provider):
    """
    Planner rule-based quality check on 5 distinct technical questions:
    1. Returns 3-8 sub-questions
    2. None shorter than 6 words
    3. Each ends in a question mark or imperative clause
    4. No duplicates by token overlap (Jaccard < 0.7)
    """
    sample_questions = [
        "How do distributed consensus algorithms handle asymmetric network partitions?",
        "What are the performance trade-offs of LSM-trees versus B-trees in high-write workloads?",
        "How does zero-knowledge proof verification scale on layer-2 blockchains?",
        "What are the memory and cache locality implications of columnar storage engines?",
        "How do modern garbage collectors in JVM minimize stop-the-world pauses in multi-terabyte heaps?",
    ]

    for q in sample_questions:
        plan, trace = plan_research(q)
        assert isinstance(plan, ResearchPlan)
        assert 3 <= len(plan.sub_questions) <= 8, f"Sub-question count out of bounds for: {q}"
        assert trace.provider == "fake"
        assert trace.total_tokens > 0

        # Quality assertions per sub-question
        for sq in plan.sub_questions:
            assert isinstance(sq, SubQuestion)
            words = sq.text.strip().split()
            assert len(words) >= 6, f"Sub-question '{sq.text}' is shorter than 6 words"
            assert (
                sq.text.endswith("?")
                or sq.text.endswith(".")
                or any(
                    sq.text.lower().startswith(w) for w in ["analyze", "compare", "evaluate", "examine", "determine"]
                )
            ), f"Sub-question '{sq.text}' does not end in question or imperative structure"
            assert sq.rationale and len(sq.rationale.split()) >= 3
            assert sq.priority in [1, 2, 3, 4, 5]
            assert isinstance(sq.expected_source_types, list)

        # Overlap check between all pairs
        for i in range(len(plan.sub_questions)):
            for j in range(i + 1, len(plan.sub_questions)):
                sq1 = plan.sub_questions[i]
                sq2 = plan.sub_questions[j]
                overlap = _jaccard_overlap(sq1.text, sq2.text)
                assert overlap < 0.70, (
                    f"Sub-questions '{sq1.text}' and '{sq2.text}' exceed similarity threshold (overlap: {overlap:.2f})"
                )


def test_schema_validation_failure_triggers_repair_and_succeeds():
    """Validates that a validation failure triggers repair and succeeds on the second attempt."""
    fake = FakeProvider()
    llm_client.set_override_provider(fake)

    # First return invalid output, second return valid ResearchPlan
    fake.return_raw_invalid_json = True
    valid_plan = ResearchPlan(
        topic="Consensus",
        sub_questions=[
            SubQuestion(
                id="sq_1",
                text="What are leader lease invariants in Raft consensus?",
                rationale="Prevents split-brain leadership.",
                priority=1,
            ),
            SubQuestion(
                id="sq_2",
                text="How does pre-vote phase eliminate election disruption?",
                rationale="Prevents partitioned nodes from bumping terms.",
                priority=2,
            ),
            SubQuestion(
                id="sq_3",
                text="What are snapshot compaction latency characteristics?",
                rationale="Prevents disk write saturation.",
                priority=3,
            ),
        ],
    )
    fake.queue_response(valid_plan)

    plan, trace = plan_research("Consensus Protocols")
    assert isinstance(plan, ResearchPlan)
    assert len(plan.sub_questions) == 3
    assert trace.retries == 1, "Expected 1 repair attempt recorded in trace"


def test_schema_validation_exceeding_max_repairs_raises_structured_error():
    """Validates that validation failures exceeding max repairs (2) raise InvalidOutputError."""
    fake = FakeProvider()
    llm_client.set_override_provider(fake)

    # Queue 4 invalid errors to surpass max_repairs=2
    fake.queue_response(InvalidOutputError("Format failure 1"))
    fake.queue_response(InvalidOutputError("Format failure 2"))
    fake.queue_response(InvalidOutputError("Format failure 3"))

    with pytest.raises(InvalidOutputError) as exc_info:
        plan_research("Fault tolerance", max_subquestions=3)

    assert "Schema validation failed after 2 repair attempts" in str(exc_info.value)


def test_researcher_specialist_output_and_trace():
    """Validates Researcher specialist produces structured findings with exact provenance."""
    fake = FakeProvider()
    llm_client.set_override_provider(fake)

    sub_q = SubQuestion(
        id="sq_1",
        text="What are the failover benchmarks of Paxos implementations in high throughput?",
        rationale="Determines failover SLA.",
        priority=1,
    )
    docs = [
        {
            "url": "https://example.com/paxos-paper.pdf",
            "domain": "example.com",
            "text": "Paxos leader election completes within 15ms under LAN latency.",
        }
    ]

    findings_res, trace = research_subquestion(sub_question=sub_q, source_documents=docs)
    assert isinstance(findings_res, ResearchFindings)
    assert findings_res.sub_question_id == "sq_1"
    assert len(findings_res.findings) >= 1
    for f in findings_res.findings:
        assert isinstance(f, Finding)
        assert f.sub_question_id == "sq_1"
        assert f.claim
        assert f.source_url
        assert f.domain
        assert f.content_hash
        assert 0.0 <= f.confidence <= 1.0

    assert trace.agent == "researcher"
    assert trace.provider == "fake"
    assert trace.total_tokens > 0


def test_writer_specialist_with_conflicting_evidence():
    """Validates Writer specialist synthesizes report, citations map, and conflicting evidence."""
    fake = FakeProvider()
    llm_client.set_override_provider(fake)

    findings = [
        Finding(
            id="fnd_1",
            sub_question_id="sq_1",
            claim="Raft throughput matches Paxos under uniform workloads.",
            source_url="https://raft.github.io/raft.pdf",
            domain="raft.github.io",
            snippet="Evaluation indicates indistinguishable throughput between Raft and Multi-Paxos.",
            retrieval_timestamp=1700000000.0,
            content_hash="hash_1",
            confidence=0.9,
        )
    ]
    conflicts = [
        Contradiction(
            claim_a="Throughput is identical under all loads",
            claim_b="Multi-Paxos achieves 20% higher throughput under asymmetric network partitions",
            sources=["https://raft.github.io/raft.pdf", "https://paxos.systems/perf"],
            severity="medium",
            explanation="Different performance observed under asymmetric latency injection.",
        )
    ]

    report, trace = write_report(
        topic="Consensus Performance",
        findings=findings,
        conflicting_evidence=conflicts,
        partial=False,
    )

    assert isinstance(report, Report)
    assert report.topic == "Consensus Performance"
    assert len(report.sections) >= 1
    assert len(report.citations) >= 1
    assert report.partial is False
    assert trace.agent == "writer"
    assert trace.provider == "fake"


def test_pricing_and_cost_accounting():
    """Validates cost calculations across multiple model tiers."""
    cost_haiku = calculate_cost("anthropic", "claude-haiku-5-5", tokens_in=100_000, tokens_out=50_000)
    # Haiku: $0.80/1M in ($0.08) + $4.00/1M out ($0.20) = $0.28
    assert round(cost_haiku, 4) == 0.2800

    cost_sonnet = calculate_cost("anthropic", "claude-sonnet-5-5", tokens_in=100_000, tokens_out=50_000)
    # Sonnet: $3.00/1M in ($0.30) + $15.00/1M out ($0.75) = $1.05
    assert round(cost_sonnet, 4) == 1.0500

    cost_fake = calculate_cost("fake", "fake-model", tokens_in=500_000, tokens_out=500_000)
    assert cost_fake == 0.0


@pytest.mark.live
def test_live_anthropic_planner_smoke():
    """Live smoke test using real Anthropic API key (skipped by default)."""
    import os

    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    plan, trace = plan_research("What are the architectural trade-offs of LSM trees vs B-trees?")
    assert len(plan.sub_questions) >= 3
    assert trace.provider == "anthropic"
