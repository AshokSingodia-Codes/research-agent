import pytest

from src.agents.planner import plan_research
from src.agents.researcher import research_subquestion
from src.agents.writer import write_research_report
from src.models.schemas import EvidenceRecord, SearchStatus, SubQuestion


@pytest.mark.unit
def test_planner_in_isolation():
    """Validates that Planner produces structured, non-vague sub-questions respecting budgets."""
    topic = "Distributed Consensus in Multi-Raft Architectures"
    max_sq = 3

    plan, trace = plan_research(topic, max_subquestions=max_sq)

    assert plan is not None
    assert plan.research_topic == topic
    assert 1 <= len(plan.sub_questions) <= max_sq

    for sq in plan.sub_questions:
        assert sq.id.startswith("sq_")
        assert len(sq.question.strip()) > 10
        assert len(sq.rationale.strip()) > 10
        assert len(sq.expected_evidence.strip()) > 10
        assert sq.status == "pending"

    assert trace.node_name == "planner"
    assert trace.latency_ms >= 0
    assert trace.tokens_used >= 0


@pytest.mark.unit
def test_researcher_in_isolation():
    """Validates that Researcher returns structured findings with strict source provenance."""
    sub_q = SubQuestion(
        id="sq_1",
        question="What is the leader election latency in Multi-Raft?",
        rationale="Evaluates consensus recovery time.",
        expected_evidence="Benchmark figures in milliseconds."
    )

    output, trace = research_subquestion(sub_q)

    assert output.sub_question_id == "sq_1"
    assert output.search_status.status == "success"
    assert len(output.findings) > 0

    for finding in output.findings:
        assert finding.claim != ""
        assert finding.source_url.startswith("http")
        assert finding.domain != ""
        assert finding.snippet != ""
        assert finding.retrieval_timestamp != ""

    assert trace.node_name == "researcher"
    assert len(trace.tool_calls) >= 1
    assert trace.tool_calls[0]["tool"] == "search_and_extract"


@pytest.mark.unit
def test_researcher_handles_search_failure_gracefully(monkeypatch):
    """Validates that search failures yield structured status instead of raising uncaught exceptions."""
    from src.tools import search

    def mock_failing_search(*args, **kwargs):
        return [], SearchStatus(
            status="rate_limited",
            error_message="HTTP 429 Too Many Requests",
            result_count=0
        )

    monkeypatch.setattr(search, "search_and_extract", mock_failing_search)

    sub_q = SubQuestion(
        id="sq_fail",
        question="What happens when rate limited?",
        rationale="Fault tolerance test.",
        expected_evidence="None"
    )

    output, trace = research_subquestion(sub_q)

    assert output.search_status.status == "rate_limited"
    assert output.findings == []
    assert trace.node_name == "researcher"
    assert "rate_limited" in trace.output_summary


@pytest.mark.unit
def test_writer_in_isolation():
    """Validates that Writer synthesizes evidence into a cited report matching provenance."""
    topic = "Multi-Raft Consensus Performance"
    mock_findings = [
        EvidenceRecord(
            claim="Multi-Raft shards data into disjoint consensus groups.",
            source_url="https://arxiv.org/abs/2401.0001",
            domain="arxiv.org",
            snippet="Sharding into multiple raft groups reduces lock contention.",
            confidence=0.9
        ),
        EvidenceRecord(
            claim="Average leader election latency is under 50ms with heartbeats set to 10ms.",
            source_url="https://github.com/tikv/tikv",
            domain="github.com",
            snippet="Empirical benchmarks indicate election time is below 50ms.",
            confidence=0.95
        ),
    ]

    report, trace = write_research_report(topic, mock_findings)

    assert report.topic == topic
    assert report.executive_summary != ""
    assert len(report.sections) >= 2
    assert len(report.citations) == 2

    # Check citation provenance matching
    assert report.citations[0].citation_id == "[1]"
    assert report.citations[0].source_url == "https://arxiv.org/abs/2401.0001"
    assert report.citations[0].is_verified is True

    assert report.citations[1].citation_id == "[2]"
    assert report.citations[1].source_url == "https://github.com/tikv/tikv"
    assert report.citations[1].is_verified is True

    assert trace.node_name == "writer"
    assert trace.latency_ms >= 0
