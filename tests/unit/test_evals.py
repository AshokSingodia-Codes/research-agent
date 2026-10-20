from pathlib import Path

from app.evals.cli import load_dataset, run_evaluations
from app.evals.scorers.baseline import NaiveBaselineRunner
from app.evals.scorers.budget import score_budget_adherence
from app.evals.scorers.citation import score_citations
from app.evals.scorers.contradiction import score_contradiction_detection
from app.evals.scorers.coverage import score_plan_coverage
from app.evals.scorers.dedupe import score_source_diversity
from app.models.schemas import Finding, Report, ReportSection, ResearchPlan, SubQuestion


def test_core_dataset_structure_and_count():
    dataset_file = Path("app/evals/datasets/core.jsonl")
    assert dataset_file.exists()
    items = load_dataset(dataset_file)
    assert len(items) >= 16, f"Expected at least 16 queries, got {len(items)}"
    for item in items:
        assert "id" in item
        assert "topic" in item
        assert "category" in item


def test_citation_scorer_logic():
    from app.models.schemas import CitationKey

    finding = Finding(
        id="f_valid_1",
        sub_question_id="sq_1",
        claim="Raft provides consensus.",
        source_url="https://raft.github.io/raft.pdf",
        domain="raft.github.io",
        snippet="Raft decomposes consensus.",
        retrieval_timestamp=100.0,
        content_hash="h1",
        confidence=0.9,
    )
    citation_key_1 = CitationKey(
        key="[1]",
        finding_id=finding.id,
        source_url=finding.source_url,
        domain=finding.domain,
        claim=finding.claim,
    )
    report = Report(
        topic="Consensus",
        summary="Summary",
        sections=[ReportSection(title="Overview", content="Raft is leader-based [1]. Phantom cite [2].", citations=["[1]", "[2]"])],
        citations={
            "[1]": citation_key_1,
            # [2] is hallucinated
        },
    )
    scores = score_citations(report, [finding])
    assert scores["total_citations_referenced"] == 2
    assert scores["valid_citations"] == 1
    assert scores["hallucinated_citations"] == 1
    assert scores["citation_validity_ratio"] == 0.5


def test_plan_coverage_scorer_logic():
    plan = ResearchPlan(
        topic="Consensus",
        sub_questions=[
            SubQuestion(id="sq_1", text="What is Raft log compaction throughput?", rationale="Compaction eval", priority=1),
            SubQuestion(id="sq_2", text="How do network partitions impact latency?", rationale="Partition eval", priority=2),
        ],
    )
    # Report addresses compaction but not partition
    report = Report(
        topic="Consensus",
        summary="Summary",
        sections=[ReportSection(title="Compaction", content="Log compaction throughput in Raft is evaluated here.", citations=[])],
    )
    scores = score_plan_coverage(plan, report)
    assert scores["total_subquestions"] == 2
    assert scores["covered_subquestions"] >= 1
    assert scores["plan_coverage_ratio"] >= 0.5


def test_source_diversity_scorer_logic():
    f1 = Finding(
        id="f1",
        sub_question_id="sq_1",
        claim="Claim A",
        source_url="https://domain-a.org/doc",
        domain="domain-a.org",
        snippet="Snippet A",
        retrieval_timestamp=100.0,
        content_hash="hash_a",
    )
    f2 = Finding(
        id="f2",
        sub_question_id="sq_1",
        claim="Claim B",
        source_url="https://domain-b.org/doc",
        domain="domain-b.org",
        snippet="Snippet B",
        retrieval_timestamp=100.0,
        content_hash="hash_b",
    )
    scores = score_source_diversity([f1, f2])
    assert scores["unique_domains_count"] == 2
    assert scores["domain_diversity_ratio"] == 1.0
    assert scores["duplicate_content_ratio"] == 0.0


def test_budget_adherence_scorer_logic():
    scores = score_budget_adherence(total_tokens=1500, total_usd_cost=0.03, wall_clock_seconds=5.0)
    assert scores["budget_compliance"] is True
    assert scores["token_limit_exceeded"] is False
    assert scores["usd_limit_exceeded"] is False


def test_contradiction_scorer_logic():
    report_with_contra = Report(
        topic="Sync Replication",
        summary="Summary",
        sections=[],
        conflicting_evidence=[{
            "topic": "Latency",
            "claim_a": "High latency in WAN",
            "claim_b": "Low latency with pipelining",
            "explanation": "Conflicting findings on commit latency under WAN conditions",
        }],
    )
    scores = score_contradiction_detection(report_with_contra, contradiction_expected=True)
    assert scores["contradiction_detected"] is True
    assert scores["contradiction_accuracy"] == 1.0


def test_naive_baseline_runner_execution():
    runner = NaiveBaselineRunner()
    res = runner.run("Distributed Consensus Protocols")
    assert "report" in res
    assert "total_tokens" in res
    assert res["report"].topic == "Distributed Consensus Protocols"


def test_run_evaluations_cli_mock_integration(tmp_path):
    dataset_file = Path("app/evals/datasets/core.jsonl")
    out_dir = tmp_path / "eval_out"
    # Run against first 2 queries in mock mode
    res = run_evaluations(dataset_file, out_dir, mock=True, sample_limit=2)
    assert "summary" in res
    assert "results" in res
    assert len(res["results"]) == 2
    assert (out_dir / "eval_report.json").exists()
    assert (out_dir / "eval_report.md").exists()
