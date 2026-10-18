import time

from app.models.schemas import CitationKey, Finding, Report, ReportSection
from app.services.citation_validator import validate_citations


def test_citation_validator_rejects_fabricated_citations():
    """Validates that citation validator detects and strips fabricated citations not in evidence."""
    verified_findings = [
        Finding(
            id="fnd_real_1",
            sub_question_id="sq_1",
            claim="Raft provides leader leases to avoid split-brain.",
            source_url="https://raft.github.io/raft.pdf",
            domain="raft.github.io",
            snippet="Leader leases enforce quorum safety.",
            retrieval_timestamp=time.time(),
            content_hash="h_real",
        )
    ]

    report = Report(
        topic="Raft Consensus",
        summary="Overview of Raft [1].",
        sections=[
            ReportSection(
                title="Leader Safety",
                content="Raft uses leader leases [1]. It also claims 99.999% throughput without consensus [2].",
                citations=["[1]", "[2]"],
            )
        ],
        citations={
            "[1]": CitationKey(
                key="[1]",
                finding_id="fnd_real_1",
                source_url="https://raft.github.io/raft.pdf",
                domain="raft.github.io",
                claim="Raft provides leader leases.",
            ),
            # Fabricated citation [2] pointing to nonexistent finding
            "[2]": CitationKey(
                key="[2]",
                finding_id="fnd_fake_999",
                source_url="https://hallucinated-source.com/fake.pdf",
                domain="hallucinated-source.com",
                claim="Fabricated claim without evidence.",
            ),
        },
    )

    validated, coverage, warnings = validate_citations(report, verified_findings, threshold=0.80)
    assert coverage == 0.50  # 1 out of 2 valid
    assert "[1]" in validated.citations
    assert "[2]" not in validated.citations  # Fabricated citation stripped
    assert len(warnings) >= 1
    assert "Fabricated citation" in warnings[0]


def test_citation_validator_filters_hallucinated_sources():
    """Validates that citation validator confirms real sources and flags hallucinated citations."""
    real_evidence = [
        Finding(
            id="fnd_1",
            sub_question_id="sq_1",
            claim="Multi-Raft shards data into consensus partitions.",
            source_url="https://arxiv.org/abs/2401.0001",
            domain="arxiv.org",
            snippet="Partitions reduce write bottlenecks.",
            retrieval_timestamp=time.time(),
            content_hash="h_1",
        )
    ]

    report = Report(
        topic="Multi-Raft Sharding",
        summary="Summary of sharding [1].",
        sections=[
            ReportSection(
                title="Sharding Architecture",
                content="Content referencing [1] and [2]",
                citations=["[1]", "[2]"],
            )
        ],
        citations={
            "[1]": CitationKey(
                key="[1]",
                finding_id="fnd_1",
                source_url="https://arxiv.org/abs/2401.0001",
                domain="arxiv.org",
                claim="Multi-Raft shards data",
            ),
            "[2]": CitationKey(
                key="[2]",
                finding_id="fnd_fake",
                source_url="https://fake-hallucinated-source.com/paper",
                domain="fake-hallucinated-source.com",
                claim="Hallucinated claim",
            ),
        },
    )

    validated, coverage, warnings = validate_citations(report, real_evidence)
    assert len(validated.citations) == 1
    assert "[1]" in validated.citations
    assert "[2]" not in validated.citations
    assert len(warnings) == 1
    assert "fake-hallucinated-source.com" in warnings[0]
