"""Unit tests for the report renderer service (Markdown and HTML generation)."""

from app.models.schemas import CitationKey, Contradiction, Report, ReportSection
from app.services.renderer import render_html, render_markdown


def create_sample_report() -> Report:
    return Report(
        topic="Neural Architecture Benchmarking",
        summary="A comprehensive survey comparing Transformer vs SSM architectures.",
        sections=[
            ReportSection(
                title="Attention Mechanisms",
                content="Transformers rely on multi-head self-attention with quadratic complexity [1].",
                citations=["[1]"],
            ),
            ReportSection(
                title="Linear Recurrences",
                content="State Space Models achieve linear scaling through selective state spaces [2].",
                citations=["[2]"],
            ),
        ],
        conflicting_evidence=[
            Contradiction(
                severity="high",
                claim_a="SSMs outperform Transformers on associative recall.",
                claim_b="SSMs degrade rapidly on long-context multi-query associative recall.",
                explanation="The dispute arises from differences in hidden dimension capacity and memory budgeting.",
            )
        ],
        citations={
            "[1]": CitationKey(
                key="[1]",
                finding_id="f1",
                source_url="https://arxiv.org/abs/1706.03762",
                domain="arxiv.org",
                claim="Attention Is All You Need",
            ),
            "[2]": CitationKey(
                key="[2]",
                finding_id="f2",
                source_url="https://arxiv.org/abs/2312.00752",
                domain="arxiv.org",
                claim="Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
            ),
        },
        citation_coverage=1.0,
    )


def test_render_markdown():
    report = create_sample_report()
    md = render_markdown(report)

    assert "# Neural Architecture Benchmarking" in md
    assert "## Executive Summary" in md
    assert "## Attention Mechanisms" in md
    assert "[1]" in md
    assert "## Conflicting Evidence & Contradictions" in md
    assert "Dispute #1 [HIGH]" in md
    assert "## References & Evidence Sources" in md
    assert "https://arxiv.org/abs/1706.03762" in md


def test_render_html():
    report = create_sample_report()
    html_output = render_html(report, run_id="run_test_123")

    assert "<!DOCTYPE html>" in html_output
    assert "Neural Architecture Benchmarking" in html_output
    assert "run_test_123" in html_output
    # Check that citation jump link is created
    assert '<a href="#cite-1" class="cite-tag"' in html_output
    assert '<li id="cite-1" class="reference-item">' in html_output
    assert "Dispute #1" in html_output
    assert "arxiv.org" in html_output

