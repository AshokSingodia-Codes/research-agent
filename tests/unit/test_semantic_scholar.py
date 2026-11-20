"""Unit tests for the Semantic Scholar search adapter."""

from app.tools.semantic_scholar import parse_semantic_scholar_item


def test_parse_semantic_scholar_item():
    item = {
        "paperId": "abc12345",
        "title": "Signs as Tokens: A Retrieval-Enhanced Multilingual Sign Language Generator",
        "abstract": "We present a sign language generation model using discrete VQ codebooks.",
        "year": 2024,
        "authors": [{"name": "Jane Doe"}, {"name": "John Smith"}],
        "citationCount": 42,
        "venue": "CVPR",
        "externalIds": {"ArXiv": "2411.17799", "DOI": "10.1145/test"},
        "url": "https://www.semanticscholar.org/paper/abc12345",
    }

    finding = parse_semantic_scholar_item(item, sub_question_id="sq_1", run_id="run_s2_test")
    assert finding is not None
    assert "Signs as Tokens" in finding.claim
    assert "Jane Doe et al." in finding.claim
    assert finding.domain == "semanticscholar.org"
    assert finding.source_url == "https://www.semanticscholar.org/paper/abc12345"
    assert "2024" in finding.extracted_text
    assert finding.confidence > 0.9

