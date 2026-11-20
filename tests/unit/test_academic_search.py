from app.tools.arxiv import parse_arxiv_feed
from app.tools.openalex import parse_openalex_work, reconstruct_abstract


def test_parse_arxiv_feed():
    xml_sample = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>http://arxiv.org/abs/2301.00001v1</id>
        <published>2023-01-01T00:00:00Z</published>
        <title>Signs as Tokens: Multilingual Sign Language Generation</title>
        <summary>We introduce discrete motion tokenization for sign language synthesis.</summary>
        <author><name>Jane Doe</name></author>
      </entry>
    </feed>
    """
    findings = parse_arxiv_feed(xml_sample, sub_question_id="sq_1", run_id="run_test")
    assert len(findings) == 1
    f = findings[0]
    assert "Signs as Tokens" in f.claim
    assert f.domain == "arxiv.org"
    assert f.source_url == "https://arxiv.org/abs/2301.00001"
    assert f.source_quality_score == 0.95


def test_openalex_reconstruct_abstract():
    inverted = {
        "Deep": [0],
        "learning": [1],
        "for": [2],
        "distributed": [3],
        "systems": [4],
    }
    abstract = reconstruct_abstract(inverted)
    assert abstract == "Deep learning for distributed systems"


def test_parse_openalex_work():
    raw_work = {
        "id": "https://openalex.org/W12345",
        "doi": "https://doi.org/10.1234/example.2023",
        "title": "Empirical Benchmarking of eBPF Overhead",
        "publication_year": 2023,
        "abstract_inverted_index": {"eBPF": [0], "reduces": [1], "latency": [2]},
        "primary_location": {"landing_page_url": "https://doi.org/10.1234/example.2023"},
        "cited_by_count": 42,
    }
    finding = parse_openalex_work(raw_work, sub_question_id="sq_1", run_id="run_test")
    assert finding is not None
    assert "Empirical Benchmarking of eBPF Overhead" in finding.claim
    assert finding.domain == "openalex.org"
    assert "eBPF reduces latency" in finding.snippet
