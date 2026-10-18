import time

import pytest

from app.budget.guard import BudgetGuard
from app.graph.parallel_runner import (
    get_ready_subquestions,
    run_parallel_fanout,
)
from app.llm.client import llm_client
from app.llm.fake_provider import FakeProvider
from app.models.enums import ToolStatus
from app.models.schemas import Finding, SubQuestion
from app.tools.contradiction import detect_contradictions
from app.tools.dedupe import canonicalize_url, hash_content
from app.tools.fetch_extract import extract_full_page
from app.tools.scoring import calculate_source_quality
from app.tools.search import DomainCircuitBreaker, _process_raw_results


@pytest.fixture(autouse=True)
def setup_fake_llm():
    fake = FakeProvider()
    llm_client.set_override_provider(fake)
    yield fake
    llm_client.set_override_provider(None)


def test_url_canonicalization():
    """Validates stripping of tracking parameters, port/path normalization, and lowercasing."""
    raw_1 = "https://WWW.Example.com/Docs/Architecture/?utm_source=twitter&ref=share#section"
    raw_2 = "https://example.com/docs/architecture"
    assert canonicalize_url(raw_1) == canonicalize_url(raw_2)

    raw_tracking = "http://tech.blog/post/?utm_medium=email&fbclid=12345"
    assert "fbclid" not in canonicalize_url(raw_tracking)
    assert "utm_medium" not in canonicalize_url(raw_tracking)


def test_content_hashing_deduplication():
    """Validates that identical normalized text produces matching SHA-256 hash."""
    t1 = "  Distributed Consensus   requires strong consistency. \n\n"
    t2 = "Distributed consensus requires strong consistency."
    assert hash_content(t1) == hash_content(t2)


def test_deduplication_and_domain_cap_enforcement():
    """Validates that deduplication by URL & content hash and domain caps (max 2) are enforced."""
    raw_results = [
        {"url": "https://docs.db.com/intro", "snippet": "Introduction to Raft replication protocol."},
        # Same URL with tracking param -> must be deduplicated
        {"url": "https://docs.db.com/intro?utm_source=hackernews", "snippet": "Duplicate URL intro."},
        # Same domain 2nd result -> allowed
        {"url": "https://docs.db.com/benchmarks", "snippet": "Throughput scalability benchmarks."},
        # Same domain 3rd result -> exceeds domain cap (max 2) -> must be filtered
        {"url": "https://docs.db.com/internals", "snippet": "Internal storage log compaction."},
        # Different domain -> allowed
        {"url": "https://stanford.edu/paper.pdf", "snippet": "Formal verification of Raft."},
        # Duplicate content from different site -> must be deduplicated by content hash
        {"url": "https://mirror.org/paper.pdf", "snippet": "Formal verification of Raft."},
    ]

    result = _process_raw_results(
        raw_results=raw_results,
        query="Raft consensus",
        sub_question_id="sq_1",
        max_results=6,
        max_per_domain=2,
        fetch_full_page=False,
        token_cap=1000,
    )

    assert result.status == ToolStatus.OK
    findings = result.findings
    # Should keep docs.db.com/intro, docs.db.com/benchmarks, stanford.edu/paper.pdf
    domains = [f.domain for f in findings]
    assert domains.count("docs.db.com") == 2
    assert domains.count("stanford.edu") == 1
    assert "mirror.org" not in domains  # filtered due to duplicate content hash


def test_circuit_breaker_trips_after_failures():
    """Validates that DomainCircuitBreaker trips open after 3 consecutive failures."""
    cb = DomainCircuitBreaker(failure_threshold=3)
    domain = "unreliable-host.org"

    assert cb.is_tripped(domain) is False
    cb.record_failure(domain)
    assert cb.is_tripped(domain) is False
    cb.record_failure(domain)
    assert cb.is_tripped(domain) is False
    cb.record_failure(domain)

    # Now tripped
    assert cb.is_tripped(domain) is True

    # Recovers on success
    cb.record_success(domain)
    assert cb.is_tripped(domain) is False


def test_tool_failure_statuses_route_correctly(monkeypatch):
    """Validates structured ToolResult mapping for each failure path."""
    # 1. No results
    empty_res = _process_raw_results([], "query", "sq_1", 3, 2, False, 1000)
    assert empty_res.status == ToolStatus.NO_RESULTS

    # 2. Paywalled simulation
    monkeypatch.setattr(
        "httpx.Client.get",
        lambda *args, **kwargs: type("Resp", (), {"status_code": 402, "text": "Payment Required"})(),
    )
    text, status = extract_full_page("https://paywalled-journal.com/article")
    assert status == ToolStatus.PAYWALLED

    # 3. Timeout simulation
    import httpx

    def raise_timeout(*args, **kwargs):
        raise httpx.TimeoutException("Connection timed out")

    monkeypatch.setattr("httpx.Client.get", raise_timeout)
    text, status = extract_full_page("https://slow-server.org")
    assert status == ToolStatus.TIMEOUT


def test_source_quality_heuristic_scoring():
    """Validates quality scoring elevates authoritative domains and penalizes low-grade sites."""
    q_edu = calculate_source_quality("https://cs.stanford.edu/paper.pdf", "cs.stanford.edu", "benchmark evaluations")
    q_blog = calculate_source_quality("https://medium.com/@user/opinion", "medium.com", "my thoughts")

    assert q_edu >= 0.85
    assert q_blog <= 0.60
    assert q_edu > q_blog


def test_contradiction_detector_flags_planted_conflict():
    """Validates that ContradictionDetector flags conflicting claims between different sources."""
    findings = [
        Finding(
            id="fnd_a",
            sub_question_id="sq_1",
            claim="Raft achieves higher throughput than Multi-Paxos under leader election contention.",
            source_url="https://raft.github.io/paper.pdf",
            domain="raft.github.io",
            snippet="Evaluations demonstrate superior throughput.",
            retrieval_timestamp=time.time(),
            content_hash="hash_a",
            confidence=0.9,
        ),
        Finding(
            id="fnd_b",
            sub_question_id="sq_1",
            claim="Multi-Paxos achieves lower throughput and higher latency than Raft under heavy contention.",
            source_url="https://paxos.systems/eval",
            domain="paxos.systems",
            snippet="Paxos throughput is significantly lower.",
            retrieval_timestamp=time.time(),
            content_hash="hash_b",
            confidence=0.85,
        ),
        Finding(
            id="fnd_c",
            sub_question_id="sq_1",
            claim="Multi-Paxos demonstrates superior throughput compared to Raft under asymmetric networks.",
            source_url="https://dist-sys.stanford.edu/paper",
            domain="stanford.edu",
            snippet="Paxos achieves higher throughput in partitioned networks.",
            retrieval_timestamp=time.time(),
            content_hash="hash_c",
            confidence=0.95,
        ),
    ]

    conflicts = detect_contradictions(findings)
    assert len(conflicts) >= 1
    c = conflicts[0]
    assert len(c.sources) == 2
    assert c.severity in ["medium", "high"]


@pytest.mark.asyncio
async def test_parallel_fanout_deterministic_merge():
    """
    Validates that parallel fan-out produces identical, deterministically ordered results
    regardless of completion timing or execution order.
    """
    guard1 = BudgetGuard()
    guard2 = BudgetGuard()

    sub_questions = [
        SubQuestion(id="sq_alpha", text="Alpha query on distributed state?", rationale="Alpha", priority=1),
        SubQuestion(id="sq_beta", text="Beta query on recovery log compaction?", rationale="Beta", priority=2),
        SubQuestion(id="sq_gamma", text="Gamma query on partition tolerances?", rationale="Gamma", priority=3),
    ]

    # Run 1
    f1, d1, t1 = await run_parallel_fanout(sub_questions, guard1, "run_p1", concurrency_limit=3)
    # Run 2
    f2, d2, t2 = await run_parallel_fanout(sub_questions, guard2, "run_p2", concurrency_limit=3)

    assert len(f1) == len(f2)
    assert len(d1) == len(d2)

    # Assert exact deterministic ordering
    for i in range(len(f1)):
        assert f1[i].sub_question_id == f2[i].sub_question_id
        assert f1[i].claim == f2[i].claim
        assert f1[i].source_url == f2[i].source_url

    for i in range(len(d1)):
        assert d1[i].sub_question_id == d2[i].sub_question_id
        assert d1[i].action == d2[i].action


def test_concurrency_atomic_budget_counters():
    """Validates that concurrent workers increment BudgetGuard atomically with exact sums."""
    import concurrent.futures

    guard = BudgetGuard(max_total_tokens=1_000_000, max_usd_cost=100.0)

    def worker_spend():
        for _ in range(50):
            # spend 10 tokens and $0.0001
            guard.record_spend(tokens=10, usd_cost=0.0001)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker_spend) for _ in range(8)]
        concurrent.futures.wait(futures)

    # 8 workers * 50 iterations * 10 tokens = 4000 tokens
    assert guard.current_tokens == 4000
    # 8 workers * 50 iterations * 0.0001 = $0.04
    assert round(guard.current_usd_cost, 4) == 0.0400


def test_ready_subquestions_dependency_resolution():
    """Validates that depends_on constraints are strictly respected by dispatcher."""
    all_sqs = [
        SubQuestion(id="sq_1", text="Root question", rationale="R1", depends_on=[]),
        SubQuestion(id="sq_2", text="Dependent question", rationale="R2", depends_on=["sq_1"]),
        SubQuestion(id="sq_3", text="Chained question", rationale="R3", depends_on=["sq_2"]),
    ]

    # Initially only sq_1 is ready
    ready = get_ready_subquestions(all_sqs, completed_ids=set(), in_flight_ids=set())
    assert [sq.id for sq in ready] == ["sq_1"]

    # When sq_1 completes, sq_2 becomes ready
    ready2 = get_ready_subquestions(all_sqs, completed_ids={"sq_1"}, in_flight_ids=set())
    assert [sq.id for sq in ready2] == ["sq_2"]

    # When sq_1 and sq_2 complete, sq_3 becomes ready
    ready3 = get_ready_subquestions(all_sqs, completed_ids={"sq_1", "sq_2"}, in_flight_ids=set())
    assert [sq.id for sq in ready3] == ["sq_3"]


def test_domain_extraction():
    from app.tools.search import _extract_domain

    assert _extract_domain("https://www.example.com/path?q=1") == "example.com"
    assert _extract_domain("https://sub.docs.redis.io/item") == "sub.docs.redis.io"
    assert _extract_domain("invalid-url") == "unknown"


def test_url_deduplication_and_domain_capping(monkeypatch):
    """Validates that search deduplicates identical URLs and caps domains to max_per_domain."""
    import sys
    from unittest.mock import MagicMock

    import app.tools.search as search_module
    from app.tools.search import search_and_extract

    class MockTavilyClient:
        def __init__(self, api_key):
            pass

        def search(self, **kwargs):
            return {
                "results": [
                    {"url": "https://domainA.com/page1", "title": "A1", "content": "Content A1"},
                    {
                        "url": "https://domainA.com/page1",
                        "title": "A1 Duplicate",
                        "content": "Content A1",
                    },
                    {"url": "https://domainA.com/page2", "title": "A2", "content": "Content A2"},
                    {
                        "url": "https://domainA.com/page3",
                        "title": "A3",
                        "content": "Content A3",
                    },
                    {"url": "https://domainB.com/page1", "title": "B1", "content": "Content B1"},
                ]
            }

    monkeypatch.setattr(search_module.settings, "mock_search", "false")
    monkeypatch.setattr(search_module.settings, "tavily_api_key", "dummy_key")

    mock_tavily = MagicMock()
    mock_tavily.TavilyClient = MockTavilyClient
    monkeypatch.setitem(sys.modules, "tavily", mock_tavily)

    res = search_and_extract(query="test query", max_results=5, max_per_domain=2, fetch_full_page=False)
    findings, status = res[0], res[-1]

    assert status.status == "success"
    assert len(findings) == 3
    urls = [f.source_url for f in findings]
    assert len(urls) == len(set(urls))

    domainA_count = sum(1 for f in findings if f.domain == "domaina.com")
    assert domainA_count <= 2


def test_page_extractor_html_cleaning(monkeypatch):
    """Validates that page extractor strips scripts, styles, and headers and extracts clean text."""
    import httpx

    from app.tools.fetch_extract import extract_page_content

    sample_html = """
    <html>
        <head><style>body { color: red; }</style></head>
        <body>
            <nav><a href="/">Home</a></nav>
            <script>alert("hack");</script>
            <main>
                <h1>System Scalability</h1>
                <p>Distributed key-value checkpoints guarantee state persistence.</p>
            </main>
            <footer>Copyright 2026</footer>
        </body>
    </html>
    """

    class MockResponse:
        status_code = 200
        text = sample_html

    class MockClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, url):
            return MockResponse()

    monkeypatch.setattr(httpx, "Client", MockClient)

    text, error = extract_page_content("https://test.example.com/benchmarks")
    assert error is None
    assert text is not None
    assert "System Scalability" in text
    assert "Distributed key-value checkpoints guarantee state persistence." in text
    assert "alert" not in text
    assert "color: red" not in text
    assert "Copyright" not in text


def test_page_extractor_failure_paths(monkeypatch):
    """Validates structured classification of extractor HTTP failure paths."""
    import httpx

    from app.tools.fetch_extract import extract_page_content

    class FailingClient:
        def __init__(self, status_code=403, raise_timeout=False, **kwargs):
            self.status_code = status_code
            self.raise_timeout = raise_timeout

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, url):
            if self.raise_timeout:
                raise httpx.TimeoutException("Connection timed out")

            class Resp:
                status_code = self.status_code
                text = "Access Denied"

            return Resp()

    # 1. Paywalled / Forbidden (403)
    monkeypatch.setattr(httpx, "Client", lambda **kw: FailingClient(status_code=403))
    text, err = extract_page_content("https://paywalled.com/article")
    assert text is None
    assert "paywalled" in err

    # 2. Rate Limited (429)
    monkeypatch.setattr(httpx, "Client", lambda **kw: FailingClient(status_code=429))
    text, err = extract_page_content("https://ratelimited.com/resource")
    assert text is None
    assert err == "rate_limited"

    # 3. Timeout
    monkeypatch.setattr(httpx, "Client", lambda **kw: FailingClient(raise_timeout=True))
    text, err = extract_page_content("https://slowserver.com/delay")
    assert text is None
    assert err == "timeout"
