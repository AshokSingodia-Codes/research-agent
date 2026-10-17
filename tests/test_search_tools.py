import httpx
import pytest

from src.tools.extractor import extract_page_content
from src.tools.search import _extract_domain, search_and_extract


@pytest.mark.unit
def test_domain_extraction():
    assert _extract_domain("https://www.example.com/path?q=1") == "example.com"
    assert _extract_domain("https://sub.docs.redis.io/item") == "sub.docs.redis.io"
    assert _extract_domain("invalid-url") == "unknown"


@pytest.mark.unit
def test_url_deduplication_and_domain_capping(monkeypatch):
    """Validates that search deduplicates identical URLs and caps domains to max_per_domain."""
    import src.tools.search as search_module

    # Mock raw search response with duplicate URLs and 4 items from the same domain
    class MockTavilyClient:
        def __init__(self, api_key):
            pass

        def search(self, **kwargs):
            return {
                "results": [
                    {"url": "https://domainA.com/page1", "title": "A1", "content": "Content A1"},
                    {"url": "https://domainA.com/page1", "title": "A1 Duplicate", "content": "Content A1"},  # duplicate URL
                    {"url": "https://domainA.com/page2", "title": "A2", "content": "Content A2"},
                    {"url": "https://domainA.com/page3", "title": "A3", "content": "Content A3"},  # exceeds domain cap (max 2)
                    {"url": "https://domainB.com/page1", "title": "B1", "content": "Content B1"},
                ]
            }

    # Temporarily set live mode mock tavily client
    monkeypatch.setattr(search_module.settings, "mock_search", "false")
    monkeypatch.setattr(search_module.settings, "tavily_api_key", "dummy_key")

    import sys
    from unittest.mock import MagicMock
    mock_tavily = MagicMock()
    mock_tavily.TavilyClient = MockTavilyClient
    monkeypatch.setitem(sys.modules, "tavily", mock_tavily)

    # Disable network extraction for pure deduplication test
    findings, status = search_and_extract(
        query="test query",
        max_results=5,
        max_per_domain=2,
        fetch_full_page=False
    )

    assert status.status == "success"
    # Should only have 3: 2 from domainA, 1 from domainB
    assert len(findings) == 3
    urls = [f.source_url for f in findings]
    assert len(urls) == len(set(urls))  # No duplicates

    domainA_count = sum(1 for f in findings if f.domain == "domaina.com")
    assert domainA_count <= 2


@pytest.mark.unit
def test_page_extractor_html_cleaning(monkeypatch):
    """Validates that page extractor strips scripts, styles, and headers and extracts clean text."""
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
    assert "alert" not in text  # script removed
    assert "color: red" not in text  # style removed
    assert "Copyright" not in text  # footer removed


@pytest.mark.unit
def test_page_extractor_failure_paths(monkeypatch):
    """Validates structured classification of extractor HTTP failure paths."""
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
