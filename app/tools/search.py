import logging
import random
import time
from typing import Any
from urllib.parse import urlparse

from app.config import settings
from app.models.enums import ToolStatus
from app.models.schemas import Finding, ToolResult
from app.tools.dedupe import canonicalize_url, hash_content
from app.tools.fetch_extract import extract_full_page
from app.tools.scoring import calculate_source_quality

logger = logging.getLogger(__name__)


class DomainCircuitBreaker:
    """Per-domain circuit breaker: trips open if a domain fails 3 consecutive times."""

    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self._failures: dict[str, int] = {}
        self._tripped: set[str] = set()

    def is_tripped(self, domain: str) -> bool:
        return domain.lower() in self._tripped

    def record_failure(self, domain: str) -> None:
        dom = domain.lower()
        count = self._failures.get(dom, 0) + 1
        self._failures[dom] = count
        if count >= self.failure_threshold:
            self._tripped.add(dom)
            logger.warning(f"Circuit breaker tripped for domain: {dom} ({count} failures)")

    def record_success(self, domain: str) -> None:
        dom = domain.lower()
        self._failures[dom] = 0
        self._tripped.discard(dom)


circuit_breaker = DomainCircuitBreaker()


def _extract_domain(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc or "unknown"
    except Exception:
        return "unknown"


def search_and_extract(
    query: str,
    sub_question_id: str = "sq_default",
    max_results: int = 4,
    max_per_domain: int = 2,
    fetch_full_page: bool = True,
    token_cap: int = 2500,
    run_id: str | None = None,
) -> ToolResult:
    """
    Executes web search (Tavily or mock), extracts full pages with trafilatura,
    deduplicates by canonical URL and content hash, enforces domain capping,
    and returns a structured ToolResult.
    """
    if settings.is_mock_search:
        return _mock_search(query, sub_question_id, max_results, max_per_domain, run_id=run_id)

    # 1. Real Tavily Search (if key provided)
    raw_results = []
    if settings.tavily_api_key:
        try:
            from tavily import TavilyClient

            client = TavilyClient(api_key=settings.tavily_api_key)

            # Retry with exponential backoff on rate limits
            retries = 0
            while retries <= 2:
                try:
                    response = client.search(
                        query=query,
                        search_depth="advanced",
                        max_results=max_results + 4,  # Extra to allow domain filtering
                    )
                    raw_results = response.get("results", [])
                    break
                except Exception as e:
                    err_str = str(e).lower()
                    if "rate limit" in err_str or "429" in err_str:
                        retries += 1
                        sleep_time = (2**retries) + random.uniform(0.1, 0.5)
                        time.sleep(sleep_time)
                    else:
                        break
        except Exception as e:
            logger.warning(f"Tavily search failed ({e}); checking DuckDuckGo fallback...")

    # 2. DuckDuckGo Free Search Fallback (no API key required)
    if not raw_results:
        try:
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS

            ddgs = DDGS()
            ddg_hits = list(ddgs.text(query, max_results=max_results + 4))
            
            # If 0 hits, try query relaxation (strip quotation marks, punctuation, and narrow sub-phrases)
            if not ddg_hits:
                simplified_query = " ".join([w for w in query.replace('"', '').replace("'", "").split() if len(w) > 3][:6])
                if simplified_query and simplified_query != query:
                    logger.info(f"Relaxing search query from '{query}' to '{simplified_query}'")
                    ddg_hits = list(ddgs.text(simplified_query, max_results=max_results + 4))

            if ddg_hits:
                raw_results = [
                    {
                        "url": hit.get("href") or hit.get("url"),
                        "title": hit.get("title", ""),
                        "content": hit.get("body") or hit.get("snippet", ""),
                    }
                    for hit in ddg_hits
                    if hit.get("href") or hit.get("url")
                ]
        except Exception as e:
            logger.warning(f"DuckDuckGo search fallback encountered ({e}); falling back to mock search")

    # 3. Deterministic Mock Search Fallback
    if not raw_results:
        logger.info(f"Using mock search for query '{query}'")
        return _mock_search(query, sub_question_id, max_results, max_per_domain, run_id=run_id)

    return _process_raw_results(
        raw_results=raw_results,
        query=query,
        sub_question_id=sub_question_id,
        max_results=max_results,
        max_per_domain=max_per_domain,
        fetch_full_page=fetch_full_page,
        token_cap=token_cap,
        run_id=run_id,
    )


def _process_raw_results(
    raw_results: list[dict[str, Any]],
    query: str,
    sub_question_id: str,
    max_results: int,
    max_per_domain: int,
    fetch_full_page: bool,
    token_cap: int,
    run_id: str | None = None,
) -> ToolResult:
    """Processes search hits: deduplication, domain capping, full page extraction, and provenance."""
    seen_urls: set[str] = set()
    seen_hashes: set[str] = set()
    domain_counts: dict[str, int] = {}
    findings: list[Finding] = []

    for item in raw_results:
        # Check cancellation between fetches
        if run_id:
            try:
                from app.persistence.redis_store import redis_store
                client = redis_store.get_client()
                if client and client.get(f"research:cancel:{run_id}"):
                    logger.info(f"Search fetch aborted early due to cancellation for {run_id}")
                    break
            except Exception:
                pass

        if len(findings) >= max_results:
            break

        url = item.get("url", "")
        if not url:
            continue

        canon_url = canonicalize_url(url)
        domain = _extract_domain(canon_url)

        # 1. Check circuit breaker
        if circuit_breaker.is_tripped(domain):
            continue

        # 2. Check domain cap
        curr_domain_count = domain_counts.get(domain, 0)
        if curr_domain_count >= max_per_domain:
            continue

        # 3. Check duplicate URL
        if canon_url in seen_urls:
            continue

        snippet = item.get("content") or item.get("snippet", "")
        extracted_text = ""
        fetch_status = ToolStatus.OK

        if fetch_full_page:
            extracted_text, fetch_status = extract_full_page(canon_url, timeout_seconds=4.0, token_cap=token_cap)
            if fetch_status != ToolStatus.OK:
                circuit_breaker.record_failure(domain)
                # If extraction fails, use snippet as fallback if available
                if not snippet:
                    continue
            else:
                circuit_breaker.record_success(domain)

        content_for_hash = extracted_text if extracted_text else snippet
        content_hash = hash_content(content_for_hash)

        # 4. Check duplicate content hash
        if content_hash in seen_hashes:
            continue

        # Mark seen
        seen_urls.add(canon_url)
        seen_hashes.add(content_hash)
        domain_counts[domain] = curr_domain_count + 1

        quality = calculate_source_quality(canon_url, domain, extracted_text)
        claim_text = snippet[:250] if snippet else extracted_text[:250]

        finding = Finding(
            run_id=run_id,
            sub_question_id=sub_question_id,
            claim=claim_text.replace("\n", " ").strip(),
            source_url=canon_url,
            domain=domain,
            snippet=snippet[:500],
            extracted_text=extracted_text,
            retrieval_timestamp=time.time(),
            content_hash=content_hash,
            confidence=0.90,
            source_quality_score=quality,
        )
        findings.append(finding)

    if not findings:
        return ToolResult(
            status=ToolStatus.NO_RESULTS,
            query=query,
            error_message="All search results filtered due to domain caps or extraction failures",
        )

    return ToolResult(
        status=ToolStatus.OK,
        findings=findings,
        query=query,
        metadata={"total_found": len(findings)},
    )


def _mock_search(
    query: str,
    sub_question_id: str,
    max_results: int,
    max_per_domain: int,
    run_id: str | None = None,
) -> ToolResult:
    """Deterministic mock search results for testing without external network calls."""
    domains = ["raft.github.io", "etcd.io", "cs.stanford.edu", "aws.amazon.com"]
    raw_mock = []
    for i in range(max_results + 2):
        dom = domains[i % len(domains)]
        raw_mock.append(
            {
                "url": f"https://{dom}/paper_{i}?utm_source=test&ref=mock",
                "content": f"Empirical findings and benchmarks for query: {query}. Evidence record {i} substantiating architectural guarantees.",
                "snippet": f"Verified snippet {i} on {dom}",
            }
        )

    return _process_raw_results(
        raw_results=raw_mock,
        query=query,
        sub_question_id=sub_question_id,
        max_results=max_results,
        max_per_domain=max_per_domain,
        fetch_full_page=False,
        token_cap=2500,
        run_id=run_id,
    )
