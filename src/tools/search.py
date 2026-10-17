import os
import re
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

from src.config import settings
from src.models.schemas import EvidenceRecord, SearchStatus
from src.tools.extractor import extract_page_content


def _extract_domain(url: str) -> str:
    """Extracts a clean hostname from URL."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain or "unknown"
    except Exception:
        return "unknown"


def search_and_extract(
    query: str,
    max_results: int = 4,
    max_per_domain: int = 2,
    fetch_full_page: bool = True
) -> Tuple[List[EvidenceRecord], SearchStatus]:
    """
    Executes search with Tavily (or mock), deduplicates by URL, caps results per domain,
    and extracts page content rather than trusting search snippets alone.
    Returns: (list_of_evidence_records, search_status)
    """
    if settings.is_mock_search:
        return _mock_search_results(query, max_results)

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=settings.tavily_api_key)
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results + 3  # request extra to allow domain filtering
        )

        raw_results = response.get("results", [])
        if not raw_results:
            return [], SearchStatus(
                status="no_results",
                query=query,
                result_count=0,
                error_message="Search returned no matching documents"
            )

        evidence_records: List[EvidenceRecord] = []
        domain_counts: Dict[str, int] = {}
        seen_urls = set()

        for item in raw_results:
            url = item.get("url", "")
            if not url or url in seen_urls:
                continue

            domain = _extract_domain(url)
            if domain_counts.get(domain, 0) >= max_per_domain:
                continue

            snippet = item.get("content", "").strip()
            title = item.get("title", "")

            # Extract full page content for deeper evidence provenance
            extracted_text = None
            if fetch_full_page:
                full_text, err = extract_page_content(url, timeout_seconds=6.0)
                if full_text:
                    extracted_text = full_text

            record = EvidenceRecord(
                claim=title or snippet[:150],
                source_url=url,
                domain=domain,
                snippet=snippet or "No snippet available",
                extracted_text=extracted_text,
                confidence=0.9
            )

            seen_urls.add(url)
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            evidence_records.append(record)

            if len(evidence_records) >= max_results:
                break

        return evidence_records, SearchStatus(
            status="success",
            query=query,
            result_count=len(evidence_records)
        )

    except Exception as e:
        err_msg = str(e).lower()
        if "rate limit" in err_msg or "429" in err_msg:
            status = "rate_limited"
        elif "timeout" in err_msg:
            status = "timeout"
        elif "unauthorized" in err_msg or "401" in err_msg or "api key" in err_msg:
            status = "error"
        else:
            status = "error"

        return [], SearchStatus(
            status=status,
            query=query,
            result_count=0,
            error_message=str(e)
        )


def _mock_search_results(query: str, max_results: int) -> Tuple[List[EvidenceRecord], SearchStatus]:
    """Generates deterministic, verified evidence records for testing and offline runs."""
    mock_domains = [
        ("arxiv.org", "https://arxiv.org/abs/2401.0001"),
        ("github.com", "https://github.com/langchain-ai/langgraph"),
        ("redis.io", "https://redis.io/docs/latest/develop/data-types/"),
        ("anthropic.com", "https://docs.anthropic.com/en/docs/build-with-claude/tool-use")
    ]

    records: List[EvidenceRecord] = []
    for i, (domain, url) in enumerate(mock_domains[:max_results]):
        records.append(EvidenceRecord(
            claim=f"Verified finding {i+1} concerning '{query[:60]}': Empirical evaluation demonstrates production reliability under load.",
            source_url=url,
            domain=domain,
            snippet=f"Detailed documentation and benchmarks from {domain} verifying architectural robustness and token efficiency.",
            extracted_text=f"Full extracted context for {query}: Systematic benchmarks show state persistence in distributed key-value stores minimizes recovery latency by 94%.",
            confidence=0.95
        ))

    return records, SearchStatus(
        status="success",
        query=query,
        result_count=len(records)
    )
