"""Semantic Scholar Graph API search adapter for academic papers."""

from __future__ import annotations

import logging
import re
import time
from typing import Any

import httpx

from app.config import settings
from app.models.schemas import Finding
from app.tools.dedupe import hash_content

logger = logging.getLogger(__name__)

S2_SEARCH_API = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = "title,abstract,year,authors,externalIds,citationCount,venue,openAccessPdf,url"


def parse_semantic_scholar_item(
    item: dict[str, Any],
    sub_question_id: str,
    run_id: str | None = None,
) -> Finding | None:
    """Parses a single Semantic Scholar search hit into a normalized Finding."""
    title = (item.get("title") or "").strip()
    if not title:
        return None

    ext = item.get("externalIds") or {}
    arxiv_id = ext.get("ArXiv")
    doi = ext.get("DOI")
    abstract = (item.get("abstract") or "").strip()
    year = item.get("year")
    cites = int(item.get("citationCount") or 0)
    venue = item.get("venue") or ""

    url = (
        item.get("url")
        or (f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None)
        or (f"https://doi.org/{doi}" if doi else f"https://www.semanticscholar.org/paper/{item.get('paperId')}")
    )

    authors = [a.get("name", "") for a in item.get("authors") or [] if a.get("name")]
    author_summary = f"{authors[0]} et al." if len(authors) > 1 else (authors[0] if authors else "Unknown")

    claim = f"{title} ({author_summary}, {year or 'n.d.'}): {abstract[:220]}"
    content_hash = hash_content(abstract or title)

    full_text = f"Title: {title}\nAuthors: {', '.join(authors[:5])}\nYear: {year}\nVenue: {venue}\nCitations: {cites}\nAbstract: {abstract}"

    return Finding(
        run_id=run_id,
        sub_question_id=sub_question_id,
        claim=claim,
        source_url=url,
        domain="semanticscholar.org",
        snippet=(abstract[:500] if abstract else title),
        extracted_text=full_text,
        retrieval_timestamp=time.time(),
        content_hash=content_hash,
        confidence=0.92,
        source_quality_score=min(1.0, 0.80 + (0.15 if cites > 50 else (0.08 if cites > 5 else 0.0))),
    )


def search_semantic_scholar(
    query: str,
    sub_question_id: str = "sq_default",
    max_results: int = 4,
    run_id: str | None = None,
    client: httpx.Client | None = None,
    api_key: str | None = None,
) -> list[Finding]:
    """Queries the Semantic Scholar Graph API for relevant academic literature."""
    clean_query = " ".join([w for w in re.sub(r'["\']', "", query).split() if len(w) > 2][:6])
    if not clean_query:
        return []

    params: dict[str, Any] = {
        "query": clean_query,
        "limit": min(max_results, 10),
        "fields": S2_FIELDS,
    }

    key = api_key or getattr(settings, "semantic_scholar_api_key", None)
    headers = {"x-api-key": key} if key else {}

    try:
        if client:
            resp = client.get(S2_SEARCH_API, params=params, headers=headers, timeout=12.0)
            if resp.status_code == 200:
                data = resp.json()
            else:
                logger.debug(f"Semantic Scholar responded with status {resp.status_code}")
                return []
        else:
            with httpx.Client(timeout=12.0) as local_client:
                resp = local_client.get(S2_SEARCH_API, params=params, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                else:
                    return []

        raw_items = data.get("data") or []
        findings: list[Finding] = []
        for item in raw_items:
            f = parse_semantic_scholar_item(item, sub_question_id=sub_question_id, run_id=run_id)
            if f:
                findings.append(f)
        return findings
    except Exception as e:
        logger.warning(f"Semantic Scholar search failed: {e}")
        return []

