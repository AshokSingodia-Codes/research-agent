"""OpenAlex Works API search adapter for academic and open access papers."""

from __future__ import annotations

import re
import time
from typing import Any

import httpx

from app.models.schemas import Finding
from app.tools.dedupe import hash_content

OPENALEX_WORKS = "https://api.openalex.org/works"


def reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    if not inverted_index:
        return ""
    indexed: list[tuple[int, str]] = []
    for token, positions in inverted_index.items():
        if isinstance(token, str) and isinstance(positions, list):
            for pos in positions:
                if isinstance(pos, int):
                    indexed.append((pos, token))
    indexed.sort(key=lambda p: p[0])
    return " ".join(t for _, t in indexed).strip()


def parse_openalex_work(
    r: dict[str, Any],
    sub_question_id: str = "sq_default",
    run_id: str | None = None,
) -> Finding | None:
    """Parses one raw OpenAlex work dictionary into a Finding."""
    title = r.get("title") or ""
    if not title:
        return None
    year = r.get("publication_year") or ""
    abstract = reconstruct_abstract(r.get("abstract_inverted_index"))
    doi = r.get("doi") or ""
    url = (
        doi
        or (r.get("primary_location") or {}).get("landing_page_url")
        or r.get("id")
        or "https://openalex.org"
    )
    claim = f"{title} ({year}): {abstract[:200]}" if abstract else f"{title} ({year})"
    content_hash = hash_content(abstract or title)

    return Finding(
        run_id=run_id,
        sub_question_id=sub_question_id,
        claim=claim,
        source_url=url,
        domain="openalex.org",
        snippet=abstract[:500] if abstract else title,
        extracted_text=f"Title: {title}\nYear: {year}\nAbstract: {abstract}",
        retrieval_timestamp=time.time(),
        content_hash=content_hash,
        confidence=0.90,
        source_quality_score=0.92,
    )


def search_openalex(
    query: str,
    sub_question_id: str = "sq_default",
    max_results: int = 4,
    run_id: str | None = None,
    client: httpx.Client | None = None,
    api_key: str | None = None,
    mailto: str | None = None,
) -> list[Finding]:
    """Searches OpenAlex Works API and converts works into Findings."""
    clean_q = " ".join([w for w in re.sub(r'["\']', '', query).split() if len(w) > 2][:6])
    params: dict[str, Any] = {
        "search": clean_q,
        "per_page": max_results,
        "select": "id,doi,title,publication_year,abstract_inverted_index,primary_location,cited_by_count",
    }
    if api_key:
        params["api_key"] = api_key
    if mailto:
        params["mailto"] = mailto

    should_close = False
    if client is None:
        client = httpx.Client(timeout=8.0, headers={"User-Agent": "multiagent-research/1.0"})
        should_close = True

    try:
        resp = client.get(OPENALEX_WORKS, params=params)
        if resp.status_code != 200:
            return []
        data = resp.json()
        results = data.get("results") or []
        findings: list[Finding] = []
        for r in results:
            finding = parse_openalex_work(r, sub_question_id=sub_question_id, run_id=run_id)
            if finding:
                findings.append(finding)
        return findings
    except Exception:
        return []
    finally:
        if should_close:
            client.close()
