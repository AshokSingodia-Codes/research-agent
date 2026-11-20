"""arXiv Atom API search adapter for academic papers."""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET

import httpx

from app.models.schemas import Finding
from app.tools.dedupe import hash_content

ARXIV_API = "https://export.arxiv.org/api/query"
_NS = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
_ID = re.compile(r"abs/(.+?)(?:v\d+)?$")


def _text(node: ET.Element | None) -> str:
    return " ".join((node.text or "").split()) if node is not None else ""


def parse_arxiv_feed(xml_text: str, sub_question_id: str, run_id: str | None = None) -> list[Finding]:
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return []

    findings: list[Finding] = []
    for entry in root.findall("a:entry", _NS):
        m = _ID.search(_text(entry.find("a:id", _NS)))
        title = _text(entry.find("a:title", _NS))
        summary = _text(entry.find("a:summary", _NS))
        if not m or not title:
            continue
        arxiv_id = m.group(1)
        url = f"https://arxiv.org/abs/{arxiv_id}"
        published = _text(entry.find("a:published", _NS))[:4]
        claim = f"{title} ({published}): {summary[:200]}"
        content_hash = hash_content(summary or title)

        findings.append(
            Finding(
                run_id=run_id,
                sub_question_id=sub_question_id,
                claim=claim,
                source_url=url,
                domain="arxiv.org",
                snippet=summary[:500],
                extracted_text=f"Title: {title}\nPublished: {published}\nAbstract: {summary}",
                retrieval_timestamp=time.time(),
                content_hash=content_hash,
                confidence=0.92,
                source_quality_score=0.95,
            )
        )
    return findings


def search_arxiv(
    query: str,
    sub_question_id: str = "sq_default",
    max_results: int = 4,
    run_id: str | None = None,
    client: httpx.Client | None = None,
) -> list[Finding]:
    """Searches arXiv Atom API for technical and academic papers."""
    clean_query = " ".join([w for w in re.sub(r'["\']', '', query).split() if len(w) > 2][:6])
    params = {
        "search_query": f"all:{clean_query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
    }
    should_close = False
    if client is None:
        client = httpx.Client(timeout=8.0, headers={"User-Agent": "multiagent-research/1.0"})
        should_close = True

    try:
        resp = client.get(ARXIV_API, params=params)
        if resp.status_code == 200:
            return parse_arxiv_feed(resp.text, sub_question_id=sub_question_id, run_id=run_id)
        return []
    except Exception:
        return []
    finally:
        if should_close:
            client.close()
