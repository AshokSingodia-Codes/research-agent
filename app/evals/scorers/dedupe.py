from typing import Any
from urllib.parse import urlparse

from app.models.schemas import Finding


def score_source_diversity(findings: list[Finding] | list[dict[str, Any]]) -> dict[str, Any]:
    """
    Evaluates source diversity and domain deduplication:
    - unique_domains_count: count of distinct root domains cited
    - total_sources: total findings gathered
    - domain_diversity_ratio: unique domains / total sources
    - max_domain_concentration: highest percentage of sources from a single domain
    - duplicate_content_ratio: fraction of duplicate content hashes
    """
    if not findings:
        return {
            "unique_domains_count": 0,
            "total_sources": 0,
            "domain_diversity_ratio": 1.0,
            "max_domain_concentration": 0.0,
            "duplicate_content_ratio": 0.0,
        }

    domain_counts: dict[str, int] = {}
    content_hashes: set[str] = set()
    duplicate_hashes: int = 0

    for f in findings:
        url = f.get("source_url", "") if isinstance(f, dict) else getattr(f, "source_url", "")
        chash = f.get("content_hash", "") if isinstance(f, dict) else getattr(f, "content_hash", "")

        if url:
            domain = urlparse(url).netloc.lower() or "unknown"
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

        if chash:
            if chash in content_hashes:
                duplicate_hashes += 1
            else:
                content_hashes.add(chash)

    total_sources = len(findings)
    unique_domains = len(domain_counts)
    diversity_ratio = round(unique_domains / total_sources, 4) if total_sources > 0 else 1.0

    max_count = max(domain_counts.values()) if domain_counts else 0
    max_concentration = round(max_count / total_sources, 4) if total_sources > 0 else 0.0
    duplicate_ratio = round(duplicate_hashes / total_sources, 4) if total_sources > 0 else 0.0

    return {
        "unique_domains_count": unique_domains,
        "total_sources": total_sources,
        "domain_diversity_ratio": diversity_ratio,
        "max_domain_concentration": max_concentration,
        "duplicate_content_ratio": duplicate_ratio,
    }
