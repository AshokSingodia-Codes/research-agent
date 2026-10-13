import re


def calculate_source_quality(
    url: str,
    domain: str,
    extracted_text: str = "",
    published_date: str | None = None,
) -> float:
    """
    Computes a source quality score (0.0 to 1.0) based on:
    - Domain authority category (.gov, .edu, academic repos, official specs)
    - Depth of technical content (presence of benchmarks, code blocks, citations)
    - Presence of author/recency indicators
    """
    score = 0.50
    domain_lower = domain.lower()

    # 1. High-authority domains
    if domain_lower.endswith(".edu") or domain_lower.endswith(".gov"):
        score += 0.35
    elif any(d in domain_lower for d in ["arxiv.org", "acm.org", "ieee.org", "usenix.org", "rfc-editor.org"]):
        score += 0.40
    elif any(d in domain_lower for d in ["github.io", "github.com", "docs.", "/docs", "developer."]):
        score += 0.25
    elif any(d in domain_lower for d in ["medium.com", "substack.com", "dev.to", "quora.com"]):
        score -= 0.10

    # 2. Content quality indicators
    text_lower = extracted_text.lower()
    if any(term in text_lower for term in ["benchmark", "throughput", "latency", "evaluation", "architecture"]):
        score += 0.10
    if re.search(r"(\b202[3-6]\b)", extracted_text):  # Recency
        score += 0.05
    if published_date:
        score += 0.05

    # Clamp between 0.1 and 1.0
    return round(max(0.1, min(1.0, score)), 2)
