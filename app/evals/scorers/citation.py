import re
from typing import Any

from app.models.schemas import Finding, Report


def score_citations(report: Report | dict[str, Any], verified_findings: list[Finding] | list[dict[str, Any]]) -> dict[str, Any]:
    """
    Evaluates report citation provenance:
    - citation_validity_ratio: fraction of inline citations that point to an existing finding
    - citation_density: count of citations per 100 words of section text
    - hallucinated_citations: count of citations with no source backing
    - total_citations_referenced: count of distinct citation keys found in text
    """
    if isinstance(report, dict):
        sections = report.get("sections", [])
        citations_dict = report.get("citations", {})
    else:
        sections = [s.model_dump() if hasattr(s, "model_dump") else s for s in report.sections]
        citations_dict = {
            k: (v.model_dump() if hasattr(v, "model_dump") else v)
            for k, v in report.citations.items()
        }

    finding_ids = set()
    finding_urls = set()
    for f in verified_findings:
        if isinstance(f, dict):
            finding_ids.add(f.get("id"))
            finding_urls.add(f.get("source_url"))
        else:
            finding_ids.add(getattr(f, "id", None))
            finding_urls.add(getattr(f, "source_url", None))

    finding_ids.discard(None)
    finding_urls.discard(None)

    total_words = 0
    referenced_keys = set()
    for sec in sections:
        content = sec.get("content", "") if isinstance(sec, dict) else getattr(sec, "content", "")
        words = len(content.split())
        total_words += words
        keys = re.findall(r"\[\d+\]", content)
        referenced_keys.update(keys)

    valid_keys = 0
    hallucinated_keys = 0

    for key in referenced_keys:
        info = citations_dict.get(key)
        if not info:
            hallucinated_keys += 1
            continue

        fid = info.get("finding_id") if isinstance(info, dict) else getattr(info, "finding_id", None)
        url = info.get("source_url") if isinstance(info, dict) else getattr(info, "source_url", None)

        if (fid and fid in finding_ids) or (url and url in finding_urls):
            valid_keys += 1
        else:
            hallucinated_keys += 1

    validity_ratio = (valid_keys / len(referenced_keys)) if referenced_keys else (1.0 if not verified_findings else 0.0)
    density = (len(referenced_keys) / (total_words / 100.0)) if total_words > 0 else 0.0

    return {
        "citation_validity_ratio": round(validity_ratio, 4),
        "citation_density_per_100_words": round(density, 2),
        "total_citations_referenced": len(referenced_keys),
        "valid_citations": valid_keys,
        "hallucinated_citations": hallucinated_keys,
    }
