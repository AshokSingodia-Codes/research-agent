import re

from app.models.schemas import Finding, Report


def validate_citations(
    report: Report,
    verified_findings: list[Finding],
    threshold: float = 0.80,
) -> tuple[Report, float, list[str]]:
    """
    Validates report citation provenance:
    1. Ensures every referenced citation key corresponds to a real, verified Finding record.
    2. Detects fabricated citation keys and ungrounded claims.
    3. Calculates citation_coverage = valid_citations / total_referenced_citations.
    4. Flags fabricated citations.
    """
    valid_finding_ids = {f.id for f in verified_findings}
    valid_urls = {f.source_url for f in verified_findings}

    warnings: list[str] = []
    referenced_keys = set()

    # Extract all inline citation keys [n] across all sections
    for sec in report.sections:
        found_keys = re.findall(r"\[\d+\]", sec.content)
        referenced_keys.update(found_keys)
        sec.citations = sorted(list(set(sec.citations + found_keys)))

    if not referenced_keys:
        report.citation_coverage = 1.0 if not verified_findings else 0.0
        return report, report.citation_coverage, ["No inline citations referenced in report text."]

    valid_keys_count = 0
    clean_citations = {}

    for key in referenced_keys:
        citation_info = report.citations.get(key)
        if not citation_info:
            warnings.append(f"Fabricated citation key {key}: Missing citation record in citations dictionary.")
            continue

        # Check if finding_id exists in verified evidence
        has_valid_id = citation_info.finding_id in valid_finding_ids
        has_valid_url = citation_info.source_url in valid_urls

        if has_valid_id or has_valid_url:
            valid_keys_count += 1
            clean_citations[key] = citation_info
        else:
            warnings.append(
                f"Fabricated citation {key}: finding_id '{citation_info.finding_id}' "
                f"or URL '{citation_info.source_url}' does not match any retrieved Finding."
            )

    report.citations = clean_citations
    coverage = round(valid_keys_count / max(len(referenced_keys), 1), 2)
    report.citation_coverage = coverage

    return report, coverage, warnings
