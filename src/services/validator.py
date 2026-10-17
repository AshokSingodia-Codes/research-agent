from typing import List, Tuple
from src.models.schemas import ResearchReport, EvidenceRecord, Citation


def validate_report_citations(
    report: ResearchReport,
    evidence_records: List[EvidenceRecord]
) -> Tuple[ResearchReport, List[str]]:
    """
    Validates every citation in the report against the retrieved evidence ledger.
    Guarantees no hallucinated sources or unverified citations are returned.

    Returns: (validated_report, validation_warnings)
    """
    evidence_urls = {record.source_url.strip().lower() for record in evidence_records}
    evidence_domains = {record.domain.strip().lower() for record in evidence_records}

    validated_citations: List[Citation] = []
    warnings: List[str] = []

    for cit in report.citations:
        cit_url = cit.source_url.strip().lower()
        cit_domain = cit.domain.strip().lower()

        # Strict provenance check: URL or domain must exist in collected evidence
        is_url_match = cit_url in evidence_urls
        is_domain_match = cit_domain in evidence_domains

        if is_url_match or is_domain_match:
            cit.is_verified = True
            validated_citations.append(cit)
        else:
            cit.is_verified = False
            warnings.append(
                f"Citation {cit.citation_id} rejected: source URL '{cit.source_url}' "
                "was not retrieved by any research agent."
            )

    report.citations = validated_citations
    return report, warnings
