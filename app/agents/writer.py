from typing import Any

from app.config import settings
from app.llm.client import llm_client
from app.models.schemas import (
    CitationKey,
    Contradiction,
    Finding,
    Report,
    StepTrace,
)

WRITER_SYSTEM_PROMPT = """Synthesize research findings into an authoritative, structured report with inline citation keys.

Requirements:
1. Every major factual claim inside each section's content text MUST explicitly include inline citation keys, formatted as [1], [2], etc. (e.g., 'Recent benchmarks show significant gains [1].').
2. The citations map MUST correlate each citation key [n] to the exact finding_id, source_url, domain, and claim.
3. Structure the report logically with a concise executive summary and detailed topical sections.
4. If conflicting evidence records are provided, you MUST dedicate a section titled 'Conflicting Evidence' discussing the trade-offs, discrepancies, or disputed points across sources.
5. Do NOT invent sources or citations not present in the provided findings.
"""


def write_report(
    topic: str,
    findings: list[Finding],
    conflicting_evidence: list[Contradiction] | None = None,
    partial: bool = False,
    degradation_reason: str | None = None,
    run_id: str = "run_default",
    parent_step_id: str | None = None,
    budget_guard: Any | None = None,
) -> tuple[Report, StepTrace]:
    """
    Writer specialist: Synthesizes evidence findings into a structured, cited report.
    """
    conflicting = conflicting_evidence or []

    # Build reference findings text
    findings_context = ""
    for idx, f in enumerate(findings, 1):
        findings_context += (
            f"Finding ID: {f.id}\n"
            f"Sub-Question ID: {f.sub_question_id}\n"
            f"Claim: {f.claim}\n"
            f"Source URL: {f.source_url}\n"
            f"Domain: {f.domain}\n"
            f"Snippet: {f.snippet}\n\n"
        )

    conflicts_context = ""
    if conflicting:
        conflicts_context = "Identified Contradictions Across Sources:\n"
        for c in conflicting:
            conflicts_context += (
                f"- Conflict ID: {c.id}\n"
                f"  Claim A: {c.claim_a}\n"
                f"  Claim B: {c.claim_b}\n"
                f"  Sources: {', '.join(c.sources)}\n"
                f"  Severity: {c.severity}\n"
                f"  Explanation: {c.explanation}\n\n"
            )

    user_prompt = (
        f"Research Topic: {topic}\n\n"
        f"Verified Findings:\n{findings_context or 'No findings available.'}\n\n"
        f"{conflicts_context}\n"
        f"Status: {'Partial Report (degraded due to: ' + str(degradation_reason) + ')' if partial else 'Complete Research Run'}\n\n"
        f"Synthesize the final report adhering strictly to the Report schema."
    )

    messages = [
        {"role": "system", "content": WRITER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    report, trace = llm_client.generate_with_repair(
        agent_name="writer",
        messages=messages,
        output_schema=Report,
        run_id=run_id,
        parent_step_id=parent_step_id,
        budget_guard=budget_guard,
        model=settings.writer_model,
    )

    report.topic = topic
    report.partial = partial
    report.degradation_reason = degradation_reason
    if conflicting:
        report.conflicting_evidence = conflicting

    # Always ground and align citation keys with verified findings
    if findings:
        finding_by_url = {f.source_url: f for f in findings}
        finding_by_id = {f.id: f for f in findings}
        # If citations map is sparse or missing keys used in text, populate sequentially from findings
        for idx, f in enumerate(findings, 1):
            key = f"[{idx}]"
            if key not in report.citations:
                report.citations[key] = CitationKey(
                    key=key,
                    finding_id=f.id,
                    source_url=f.source_url,
                    domain=f.domain,
                    claim=f.claim,
                )
            else:
                c_item = report.citations[key]
                # If finding_id doesn't match, map to closest matching URL or the indexed finding
                if c_item.source_url in finding_by_url:
                    matched = finding_by_url[c_item.source_url]
                    c_item.finding_id = matched.id
                    c_item.domain = matched.domain
                    c_item.claim = matched.claim
                elif c_item.finding_id not in finding_by_id:
                    c_item.finding_id = f.id
                    c_item.source_url = f.source_url
                    c_item.domain = f.domain
                    c_item.claim = f.claim

    return report, trace
