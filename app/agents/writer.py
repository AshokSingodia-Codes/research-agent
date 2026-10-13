from typing import Any

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
1. Every major factual claim in the report sections MUST be backed by an inline citation key, formatted as [1], [2], etc.
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
    )

    report.topic = topic
    report.partial = partial
    report.degradation_reason = degradation_reason
    if conflicting:
        report.conflicting_evidence = conflicting

    # Ensure citations map has valid entries matching findings if model omitted any
    if not report.citations and findings:
        for idx, f in enumerate(findings[:10], 1):
            key = f"[{idx}]"
            report.citations[key] = CitationKey(
                key=key,
                finding_id=f.id,
                source_url=f.source_url,
                domain=f.domain,
                claim=f.claim,
            )

    return report, trace
