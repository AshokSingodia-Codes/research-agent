import re
from pathlib import Path
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

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[\"'(])")
_CITE_PATTERN = re.compile(r"\[(?:cite:)?(C?\d+)\]")


def _load_writer_prompt() -> str:
    prompt_file = Path(__file__).parent.parent / "prompts" / "writer.md"
    if prompt_file.exists():
        try:
            return prompt_file.read_text(encoding="utf-8").strip()
        except Exception:
            pass
    return """Synthesize research findings into an authoritative, structured report with inline citation keys.

Requirements:
1. Every major factual claim inside each section's content text MUST explicitly include inline citation keys, formatted as [1], [2], or [C1], [C2].
2. The citations map MUST correlate each citation key to the exact finding_id, source_url, domain, and claim.
3. Structure the report logically with a concise executive summary and detailed topical sections.
4. If conflicting evidence records are provided, you MUST dedicate a section titled 'Conflicting Evidence' discussing discrepancies across sources.
5. Do NOT invent sources or citations not present in the provided findings.
"""


WRITER_SYSTEM_PROMPT = _load_writer_prompt()


def _clean_and_resolve_section(
    content: str,
    alias_to_finding: dict[str, Finding],
    cited_findings: dict[str, Finding],
) -> tuple[str, list[str]]:
    """
    Resolves citation aliases [C#] or [#] to [1]..[N], strips phantom/hallucinated
    citations, and prunes sentences whose citations were entirely hallucinated.
    """
    clean_lines = []
    used_keys = set()

    for line in content.splitlines():
        trimmed = line.strip()
        if not trimmed or trimmed.startswith(("#", "|")):
            clean_lines.append(line)
            continue

        kept_sentences = []
        for sent in _SENT.split(trimmed):
            tags = _CITE_PATTERN.findall(sent)
            if not tags:
                kept_sentences.append(sent)
                continue

            valid_in_sent = []
            has_invalid = False
            for tag in tags:
                raw_tag = tag.upper()
                norm_tag = raw_tag if raw_tag.startswith("C") else f"C{raw_tag}"
                alt_tag = raw_tag.lstrip("C")
                f = alias_to_finding.get(norm_tag) or alias_to_finding.get(alt_tag)
                if f:
                    valid_in_sent.append((tag, f))
                    key = f"[{len(cited_findings) + 1}]"
                    for k, existing_f in cited_findings.items():
                        if existing_f.id == f.id:
                            key = k
                            break
                    if key not in cited_findings:
                        cited_findings[key] = f
                    used_keys.add(key)
                else:
                    has_invalid = True

            # If all citations in the sentence were fabricated, drop the sentence
            if not valid_in_sent and has_invalid:
                continue

            cleaned_sent = sent
            # First strip invalid tags
            for tag in tags:
                if not any(tag == vt for vt, _ in valid_in_sent):
                    cleaned_sent = re.sub(rf"\[(?:cite:)?{re.escape(tag)}\]", "", cleaned_sent)

            # Replace valid aliases with canonical [n] keys
            for orig_tag, f in valid_in_sent:
                assigned_key = next((k for k, ef in cited_findings.items() if ef.id == f.id), "[1]")
                cleaned_sent = re.sub(rf"\[(?:cite:)?{re.escape(orig_tag)}\]", assigned_key, cleaned_sent)

            cleaned_sent = re.sub(r"\s+([.,;:!?])", r"\1", cleaned_sent)
            kept_sentences.append(cleaned_sent)

        if kept_sentences:
            clean_lines.append(" ".join(kept_sentences).rstrip())

    return "\n".join(clean_lines).strip(), sorted(list(used_keys))


def write_report(
    topic: str,
    findings: list[Finding],
    conflicting_evidence: list[Contradiction] | None = None,
    partial: bool = False,
    degradation_reason: str | None = None,
    previous_draft: Report | None = None,
    flagged_claims: list[str] | None = None,
    missing_topics: list[str] | None = None,
    run_id: str = "run_default",
    parent_step_id: str | None = None,
    budget_guard: Any | None = None,
) -> tuple[Report, StepTrace]:
    """
    Writer specialist: Synthesizes evidence findings into a structured, cited report
    with rigorous citation alias mapping and hallucination stripping.
    """
    conflicting = conflicting_evidence or []

    # Map findings to [C1]..[Cn] aliases
    alias_to_finding: dict[str, Finding] = {}
    findings_context = ""
    for idx, f in enumerate(findings, 1):
        alias = f"C{idx}"
        alias_to_finding[alias] = f
        alias_to_finding[str(idx)] = f
        findings_context += (
            f"[{alias}] Source: {f.domain} ({f.source_url})\n"
            f"Claim: {f.claim}\n"
            f"Excerpt: {f.snippet}\n\n"
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

    revision_context = ""
    if previous_draft:
        revision_context = f"\nPrevious Draft Summary: {previous_draft.summary}\n"
        if flagged_claims:
            revision_context += "CRITICAL REVIEWER FEEDBACK - Fix or remove these flagged unsupported claims:\n"
            for fc in flagged_claims:
                revision_context += f"- {fc}\n"
        if missing_topics:
            revision_context += "Missing Topic Coverage to address:\n"
            for mt in missing_topics:
                revision_context += f"- {mt}\n"

    user_prompt = (
        f"Research Topic: {topic}\n\n"
        f"Verified Evidence Passages:\n{findings_context or 'No findings available.'}\n\n"
        f"{conflicts_context}\n"
        f"{revision_context}\n"
        f"Status: {'Partial Report (degraded due to: ' + str(degradation_reason) + ')' if partial else 'Complete Research Run'}\n\n"
        f"Synthesize the final report adhering strictly to the Report schema. "
        f"Use ONLY the provided citation labels ([C1], [C2], etc.)."
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

    # Grounding & citation resolution
    cited_findings: dict[str, Finding] = {}
    if findings:
        for sec in report.sections:
            cleaned_text, sec_keys = _clean_and_resolve_section(sec.content, alias_to_finding, cited_findings)
            sec.content = cleaned_text
            sec.citations = sec_keys

        # Ensure all referenced cited_findings are recorded in report.citations
        new_citations: dict[str, CitationKey] = {}
        for key, f in cited_findings.items():
            new_citations[key] = CitationKey(
                key=key,
                finding_id=f.id,
                source_url=f.source_url,
                domain=f.domain,
                claim=f.claim,
            )

        # Fallback: if report had no inline citations, sequentially map findings
        if not new_citations:
            for idx, f in enumerate(findings, 1):
                key = f"[{idx}]"
                new_citations[key] = CitationKey(
                    key=key,
                    finding_id=f.id,
                    source_url=f.source_url,
                    domain=f.domain,
                    claim=f.claim,
                )

        report.citations = new_citations

    return report, trace
