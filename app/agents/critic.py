import re
from pathlib import Path
from typing import Any

from app.config import settings
from app.llm.client import llm_client
from app.models.schemas import CritiqueResult, Finding, Report, ResearchPlan, StepTrace

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[\"'(])")
_CITE_KEY = re.compile(r"\[\d+\]")


def _load_critic_prompt() -> str:
    prompt_file = Path(__file__).parent.parent / "prompts" / "critic_verify.md"
    if prompt_file.exists():
        try:
            return prompt_file.read_text(encoding="utf-8").strip()
        except Exception:
            pass
    return """Analyze a draft research report against the original research plan and verified findings.

Evaluate:
1. Unsupported claims: statements in the report sections that are not supported by any verified finding or citation key.
2. Missing plan coverage: sub-questions from the original plan that have no substantive coverage in the draft report.
3. Determine if the report passes factual verification standards (passed: bool).
"""


CRITIC_SYSTEM_PROMPT = _load_critic_prompt()


def split_cited_claims(report: Report) -> list[tuple[str, list[str]]]:
    """Extracts (claim_sentence, list_of_citation_keys) for every cited sentence."""
    claims: list[tuple[str, list[str]]] = []
    for sec in report.sections:
        for sent in _SENT.split(sec.content):
            keys = _CITE_KEY.findall(sent)
            if keys:
                clean_claim = _CITE_KEY.sub("", sent).strip()
                clean_claim = re.sub(r"\s+([.,;:!?])", r"\1", clean_claim)
                if clean_claim:
                    claims.append((clean_claim, list(dict.fromkeys(keys))))
    return claims


def remove_unsupported_sentences(report: Report, unsupported_claims: list[str]) -> Report:
    """Mechanically removes sentences corresponding to verified unsupported claims."""
    if not unsupported_claims:
        return report

    norm_unsupported = {
        re.sub(r"\s+", " ", c).strip().lower()
        for c in unsupported_claims
        if len(c.strip()) > 10
    }
    if not norm_unsupported:
        return report

    for sec in report.sections:
        kept = []
        for sent in _SENT.split(sec.content):
            clean_sent = _CITE_KEY.sub("", sent).strip().lower()
            clean_norm = re.sub(r"\s+", " ", clean_sent)
            if any(u in clean_norm or clean_norm in u for u in norm_unsupported):
                continue
            kept.append(sent)
        sec.content = " ".join(kept).strip()

    return report


def critique_report(
    plan: ResearchPlan,
    findings: list[Finding],
    draft_report: Report,
    run_id: str = "run_default",
    parent_step_id: str | None = None,
    budget_guard: Any | None = None,
) -> tuple[CritiqueResult, StepTrace]:
    """
    Critic specialist: Inspects draft report for unsupported claims, measures
    factual support rate, and determines review action (pass / revise_draft / retrieve_more).
    """
    plan_summary = "\n".join([f"- [{sq.id}] {sq.text}" for sq in plan.sub_questions])
    findings_by_id = {f.id: f for f in findings}
    finding_by_url = {f.source_url: f for f in findings}

    # Extract cited claims from report
    claims = split_cited_claims(draft_report)
    claims_context = ""
    for idx, (claim_text, keys) in enumerate(claims[:15], 1):
        cited_snippets = []
        for k in keys:
            c_info = draft_report.citations.get(k)
            if c_info:
                f = findings_by_id.get(c_info.finding_id) or finding_by_url.get(c_info.source_url)
                if f:
                    cited_snippets.append(f"[{k}] {f.snippet}")
        claims_context += (
            f"Claim {idx}: {claim_text}\n"
            f"Cited Evidence: {' | '.join(cited_snippets) if cited_snippets else 'No matching evidence found'}\n\n"
        )

    user_prompt = (
        f"Research Topic: {plan.topic}\n\n"
        f"Original Plan Sub-Questions:\n{plan_summary}\n\n"
        f"Cited Claims Under Evaluation:\n{claims_context or 'No cited claims in report.'}\n\n"
        f"Draft Summary: {draft_report.summary}\n\n"
        f"Evaluate the draft against the evidence passages. Return CritiqueResult with:\n"
        f"- unsupported_claims: list of statements that lack evidence support\n"
        f"- missing_plan_coverage: list of plan sub-questions not addressed\n"
        f"- passed: true if quality bar met, false otherwise"
    )

    messages = [
        {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    critique, trace = llm_client.generate_with_repair(
        agent_name="critic",
        messages=messages,
        output_schema=CritiqueResult,
        run_id=run_id,
        parent_step_id=parent_step_id,
        budget_guard=budget_guard,
        model=settings.critic_model,
    )

    # Calculate factual support rate
    total_claims = max(len(claims), 1)
    unsupported_count = len(critique.unsupported_claims)
    support_rate = round(max(0.0, (total_claims - unsupported_count) / total_claims), 2)
    critique.support_rate = support_rate

    # Determine action for self-correction loops
    threshold = getattr(settings, "support_threshold", 0.80)
    if unsupported_count > 0 and support_rate < threshold:
        critique.action = "revise_draft"
        critique.passed = False
    elif critique.missing_plan_coverage:
        critique.action = "retrieve_more"
        critique.passed = False
    else:
        critique.action = "pass"
        critique.passed = True

    return critique, trace
