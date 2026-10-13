from typing import Any

from app.llm.client import llm_client
from app.models.schemas import CritiqueResult, Finding, Report, ResearchPlan, StepTrace

CRITIC_SYSTEM_PROMPT = """Analyze a draft research report against the original research plan and verified findings.

Evaluate:
1. Unsupported claims: statements in the report sections that are not supported by any verified finding or citation key.
2. Missing plan coverage: sub-questions from the original plan that have no substantive coverage in the draft report.
3. Determine if the report passes factual verification standards (passed: bool).
"""


def critique_report(
    plan: ResearchPlan,
    findings: list[Finding],
    draft_report: Report,
    run_id: str = "run_default",
    parent_step_id: str | None = None,
    budget_guard: Any | None = None,
) -> tuple[CritiqueResult, StepTrace]:
    """
    Critic specialist: Inspects draft report for unsupported claims and missing coverage.
    """
    plan_summary = "\n".join([f"- [{sq.id}] {sq.text}" for sq in plan.sub_questions])
    findings_summary = "\n".join([f"- [{f.id}] {f.claim}" for f in findings[:25]])
    report_text = f"Summary: {draft_report.summary}\n\n"
    for s in draft_report.sections:
        report_text += f"Section: {s.title}\n{s.content}\nCitations: {', '.join(s.citations)}\n\n"

    user_prompt = (
        f"Research Topic: {plan.topic}\n\n"
        f"Original Plan Sub-Questions:\n{plan_summary}\n\n"
        f"Verified Findings:\n{findings_summary}\n\n"
        f"Draft Report Under Evaluation:\n{report_text}\n\n"
        f"Perform verification and return CritiqueResult."
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
    )

    return critique, trace
