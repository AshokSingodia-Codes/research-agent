from app.budget.guard import BudgetGuard
from app.config import settings
from app.llm.client import llm_client
from app.models.enums import ReviewAction
from app.models.schemas import Finding, ReviewDecision, StepTrace, SubQuestion

SUPERVISOR_SYSTEM_PROMPT = """Evaluate research findings for a specific sub-question and determine whether they meet quality and grounding criteria.

Decision Rules:
1. ACCEPT: Findings directly answer the sub-question with substantiated claims and credible sources.
2. RETRY_ONCE: Findings are insufficient, empty, or ungrounded, AND current retry count is 0. Suggest a refined search query.
3. SKIP: Findings remain insufficient after a retry, or relevant sources are completely unavailable. Proceed to next sub-question.
4. ABORT: Budget is exhausted or critical failure prevents further progress.

STRICT INVARIANT: You CANNOT recommend RETRY_ONCE if current_retry_count >= 1. You must ACCEPT or SKIP.
"""


def review_findings(
    sub_question: SubQuestion,
    findings: list[Finding],
    current_retry_count: int = 0,
    budget_guard: BudgetGuard | None = None,
    run_id: str = "run_default",
    parent_step_id: str | None = None,
) -> tuple[ReviewDecision, StepTrace]:
    """
    Supervisor specialist: Reviews findings and outputs a structured ReviewDecision.
    Enforces that retry_once is permitted at most once per sub-question.
    """
    # 1. Immediate budget circuit breaker check
    if budget_guard is not None:
        try:
            budget_guard.check_before_call(estimated_tokens=0)
        except Exception:
            pass
    if budget_guard is not None and budget_guard.is_exhausted:
        decision = ReviewDecision(
            sub_question_id=sub_question.id,
            action=ReviewAction.ABORT,
            reason=f"Terminating research due to budget exhaustion: {budget_guard.exhaustion_reason}",
        )
        trace = StepTrace(
            run_id=run_id,
            agent="supervisor",
            provider="code_guard",
            model="deterministic",
            input_data={"sub_question_id": sub_question.id, "findings_count": len(findings)},
            output_data=decision.model_dump(),
            timestamp=0.0,
        )
        return decision, trace

    # 2. Check if findings are empty or lack credible sources
    findings_count = len(findings)
    evidence_text = ""
    for f in findings:
        evidence_text += f"- Claim: {f.claim} | Source: {f.source_url} | Confidence: {f.confidence}\n"

    user_prompt = (
        f"Sub-Question ID: {sub_question.id}\n"
        f"Sub-Question: {sub_question.text}\n"
        f"Current Retry Count: {current_retry_count} (Max allowed: 1)\n"
        f"Number of Findings: {findings_count}\n\n"
        f"Findings Summary:\n{evidence_text or 'NO FINDINGS FOUND.'}\n\n"
        f"Provide your ReviewDecision adhering strictly to the schema and retry limit."
    )

    messages = [
        {"role": "system", "content": SUPERVISOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    decision, trace = llm_client.generate_with_repair(
        agent_name="supervisor",
        messages=messages,
        output_schema=ReviewDecision,
        run_id=run_id,
        parent_step_id=parent_step_id,
        budget_guard=budget_guard,
        model=settings.supervisor_model,
    )

    decision.sub_question_id = sub_question.id

    # 3. Hard Code Guard: Never allow retry_once if already retried
    if current_retry_count >= 1 and decision.action == ReviewAction.RETRY_ONCE:
        decision.action = ReviewAction.SKIP
        decision.reason = (
            f"Overridden by supervisor guard: Sub-question {sub_question.id} already used its single allowed retry."
        )

    return decision, trace
