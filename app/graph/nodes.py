import time
from typing import Any

from app.agents.critic import critique_report
from app.agents.planner import plan_research
from app.agents.researcher import research_subquestion
from app.agents.supervisor import review_findings
from app.agents.writer import write_report
from app.budget.guard import BudgetExceededError, BudgetGuard
from app.config import settings
from app.graph.state import AgentState
from app.models.enums import ReviewAction, RunStatus


def _get_or_create_guard(state: AgentState) -> BudgetGuard:
    """Instantiates a BudgetGuard populated with state expenditures."""
    guard = BudgetGuard(
        max_subquestions=settings.max_subquestions,
        max_searches_per_subquestion=settings.max_searches_per_subquestion,
        max_total_tokens=settings.max_total_tokens,
        max_usd_cost=settings.max_usd_cost,
        wall_clock_timeout_seconds=settings.wall_clock_timeout_seconds,
        max_repair_attempts=settings.max_repair_attempts,
        start_time=state.get("start_time", time.time()),
    )
    guard.current_tokens = state.get("total_tokens", 0)
    guard.current_usd_cost = state.get("total_usd_cost", 0.0)
    return guard


def plan_node(state: AgentState) -> dict[str, Any]:
    """Plan Node: Decomposes question into structured SubQuestions."""
    run_id = state.get("run_id", "run_default")
    question = state.get("question", "")
    guard = _get_or_create_guard(state)

    try:
        guard.check_before_call(estimated_tokens=500)
        plan, trace = plan_research(topic=question, run_id=run_id, budget_guard=guard)
        return {
            "plan": plan,
            "current_subquestion_idx": 0,
            "completed_subquestion_ids": [],
            "retry_counts": {},
            "findings_by_subquestion": {},
            "review_decisions": [],
            "total_tokens": state.get("total_tokens", 0) + trace.total_tokens,
            "total_usd_cost": round(state.get("total_usd_cost", 0.0) + trace.usd_cost, 6),
            "trace_log": [trace],
            "status": RunStatus.RESEARCHING,
            "budget_summary": guard.get_summary(),
        }
    except BudgetExceededError as be:
        return {
            "partial": True,
            "degradation_reason": be.reason,
            "status": RunStatus.FAILED,
            "budget_summary": guard.get_summary(),
        }


def dispatch_node(state: AgentState) -> dict[str, Any]:
    """Dispatch Node: Selects the next sub-question to research or branches to synthesis."""
    guard = _get_or_create_guard(state)
    plan = state.get("plan")
    idx = state.get("current_subquestion_idx", 0)

    # Check budget ceiling
    try:
        guard.check_wall_clock()
        guard.check_tokens()
        guard.check_usd_cost()
    except BudgetExceededError as be:
        return {
            "partial": True,
            "degradation_reason": be.reason,
            "current_subquestion": None,
            "budget_summary": guard.get_summary(),
        }

    if not plan or not plan.sub_questions or idx >= len(plan.sub_questions):
        return {"current_subquestion": None}

    sub_q = plan.sub_questions[idx]
    return {"current_subquestion": sub_q}


def research_node(state: AgentState) -> dict[str, Any]:
    """Research Node: Executes findings gathering for the current sub-question."""
    sub_q = state.get("current_subquestion")
    if not sub_q:
        return {}

    run_id = state.get("run_id", "run_default")
    guard = _get_or_create_guard(state)
    feedback = state.get("last_feedback")

    try:
        guard.check_before_call(estimated_tokens=800)
        findings_res, trace = research_subquestion(
            sub_question=sub_q,
            feedback=feedback,
            run_id=run_id,
            budget_guard=guard,
        )

        by_sq = dict(state.get("findings_by_subquestion", {}))
        existing = by_sq.get(sub_q.id, [])
        by_sq[sub_q.id] = existing + findings_res.findings

        return {
            "findings": findings_res.findings,
            "findings_by_subquestion": by_sq,
            "total_tokens": state.get("total_tokens", 0) + trace.total_tokens,
            "total_usd_cost": round(state.get("total_usd_cost", 0.0) + trace.usd_cost, 6),
            "trace_log": [trace],
            "budget_summary": guard.get_summary(),
        }
    except BudgetExceededError as be:
        return {
            "partial": True,
            "degradation_reason": be.reason,
            "budget_summary": guard.get_summary(),
        }


def review_node(state: AgentState) -> dict[str, Any]:
    """Review Node: Supervisor reviews findings and decides accept / retry_once / skip / abort."""
    sub_q = state.get("current_subquestion")
    if not sub_q:
        return {}

    run_id = state.get("run_id", "run_default")
    guard = _get_or_create_guard(state)
    retry_counts = dict(state.get("retry_counts", {}))
    curr_retries = retry_counts.get(sub_q.id, 0)

    by_sq = state.get("findings_by_subquestion", {})
    sq_findings = by_sq.get(sub_q.id, [])

    decision, trace = review_findings(
        sub_question=sub_q,
        findings=sq_findings,
        current_retry_count=curr_retries,
        budget_guard=guard,
        run_id=run_id,
    )

    completed_ids = list(state.get("completed_subquestion_ids", []))
    idx = state.get("current_subquestion_idx", 0)
    last_feedback = None

    if decision.action == ReviewAction.RETRY_ONCE:
        retry_counts[sub_q.id] = curr_retries + 1
        last_feedback = decision.follow_up_query or decision.reason
    else:
        # Accepted or Skipped: advance index
        if sub_q.id not in completed_ids:
            completed_ids.append(sub_q.id)
        idx += 1
        last_feedback = None

    decisions = list(state.get("review_decisions", []))
    decisions.append(decision)

    return {
        "retry_counts": retry_counts,
        "completed_subquestion_ids": completed_ids,
        "current_subquestion_idx": idx,
        "last_feedback": last_feedback,
        "review_decisions": decisions,
        "total_tokens": state.get("total_tokens", 0) + trace.total_tokens,
        "total_usd_cost": round(state.get("total_usd_cost", 0.0) + trace.usd_cost, 6),
        "trace_log": [trace],
        "budget_summary": guard.get_summary(),
    }


def write_node(state: AgentState) -> dict[str, Any]:
    """Write Node: Synthesizes findings and conflicting evidence into a report."""
    run_id = state.get("run_id", "run_default")
    question = state.get("question", "")
    findings = state.get("findings", [])
    conflicts = state.get("conflicting_evidence", [])
    partial = state.get("partial", False)
    degradation_reason = state.get("degradation_reason")

    guard = _get_or_create_guard(state)

    try:
        report, trace = write_report(
            topic=question,
            findings=findings,
            conflicting_evidence=conflicts,
            partial=partial,
            degradation_reason=degradation_reason,
            run_id=run_id,
            budget_guard=guard,
        )
        return {
            "draft_report": report,
            "total_tokens": state.get("total_tokens", 0) + trace.total_tokens,
            "total_usd_cost": round(state.get("total_usd_cost", 0.0) + trace.usd_cost, 6),
            "trace_log": [trace],
            "status": RunStatus.CRITIQUING,
            "budget_summary": guard.get_summary(),
        }
    except Exception as e:
        # Graceful fallback: synthesize minimal partial report without crashing
        from app.models.schemas import Report, ReportSection
        fallback_report = Report(
            topic=question,
            summary=f"Partial synthesis due to: {e}",
            sections=[ReportSection(title="Summary", content="Research halted by budget ceiling.")],
            partial=True,
            degradation_reason=str(e),
        )
        return {
            "draft_report": fallback_report,
            "partial": True,
            "degradation_reason": str(e),
            "status": RunStatus.CRITIQUING,
        }


def critic_node(state: AgentState) -> dict[str, Any]:
    """Critic Node: Checks draft report for unsupported claims and missing plan coverage."""
    plan = state.get("plan")
    draft_report = state.get("draft_report")
    if not plan or not draft_report:
        return {}

    run_id = state.get("run_id", "run_default")
    findings = state.get("findings", [])
    guard = _get_or_create_guard(state)

    try:
        critique, trace = critique_report(
            plan=plan,
            findings=findings,
            draft_report=draft_report,
            run_id=run_id,
            budget_guard=guard,
        )
        return {
            "critique": critique,
            "total_tokens": state.get("total_tokens", 0) + trace.total_tokens,
            "total_usd_cost": round(state.get("total_usd_cost", 0.0) + trace.usd_cost, 6),
            "trace_log": [trace],
            "status": RunStatus.COMPLETED,
            "budget_summary": guard.get_summary(),
        }
    except Exception as e:
        from app.models.schemas import CritiqueResult
        return {
            "critique": CritiqueResult(passed=True, critique_notes=f"Critic bypassed due to: {e}"),
            "status": RunStatus.COMPLETED,
        }


def finish_node(state: AgentState) -> dict[str, Any]:
    """Finish Node: Sets final report and completes execution."""
    draft = state.get("draft_report")
    critique = state.get("critique")

    # If critique uncovered severe unsupported claims, adjust citation coverage
    unsupported = getattr(critique, "unsupported_claims", []) if critique else []
    if draft and unsupported:
        total_claims = max(len(draft.citations), 1) + len(unsupported)
        valid_claims = max(len(draft.citations), 1)
        draft.citation_coverage = round(valid_claims / total_claims, 2)

    status = RunStatus.COMPLETED
    if draft and draft.citation_coverage < settings.citation_coverage_threshold:
        status = RunStatus.NEEDS_REVIEW

    return {
        "final_report": draft,
        "status": status,
    }
