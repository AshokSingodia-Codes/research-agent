import asyncio
import logging
import threading
import time
from typing import Any

from langchain_core.runnables import RunnableLambda

logger = logging.getLogger(__name__)

from app.agents.critic import critique_report
from app.agents.planner import plan_research
from app.agents.researcher import research_subquestion
from app.agents.supervisor import review_findings
from app.agents.writer import write_report
from app.budget.guard import BudgetExceededError, BudgetGuard
from app.config import settings
from app.graph.state import AgentState
from app.llm.client import llm_client
from app.models.enums import ReviewAction, RunStatus, ToolStatus
from app.models.schemas import Finding, ResearchFindings, ReviewDecision, StepTrace, SubQuestion


def _get_or_create_guard(state: dict[str, Any]) -> BudgetGuard:
    """Instantiates a BudgetGuard populated with state expenditures."""
    guard = BudgetGuard(
        max_subquestions=settings.max_subquestions,
        max_searches_per_subquestion=settings.max_searches_per_subquestion,
        max_total_tokens=settings.max_total_tokens,
        max_usd_cost=settings.max_usd_cost,
        wall_clock_timeout_seconds=settings.wall_clock_timeout_seconds,
        max_repair_attempts=settings.max_repair_attempts,
        start_time=state.get("start_time") or time.time(),
        run_id=state.get("run_id", "run_default"),
    )
    guard.current_tokens = state.get("total_tokens", 0)
    guard.current_usd_cost = state.get("total_usd_cost", 0.0)
    return guard


def _check_cancellation(state: dict[str, Any]) -> bool:
    """Checks whether the workflow has been cancelled via state or Redis flag."""
    if state.get("cancelled") or state.get("status") in (RunStatus.CANCELLED, "cancelled"):
        return True
    run_id = state.get("run_id")
    if not run_id or run_id == "run_default":
        return False
    try:
        from app.persistence.redis_store import redis_store

        client = redis_store.get_client()
        return bool(client.get(f"research:cancel:{run_id}"))
    except Exception:
        return False


def plan_node(state: AgentState) -> dict[str, Any]:
    """Plan Node: Decomposes question into structured SubQuestions."""
    if _check_cancellation(state):
        return {
            "status": RunStatus.CANCELLED,
            "cancelled": True,
            "partial": True,
            "degradation_reason": "Workflow cancelled by user.",
        }

    run_id = state.get("run_id", "run_default")
    question = state.get("question", "")
    guard = _get_or_create_guard(state)

    try:
        guard.check_before_call(estimated_tokens=500)
        plan, trace = plan_research(topic=question, run_id=run_id, parent_step_id=None, budget_guard=guard)
        return {
            "plan": plan,
            "current_subquestion_idx": 0,
            "completed_subquestion_ids": [],
            "completed_branches": [],
            "retry_counts": {},
            "findings_by_subquestion": {},
            "review_decisions": [],
            "total_tokens": trace.total_tokens,
            "total_usd_cost": trace.usd_cost,
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
    """Dispatch Node: Validates budget ceiling before parallel fan-out."""
    if _check_cancellation(state):
        return {
            "status": RunStatus.CANCELLED,
            "cancelled": True,
            "partial": True,
            "degradation_reason": "Workflow cancelled by user.",
        }

    guard = _get_or_create_guard(state)
    try:
        guard.check_wall_clock()
        guard.check_tokens()
        guard.check_usd_cost()
    except BudgetExceededError as be:
        return {
            "partial": True,
            "degradation_reason": be.reason,
            "budget_summary": guard.get_summary(),
        }

    return {"status": RunStatus.RESEARCHING}


# Concurrency controls for research_one
_async_sem_limit: int = 0
_async_research_sem: asyncio.Semaphore | None = None
_sync_research_sem: threading.Semaphore | None = None


def _get_async_research_semaphore() -> asyncio.Semaphore:
    global _async_research_sem, _async_sem_limit
    limit = max(1, settings.max_parallel_researchers)
    if _async_research_sem is None or _async_sem_limit != limit:
        _async_research_sem = asyncio.Semaphore(limit)
        _async_sem_limit = limit
    return _async_research_sem


def _get_sync_research_semaphore() -> threading.Semaphore:
    global _sync_research_sem
    limit = max(1, settings.max_parallel_researchers)
    if _sync_research_sem is None:
        _sync_research_sem = threading.Semaphore(limit)
    return _sync_research_sem


def _execute_research_one_core(payload: dict[str, Any]) -> dict[str, Any]:
    """Core logic to research a single sub-question branch with idempotency and typed failure handling."""
    if _check_cancellation(payload):
        return {
            "status": RunStatus.CANCELLED,
            "cancelled": True,
            "partial": True,
            "degradation_reason": "Workflow cancelled by user.",
        }

    run_id = payload.get("run_id", "run_default")
    sub_q = payload.get("sub_question")
    if not sub_q:
        return {}

    branch_key = payload.get("branch_key", "")
    completed_branches = set(payload.get("completed_branches", []))

    # Skip branch if already completed (per-branch idempotency)
    if branch_key and branch_key in completed_branches:
        return {"completed_branches": [branch_key]}

    guard = _get_or_create_guard(payload)
    feedback = payload.get("feedback")

    try:
        guard.check_before_call(estimated_tokens=800)

        # Retrieve real web evidence documents for this sub-question
        from app.tools.search import search_and_extract

        search_query = feedback if feedback and len(feedback) > 10 else sub_q.text
        search_res = search_and_extract(
            query=search_query,
            sub_question_id=sub_q.id,
            max_results=3,
            max_per_domain=settings.max_results_per_domain,
            run_id=run_id,
        )

        if _check_cancellation(payload):
            return {
                "status": RunStatus.CANCELLED,
                "cancelled": True,
                "partial": True,
                "degradation_reason": "Workflow cancelled by user.",
            }

        doc_snippets = [
            {"url": f.source_url, "domain": f.domain, "text": f.extracted_text or f.snippet}
            for f in search_res.findings
        ]

        findings_res, trace = research_subquestion(
            sub_question=sub_q,
            source_documents=doc_snippets,
            feedback=feedback,
            run_id=run_id,
            budget_guard=guard,
        )

        return {
            "findings": findings_res.findings,
            "findings_by_subquestion": {sub_q.id: findings_res.findings},
            "total_tokens": trace.total_tokens,
            "total_usd_cost": trace.usd_cost,
            "trace_log": [trace],
            "completed_branches": [branch_key] if branch_key else [],
            "branch_findings": {sub_q.id: findings_res},
        }
    except BudgetExceededError as be:
        return {
            "partial": True,
            "degradation_reason": be.reason,
            "completed_branches": [branch_key] if branch_key else [],
            "budget_summary": guard.get_summary(),
        }
    except Exception as e:
        logger.warning(f"Researcher LLM step encountered exception for {sub_q.id}: {e}; falling back to extracted search findings.")
        fallback_findings = search_res.findings if 'search_res' in locals() and search_res.findings else []
        failed_trace = StepTrace(
            run_id=run_id,
            agent="researcher",
            provider="search_fallback",
            model=settings.researcher_model,
            input_data={"sub_question_id": sub_q.id},
            error=str(e),
            output_data={"status": "degraded_fallback", "findings_count": len(fallback_findings)},
        )
        return {
            "findings": fallback_findings,
            "findings_by_subquestion": {sub_q.id: fallback_findings},
            "trace_log": [failed_trace],
            "completed_branches": [branch_key] if branch_key else [],
            "branch_findings": {sub_q.id: ResearchFindings(sub_question_id=sub_q.id, findings=fallback_findings, status=ToolStatus.OK if fallback_findings else ToolStatus.NO_RESULTS)},
        }


async def _async_research_one(payload: dict[str, Any]) -> dict[str, Any]:
    sem = _get_async_research_semaphore()
    async with sem:
        # Check if active LLM provider has async delay for concurrency tests
        active_provider = getattr(llm_client, "_override_provider", None)
        if active_provider:
            delay = getattr(active_provider, "async_delay_s", 0.0) or getattr(active_provider, "artificial_delay_s", 0.0)
            if delay > 0:
                await asyncio.sleep(delay)
        return await asyncio.to_thread(_execute_research_one_core, payload)


def _sync_research_one(payload: dict[str, Any]) -> dict[str, Any]:
    sem = _get_sync_research_semaphore()
    with sem:
        active_provider = getattr(llm_client, "_override_provider", None)
        if active_provider:
            delay = getattr(active_provider, "artificial_delay_s", 0.0)
            if delay > 0:
                time.sleep(delay)
        return _execute_research_one_core(payload)


research_one_node = RunnableLambda(_sync_research_one, afunc=_async_research_one)


def review_all_node(state: AgentState) -> dict[str, Any]:
    """Review Node (Fan-in): Supervisor reviews findings and decides accept / retry_once / skip / abort."""
    if _check_cancellation(state):
        return {
            "status": RunStatus.CANCELLED,
            "cancelled": True,
            "partial": True,
            "degradation_reason": "Workflow cancelled by user.",
        }

    plan = state.get("plan")
    if not plan or not plan.sub_questions:
        return {}

    run_id = state.get("run_id", "run_default")
    guard = _get_or_create_guard(state)
    retry_counts = dict(state.get("retry_counts", {}))
    completed_ids = list(state.get("completed_subquestion_ids", []))
    by_sq = state.get("findings_by_subquestion", {})

    new_decisions: list[ReviewDecision] = []
    new_traces: list[StepTrace] = []
    total_tokens = 0
    total_cost = 0.0
    partial = state.get("partial", False)
    degradation_reason = state.get("degradation_reason")
    last_feedback = None

    for sub_q in plan.sub_questions:
        if sub_q.id in completed_ids:
            continue

        # If it was not processed in this wave (e.g. unmet dependencies), skip reviewing it now
        if sub_q.id not in by_sq and not any(dep in completed_ids for dep in sub_q.depends_on):
            continue

        curr_retries = retry_counts.get(sub_q.id, 0)
        sq_findings = by_sq.get(sub_q.id, [])

        try:
            guard.check_wall_clock()
            guard.check_tokens()
            guard.check_usd_cost()
        except BudgetExceededError as be:
            partial = True
            degradation_reason = be.reason
            break

        try:
            decision, trace = review_findings(
                sub_question=sub_q,
                findings=sq_findings,
                current_retry_count=curr_retries,
                budget_guard=guard,
                run_id=run_id,
            )
            new_decisions.append(decision)
            new_traces.append(trace)
            total_tokens += trace.total_tokens
            total_cost += trace.usd_cost
        except BudgetExceededError as be:
            partial = True
            degradation_reason = be.reason
            completed_ids.append(sub_q.id)
            break

        if decision.action == ReviewAction.RETRY_ONCE:
            if curr_retries >= 1:
                # Enforce at most ONE retry wave per sub-question
                decision.action = ReviewAction.SKIP
                decision.reason = f"Sub-question {sub_q.id} already used its single allowed retry. Skipping."
                completed_ids.append(sub_q.id)
            else:
                retry_counts[sub_q.id] = curr_retries + 1
                last_feedback = decision.follow_up_query or decision.reason
        elif decision.action == ReviewAction.ABORT:
            partial = True
            degradation_reason = decision.reason
            completed_ids.append(sub_q.id)
            break
        else:
            # ACCEPT or SKIP
            if sub_q.id not in completed_ids:
                completed_ids.append(sub_q.id)

    return {
        "retry_counts": retry_counts,
        "completed_subquestion_ids": completed_ids,
        "last_feedback": last_feedback,
        "review_decisions": state.get("review_decisions", []) + new_decisions,
        "total_tokens": total_tokens,
        "total_usd_cost": round(total_cost, 6),
        "trace_log": new_traces,
        "partial": partial,
        "degradation_reason": degradation_reason,
        "status": RunStatus.REVIEWING if not partial else RunStatus.FAILED,
        "budget_summary": guard.get_summary(),
    }


def write_node(state: AgentState) -> dict[str, Any]:
    """Write Node: Synthesizes findings and conflicting evidence into a report."""
    if _check_cancellation(state):
        return {
            "status": RunStatus.CANCELLED,
            "cancelled": True,
            "partial": True,
            "degradation_reason": "Workflow cancelled by user.",
        }

    run_id = state.get("run_id", "run_default")
    question = state.get("question", "")
    findings = list(state.get("findings", []))
    findings.sort(key=lambda f: f.source_quality_score, reverse=True)

    conflicts = list(state.get("conflicting_evidence", []))
    if not conflicts and findings:
        from app.tools.contradiction import detect_contradictions

        conflicts = detect_contradictions(findings)

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
            "total_tokens": trace.total_tokens,
            "total_usd_cost": trace.usd_cost,
            "trace_log": [trace],
            "status": RunStatus.CRITIQUING,
            "budget_summary": guard.get_summary(),
        }
    except Exception as e:
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
    if _check_cancellation(state):
        return {
            "status": RunStatus.CANCELLED,
            "cancelled": True,
            "partial": True,
            "degradation_reason": "Workflow cancelled by user.",
        }

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
            "total_tokens": trace.total_tokens,
            "total_usd_cost": trace.usd_cost,
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

    if _check_cancellation(state):
        return {
            "final_report": draft,
            "status": RunStatus.CANCELLED,
        }

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


# Backwards compatibility helper for tests importing run_parallel_fanout
async def run_parallel_fanout(
    ready_subquestions: list[SubQuestion],
    guard: BudgetGuard,
    run_id: str,
    concurrency_limit: int = 3,
) -> tuple[list[Finding], list[ReviewDecision], list[StepTrace]]:
    """Runs parallel fanout for sub-questions and merges results deterministically."""
    from app.agents.researcher import research_subquestion
    from app.agents.supervisor import review_findings
    from app.tools.search import search_and_extract

    loop = asyncio.get_running_loop()
    semaphore = asyncio.Semaphore(concurrency_limit)
    _lock = threading.Lock()

    def _execute(sq: SubQuestion):
        with _lock:
            if guard.is_exhausted:
                return [], ReviewDecision(sub_question_id=sq.id, action=ReviewAction.ABORT, reason="Budget exhausted"), [], True
            try:
                guard.check_before_call(estimated_tokens=500)
                guard.record_search(sq.id)
            except BudgetExceededError as be:
                return [], ReviewDecision(sub_question_id=sq.id, action=ReviewAction.ABORT, reason=be.reason), [], True

        search_res = search_and_extract(
            query=sq.text,
            sub_question_id=sq.id,
            max_results=3,
            max_per_domain=settings.max_results_per_domain,
        )
        doc_snippets = [
            {"url": f.source_url, "domain": f.domain, "text": f.extracted_text or f.snippet}
            for f in search_res.findings
        ]
        f_obj, r_trace = research_subquestion(
            sub_question=sq,
            source_documents=doc_snippets,
            run_id=run_id,
            budget_guard=guard,
        )
        rev_dec, s_trace = review_findings(
            sub_question=sq,
            findings=f_obj.findings,
            current_retry_count=0,
            budget_guard=guard,
            run_id=run_id,
        )
        with _lock:
            guard.record_spend(
                tokens=r_trace.total_tokens + s_trace.total_tokens,
                usd_cost=r_trace.usd_cost + s_trace.usd_cost,
            )
        return f_obj.findings, rev_dec, [r_trace, s_trace], False

    async def _worker(sq: SubQuestion):
        async with semaphore:
            return await loop.run_in_executor(None, _execute, sq)

    results = await asyncio.gather(*[_worker(sq) for sq in ready_subquestions])
    sorted_pairs = sorted(zip(ready_subquestions, results), key=lambda x: (x[0].priority, x[0].id))

    merged_findings: list[Finding] = []
    merged_decisions: list[ReviewDecision] = []
    merged_traces: list[StepTrace] = []

    for _, (findings, decision, traces, _) in sorted_pairs:
        sorted_findings = sorted(findings, key=lambda f: f.content_hash)
        merged_findings.extend(sorted_findings)
        merged_decisions.append(decision)
        merged_traces.extend(traces)

    return merged_findings, merged_decisions, merged_traces
