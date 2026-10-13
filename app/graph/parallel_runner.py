import asyncio
import threading

from app.agents.researcher import research_subquestion
from app.agents.supervisor import review_findings
from app.budget.guard import BudgetExceededError, BudgetGuard
from app.config import settings
from app.models.enums import ReviewAction
from app.models.schemas import Finding, ReviewDecision, StepTrace, SubQuestion
from app.tools.search import search_and_extract

# Thread-safe lock for atomic budget tracking under concurrency
_budget_lock = threading.Lock()


def get_ready_subquestions(
    all_subquestions: list[SubQuestion],
    completed_ids: set[str],
    in_flight_ids: set[str],
) -> list[SubQuestion]:
    """
    Returns sub-questions whose dependencies are satisfied and are not yet completed or in flight.
    """
    ready = []
    for sq in all_subquestions:
        if sq.id in completed_ids or sq.id in in_flight_ids:
            continue
        # Check if all depends_on are completed
        if all(dep in completed_ids for dep in sq.depends_on):
            ready.append(sq)
    return ready


def execute_subquestion_pass(
    sub_q: SubQuestion,
    guard: BudgetGuard,
    run_id: str = "run_default",
    current_retry: int = 0,
    feedback: str | None = None,
) -> tuple[list[Finding], ReviewDecision, list[StepTrace], bool]:
    """
    Executes a single researcher + reviewer pass for one sub-question with thread-safe budget protection.
    Returns: (findings, review_decision, traces, budget_exhausted)
    """
    with _budget_lock:
        if guard.is_exhausted:
            decision = ReviewDecision(
                sub_question_id=sub_q.id,
                action=ReviewAction.ABORT,
                reason="Budget exhausted prior to execution",
            )
            return [], decision, [], True

        try:
            guard.check_before_call(estimated_tokens=500)
            guard.record_search(sub_q.id)
        except BudgetExceededError as be:
            decision = ReviewDecision(
                sub_question_id=sub_q.id,
                action=ReviewAction.ABORT,
                reason=be.reason,
            )
            return [], decision, [], True

    # 1. Search and page extraction
    search_res = search_and_extract(
        query=f"{sub_q.text} {feedback or ''}".strip(),
        sub_question_id=sub_q.id,
        max_results=3,
        max_per_domain=settings.max_results_per_domain,
    )

    doc_snippets = [
        {"url": f.source_url, "domain": f.domain, "text": f.extracted_text or f.snippet}
        for f in search_res.findings
    ]

    # 2. Researcher synthesis
    findings_obj, r_trace = research_subquestion(
        sub_question=sub_q,
        source_documents=doc_snippets,
        feedback=feedback,
        run_id=run_id,
        budget_guard=guard,
    )

    # 3. Supervisor review
    review_dec, s_trace = review_findings(
        sub_question=sub_q,
        findings=findings_obj.findings,
        current_retry_count=current_retry,
        budget_guard=guard,
        run_id=run_id,
    )

    with _budget_lock:
        guard.record_spend(
            tokens=r_trace.total_tokens + s_trace.total_tokens,
            usd_cost=r_trace.usd_cost + s_trace.usd_cost,
        )

    return findings_obj.findings, review_dec, [r_trace, s_trace], False


async def run_parallel_fanout(
    ready_subquestions: list[SubQuestion],
    guard: BudgetGuard,
    run_id: str,
    concurrency_limit: int = 3,
) -> tuple[list[Finding], list[ReviewDecision], list[StepTrace]]:
    """
    Executes multiple independent sub-questions concurrently up to concurrency limit,
    merging results deterministically by (priority, id).
    """
    loop = asyncio.get_running_loop()
    semaphore = asyncio.Semaphore(concurrency_limit)

    async def _worker(sq: SubQuestion):
        async with semaphore:
            return await loop.run_in_executor(
                None,
                execute_subquestion_pass,
                sq,
                guard,
                run_id,
                0,
                None,
            )

    results = await asyncio.gather(*[_worker(sq) for sq in ready_subquestions])

    # Deterministic merge: sort results by sub-question ID and priority
    sorted_pairs = sorted(zip(ready_subquestions, results), key=lambda x: (x[0].priority, x[0].id))

    merged_findings: list[Finding] = []
    merged_decisions: list[ReviewDecision] = []
    merged_traces: list[StepTrace] = []

    for _, (findings, decision, traces, _) in sorted_pairs:
        # Sort findings within sub-question by content_hash for determinism
        sorted_findings = sorted(findings, key=lambda f: f.content_hash)
        merged_findings.extend(sorted_findings)
        merged_decisions.append(decision)
        merged_traces.extend(traces)

    return merged_findings, merged_decisions, merged_traces
