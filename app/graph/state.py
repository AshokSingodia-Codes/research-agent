import operator
from typing import Annotated, Any

from typing_extensions import TypedDict

from app.models.schemas import (
    Contradiction,
    CritiqueResult,
    Finding,
    Report,
    ResearchPlan,
    ReviewDecision,
    StepTrace,
    SubQuestion,
)


def reduce_findings(existing: list[Finding] | None, incoming: list[Finding] | None) -> list[Finding]:
    """Reduces and deduplicates findings by (sub_question_id, source_url)."""
    merged = list(existing or [])
    seen = {(f.sub_question_id, getattr(f, "source_url", getattr(f, "url", ""))) for f in merged}
    for f in incoming or []:
        url = getattr(f, "source_url", getattr(f, "url", ""))
        key = (f.sub_question_id, url)
        if key not in seen:
            seen.add(key)
            merged.append(f)
    return merged


def reduce_completed_subquestions(existing: list[str] | None, incoming: list[str] | None) -> list[str]:
    """Combines completed sub-question IDs or branch keys preserving uniqueness."""
    merged = list(existing or [])
    seen = set(merged)
    for sq_id in incoming or []:
        if sq_id not in seen:
            seen.add(sq_id)
            merged.append(sq_id)
    return merged


def reduce_step_logs(existing: list[StepTrace] | None, incoming: list[StepTrace] | None) -> list[StepTrace]:
    """Appends step traces from concurrent branch executions."""
    merged = list(existing or [])
    merged.extend(incoming or [])
    return merged


def reduce_spend_float(existing: float | None, incoming: float | None) -> float:
    return round((existing or 0.0) + (incoming or 0.0), 6)


def reduce_dict(existing: dict[str, Any] | None, incoming: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(existing or {})
    merged.update(incoming or {})
    return merged


def reduce_findings_by_subquestion(
    existing: dict[str, list[Finding]] | None,
    incoming: dict[str, list[Finding]] | None,
) -> dict[str, list[Finding]]:
    merged = dict(existing or {})
    for sq_id, findings in (incoming or {}).items():
        if sq_id in merged:
            merged[sq_id] = reduce_findings(merged[sq_id], findings)
        else:
            merged[sq_id] = list(findings)
    return merged


class AgentState(TypedDict, total=False):
    run_id: str
    question: str
    plan: ResearchPlan | None
    current_subquestion_idx: int
    current_subquestion: SubQuestion | None
    completed_subquestion_ids: Annotated[list[str], reduce_completed_subquestions]
    completed_branches: Annotated[list[str], reduce_completed_subquestions]
    retry_counts: Annotated[dict[str, int], reduce_dict]
    findings: Annotated[list[Finding], reduce_findings]
    findings_by_subquestion: Annotated[dict[str, list[Finding]], reduce_findings_by_subquestion]
    branch_findings: Annotated[dict[str, Any], reduce_dict]
    conflicting_evidence: list[Contradiction]
    review_decisions: list[ReviewDecision]
    last_feedback: str | None
    draft_report: Report | None
    final_report: Report | None
    critique: CritiqueResult | None
    partial: bool
    degradation_reason: str | None
    total_tokens: Annotated[int, operator.add]
    total_usd_cost: Annotated[float, reduce_spend_float]
    start_time: float
    step_index: int
    require_approval: bool
    approved: bool
    cancelled: bool
    status: str
    budget_summary: dict[str, Any]
    trace_log: Annotated[list[StepTrace], reduce_step_logs]
