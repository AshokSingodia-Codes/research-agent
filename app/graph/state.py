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


class AgentState(TypedDict, total=False):
    run_id: str
    question: str
    plan: ResearchPlan | None
    current_subquestion_idx: int
    current_subquestion: SubQuestion | None
    completed_subquestion_ids: list[str]
    retry_counts: dict[str, int]
    findings: Annotated[list[Finding], operator.add]
    findings_by_subquestion: dict[str, list[Finding]]
    conflicting_evidence: list[Contradiction]
    review_decisions: list[ReviewDecision]
    last_feedback: str | None
    draft_report: Report | None
    final_report: Report | None
    critique: CritiqueResult | None
    partial: bool
    degradation_reason: str | None
    total_tokens: int
    total_usd_cost: float
    start_time: float
    step_index: int
    require_approval: bool
    approved: bool
    cancelled: bool
    status: str
    budget_summary: dict[str, Any]
    trace_log: Annotated[list[StepTrace], operator.add]
