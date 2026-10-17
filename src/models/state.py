from datetime import datetime, timezone
from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict
import operator

from src.models.schemas import (
    ResearchPlan,
    EvidenceRecord,
    ResearchReport,
    StepTrace,
)


def append_list(a: List[Any], b: List[Any]) -> List[Any]:
    """Reducer for merging lists in LangGraph state."""
    return a + b


class AgentState(TypedDict, total=False):
    """
    LangGraph State Schema.
    This shared state object is the single source of truth across all graph nodes.
    """
    # Core inputs & outputs
    run_id: str
    research_topic: str
    plan: Optional[ResearchPlan]
    report: Optional[ResearchReport]

    # Iteration & routing state
    current_subquestion_index: int
    completed_subquestion_ids: List[str]
    retries_by_subquestion: Dict[str, int]
    last_reviewer_feedback: Optional[str]

    # Findings provenance (cumulative list with reducer)
    findings: Annotated[List[EvidenceRecord], append_list]
    findings_by_subquestion: Dict[str, List[EvidenceRecord]]

    # Step-level tracing (cumulative list with reducer)
    trace_log: Annotated[List[StepTrace], append_list]

    # Hard budget tracking
    total_tokens: int
    total_searches: int
    start_time_epoch: float
    budget_exhausted: bool

    # Lifecycle & status
    status: str  # "in_progress", "completed", "failed", "budget_exceeded"
    error: Optional[str]
