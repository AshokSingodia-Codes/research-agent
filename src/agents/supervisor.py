import time
from typing import Optional, Tuple
from langchain_core.messages import SystemMessage, HumanMessage

from src.config import settings
from src.models.schemas import (
    SubQuestion,
    ResearcherOutput,
    SupervisorReview,
    StepTrace,
)
from src.agents.llm import get_anthropic_chat_model

SUPERVISOR_CRITIQUE_PROMPT = """You are an objective Research Supervisor and Reviewer.
Your role is to rigorously inspect researcher findings against the sub-question requirements.

Criteria for Approval:
1. Findings contain verifiable factual evidence addressing the core sub-question.
2. At least one credible source URL is provided.
3. If search status was failed, rate limited, or empty, the sub-question cannot be approved as sufficient.

Respond with:
- is_sufficient: bool
- feedback: specific missing elements or confirmation of quality
- allow_retry: whether a retry could remedy the missing data
"""


def review_research_output(
    sub_question: SubQuestion,
    research_output: ResearcherOutput,
    current_retry_count: int,
    total_tokens: int,
    elapsed_seconds: float,
    is_last_subquestion: bool
) -> Tuple[SupervisorReview, StepTrace]:
    """
    Evaluates researcher findings, enforces hard budgets in code, and limits revisions to exactly 1 pass.
    Returns: (SupervisorReview, StepTrace)
    """
    start_time = time.time()

    # 1. HARD BUDGET CIRCUIT BREAKERS
    # Check wall-clock ceiling
    if elapsed_seconds >= settings.wall_clock_timeout_seconds:
        review = SupervisorReview(
            is_sufficient=False,
            feedback=f"Wall-clock budget exhausted ({elapsed_seconds:.1f}s >= {settings.wall_clock_timeout_seconds}s). Forcing report synthesis.",
            allow_retry=False,
            next_action="synthesize_report"
        )
        return review, _create_trace(start_time, "Wall-clock timeout triggered", "synthesize_report")

    # Check total token budget
    if total_tokens >= settings.max_total_tokens:
        review = SupervisorReview(
            is_sufficient=False,
            feedback=f"Token budget exhausted ({total_tokens} >= {settings.max_total_tokens}). Forcing report synthesis.",
            allow_retry=False,
            next_action="synthesize_report"
        )
        return review, _create_trace(start_time, "Token budget cap triggered", "synthesize_report")

    # 2. SOURCE & QUALITY EVALUATION
    findings = research_output.findings
    has_valid_sources = len(findings) > 0 and all(f.source_url.startswith("http") for f in findings)
    search_succeeded = research_output.search_status.status == "success"

    is_sufficient = search_succeeded and has_valid_sources
    feedback = "Findings satisfy evidence requirements with source provenance."

    if not is_sufficient:
        feedback = f"Deficient evidence: search status={research_output.search_status.status}, source count={len(findings)}."

    # 3. REVISION LOOP ENFORCEMENT: Max 1 retry pass per sub-question
    can_retry = (not is_sufficient) and (current_retry_count < settings.max_revision_loops)

    if can_retry:
        next_action = "retry_subquestion"
        decision_summary = f"Rejected (attempt {current_retry_count + 1}). Sending back to researcher for 1 allowable revision."
    elif is_last_subquestion:
        next_action = "synthesize_report"
        decision_summary = "Final sub-question completed. Routing to Writer for report synthesis."
    else:
        next_action = "continue_research"
        decision_summary = f"Sub-question concluded (retries={current_retry_count}). Routing to next sub-question."

    review = SupervisorReview(
        is_sufficient=is_sufficient,
        feedback=feedback,
        allow_retry=can_retry,
        next_action=next_action
    )

    trace = _create_trace(start_time, decision_summary, next_action)
    return review, trace


def _create_trace(start_time: float, summary: str, next_action: str) -> StepTrace:
    latency_ms = (time.time() - start_time) * 1000.0
    return StepTrace(
        step_id=f"supervisor_{int(start_time)}",
        node_name="supervisor",
        latency_ms=round(latency_ms, 2),
        tokens_used=20,
        input_summary="Reviewing findings & budget constraints",
        output_summary=f"Action: {next_action}. {summary}",
        tool_calls=[]
    )
