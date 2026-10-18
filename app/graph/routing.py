import hashlib
from typing import Literal

from langgraph.types import Send

from app.graph.state import AgentState
from app.models.schemas import SubQuestion


def get_ready_subquestions(
    all_subquestions: list[SubQuestion],
    completed_ids: set[str],
    in_flight_ids: set[str] | None = None,
) -> list[SubQuestion]:
    """
    Returns sub-questions whose dependencies are satisfied and are not yet completed or in flight.
    """
    in_flight = in_flight_ids or set()
    ready = []
    for sq in all_subquestions:
        if sq.id in completed_ids or sq.id in in_flight:
            continue
        # Check if all depends_on are completed
        if all(dep in completed_ids for dep in getattr(sq, "depends_on", [])):
            ready.append(sq)
    return ready


def route_after_plan(state: AgentState) -> Literal["dispatch", "write"]:
    """Routes after planning: if plan failed or degraded, write partial report; else dispatch."""
    if state.get("partial") or not state.get("plan"):
        return "write"
    return "dispatch"


def route_after_dispatch(state: AgentState) -> list[Send] | Literal["write"]:
    """
    Computes ready sub-questions from depends_on (topological waves) and returns
    [Send("research_one", {...}) for each ready sub-question], or routes to 'write' if done.
    """
    if state.get("partial") or state.get("cancelled"):
        return "write"

    plan = state.get("plan")
    if not plan or not plan.sub_questions:
        return "write"

    completed_ids = set(state.get("completed_subquestion_ids", []))
    ready = get_ready_subquestions(plan.sub_questions, completed_ids)
    if not ready:
        return "write"

    run_id = state.get("run_id", "run_default")
    last_feedback = state.get("last_feedback")
    retry_counts = state.get("retry_counts", {})
    completed_branches = set(state.get("completed_branches", []))

    sends = []
    for sq in ready:
        feedback = last_feedback if retry_counts.get(sq.id, 0) > 0 else None
        # Per-branch idempotency key (run_id, node, sub_question_id, input_hash)
        input_data = f"{sq.text}:{feedback or ''}:{retry_counts.get(sq.id, 0)}"
        input_hash = hashlib.sha256(input_data.encode()).hexdigest()[:12]
        branch_key = f"{run_id}:research_one:{sq.id}:{input_hash}"

        sends.append(
            Send(
                "research_one",
                {
                    "run_id": run_id,
                    "sub_question": sq,
                    "feedback": feedback,
                    "branch_key": branch_key,
                    "completed_branches": list(completed_branches),
                    "start_time": state.get("start_time"),
                    "total_tokens": state.get("total_tokens", 0),
                    "total_usd_cost": state.get("total_usd_cost", 0.0),
                },
            )
        )
    return sends


def route_after_review(state: AgentState) -> Literal["dispatch", "write"]:
    """
    Routes after review_all fan-in:
    - If all sub-questions are completed (and no retries pending), routes to 'write'
    - If partial / cancelled, routes to 'write'
    - Otherwise routes back to 'dispatch' for next topological wave or retry wave
    """
    if state.get("partial") or state.get("cancelled"):
        return "write"

    plan = state.get("plan")
    if not plan or not plan.sub_questions:
        return "write"

    completed_ids = set(state.get("completed_subquestion_ids", []))
    all_sq_ids = {sq.id for sq in plan.sub_questions}

    if all_sq_ids.issubset(completed_ids):
        return "write"

    return "dispatch"
