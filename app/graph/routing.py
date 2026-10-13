from typing import Literal

from app.graph.state import AgentState
from app.models.enums import ReviewAction


def route_after_plan(state: AgentState) -> Literal["dispatch", "write"]:
    """Routes after planning: if plan failed or degraded, write partial report; else dispatch."""
    if state.get("partial") or not state.get("plan"):
        return "write"
    return "dispatch"


def route_after_dispatch(state: AgentState) -> Literal["research", "write"]:
    """Routes after dispatch: if no more sub-questions or degraded, synthesize report; else research."""
    if state.get("partial") or state.get("current_subquestion") is None:
        return "write"
    return "research"


def route_after_review(state: AgentState) -> Literal["research", "dispatch", "write"]:
    """
    Routes after supervisor review:
    - retry_once: sends work back to research node
    - abort or partial: routes immediately to write for graceful degradation
    - accept or skip: routes to dispatch for the next sub-question
    """
    if state.get("partial"):
        return "write"

    decisions = state.get("review_decisions", [])
    if not decisions:
        return "dispatch"

    last_decision = decisions[-1]
    if last_decision.action == ReviewAction.RETRY_ONCE:
        return "research"
    elif last_decision.action == ReviewAction.ABORT:
        return "write"
    else:
        return "dispatch"
