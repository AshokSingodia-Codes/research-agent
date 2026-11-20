from app.graph.routing import route_after_critic
from app.graph.state import AgentState
from app.models.schemas import CritiqueResult


def test_route_after_critic_revise_draft():
    critique = CritiqueResult(action="revise_draft", passed=False, unsupported_claims=["Bogus claim"])
    state: AgentState = {
        "critique": critique,
        "revision_loops": 0,
        "partial": False,
        "cancelled": False,
    }
    decision = route_after_critic(state)
    assert decision == "write"


def test_route_after_critic_retrieve_more():
    critique = CritiqueResult(action="retrieve_more", passed=False, missing_plan_coverage=["eBPF security"])
    state: AgentState = {
        "critique": critique,
        "revision_loops": 0,
        "partial": False,
        "cancelled": False,
    }
    decision = route_after_critic(state)
    assert decision == "dispatch"


def test_route_after_critic_iteration_limit():
    critique = CritiqueResult(action="revise_draft", passed=False, unsupported_claims=["Bogus claim"])
    # Revision loop max reached
    state: AgentState = {
        "critique": critique,
        "revision_loops": 1,
        "partial": False,
        "cancelled": False,
    }
    decision = route_after_critic(state)
    assert decision == "finish"

