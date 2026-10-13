from typing import Any

from langgraph.graph import END, START, StateGraph

from app.graph.nodes import (
    critic_node,
    dispatch_node,
    finish_node,
    plan_node,
    research_node,
    review_node,
    write_node,
)
from app.graph.routing import (
    route_after_dispatch,
    route_after_plan,
    route_after_review,
)
from app.graph.state import AgentState


def build_research_graph(
    checkpointer: Any | None = None,
    interrupt_before_write: bool = False,
):
    """
    Constructs the LangGraph multi-agent research workflow:
    plan -> dispatch -> research -> review -> (loop | write) -> critic -> finish -> END
    """
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("plan", plan_node)
    graph.add_node("dispatch", dispatch_node)
    graph.add_node("research", research_node)
    graph.add_node("review", review_node)
    graph.add_node("write", write_node)
    graph.add_node("critic", critic_node)
    graph.add_node("finish", finish_node)

    # Define edges
    graph.add_edge(START, "plan")

    graph.add_conditional_edges(
        "plan",
        route_after_plan,
        {
            "dispatch": "dispatch",
            "write": "write",
        },
    )

    graph.add_conditional_edges(
        "dispatch",
        route_after_dispatch,
        {
            "research": "research",
            "write": "write",
        },
    )

    graph.add_edge("research", "review")

    graph.add_conditional_edges(
        "review",
        route_after_review,
        {
            "research": "research",
            "dispatch": "dispatch",
            "write": "write",
        },
    )

    graph.add_edge("write", "critic")
    graph.add_edge("critic", "finish")
    graph.add_edge("finish", END)

    interrupt_nodes = ["write"] if interrupt_before_write else []
    return graph.compile(checkpointer=checkpointer, interrupt_before=interrupt_nodes)
