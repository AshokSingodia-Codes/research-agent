from typing import Any

from langgraph.graph import END, START, StateGraph

from app.graph.nodes import (
    critic_node,
    dispatch_node,
    finish_node,
    plan_node,
    research_one_node,
    review_all_node,
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
    Constructs the LangGraph multi-agent research workflow with Send() parallel fan-out:
    plan -> dispatch -> [Send(research_one)] -> review_all -> (loop | write) -> critic -> finish -> END
    """
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("plan", plan_node)
    graph.add_node("dispatch", dispatch_node)
    graph.add_node("research_one", research_one_node)
    graph.add_node("review_all", review_all_node)
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
        ["research_one", "write"],
    )

    graph.add_edge("research_one", "review_all")

    graph.add_conditional_edges(
        "review_all",
        route_after_review,
        {
            "dispatch": "dispatch",
            "write": "write",
        },
    )

    graph.add_edge("write", "critic")
    graph.add_edge("critic", "finish")
    graph.add_edge("finish", END)

    interrupt_nodes = ["write"] if interrupt_before_write else []
    return graph.compile(checkpointer=checkpointer, interrupt_before=interrupt_nodes)
