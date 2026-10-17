import time
from typing import Any, Dict, Literal
from langgraph.graph import StateGraph, START, END

from src.config import settings
from src.models.state import AgentState
from src.models.schemas import ResearchPlan, SubQuestion, StepTrace
from src.agents.planner import plan_research
from src.agents.researcher import research_subquestion
from src.agents.writer import write_research_report
from src.agents.supervisor import review_research_output
from src.graph.checkpointer import get_checkpointer
from src.graph.redis_store import redis_store


# --- GRAPH NODES ---

def planner_node(state: AgentState) -> Dict[str, Any]:
    """Planner Node: Decomposes research topic into structured SubQuestions."""
    # Idempotency check: if plan already exists in state, skip re-computation
    if state.get("plan") is not None:
        return {}

    topic = state.get("research_topic", "")
    plan, trace = plan_research(topic, max_subquestions=settings.max_subquestions)
    new_tokens = state.get("total_tokens", 0) + trace.tokens_used

    if run_id := state.get("run_id"):
        redis_store.append_trace_step(run_id, trace)
        redis_store.save_run_state(
            run_id=run_id,
            status="in_progress",
            current_node="planner",
            plan_dict=plan.model_dump(),
            completed_sq_ids=[],
            total_tokens=new_tokens,
            total_subquestions=len(plan.sub_questions)
        )

    return {
        "plan": plan,
        "current_subquestion_index": 0,
        "completed_subquestion_ids": [],
        "retries_by_subquestion": {},
        "findings": [],
        "findings_by_subquestion": {},
        "trace_log": [trace],
        "total_tokens": new_tokens,
        "status": "in_progress",
    }


def researcher_node(state: AgentState) -> Dict[str, Any]:
    """Researcher Node: Executes live search and page extraction for the current sub-question."""
    plan = state.get("plan")
    if not plan or not plan.sub_questions:
        return {}

    idx = state.get("current_subquestion_index", 0)
    if idx >= len(plan.sub_questions):
        return {}

    sub_q = plan.sub_questions[idx]
    feedback = state.get("last_reviewer_feedback")

    # Run researcher specialist
    output, trace = research_subquestion(
        sub_question=sub_q,
        feedback=feedback,
        max_searches=settings.max_searches_per_subquestion
    )

    # Accumulate findings
    new_findings = output.findings
    findings_by_sq = dict(state.get("findings_by_subquestion", {}))
    existing_for_sq = findings_by_sq.get(sub_q.id, [])
    findings_by_sq[sub_q.id] = existing_for_sq + new_findings
    new_tokens = state.get("total_tokens", 0) + trace.tokens_used

    if run_id := state.get("run_id"):
        redis_store.append_trace_step(run_id, trace)
        redis_store.save_run_state(
            run_id=run_id,
            status="in_progress",
            current_node="researcher",
            completed_sq_ids=state.get("completed_subquestion_ids", []),
            total_tokens=new_tokens,
            total_subquestions=len(plan.sub_questions)
        )

    return {
        "findings": new_findings,  # Reducer merges into cumulative findings
        "findings_by_subquestion": findings_by_sq,
        "total_searches": state.get("total_searches", 0) + output.searches_executed,
        "total_tokens": new_tokens,
        "trace_log": [trace],
    }


def supervisor_node(state: AgentState) -> Dict[str, Any]:
    """Supervisor Node: Reviews researcher output against plan & enforces budgets."""
    plan = state.get("plan")
    if not plan or not plan.sub_questions:
        return {"status": "completed"}

    idx = state.get("current_subquestion_index", 0)
    total_sub_q = len(plan.sub_questions)

    # If all sub-questions are already processed
    if idx >= total_sub_q:
        return {}

    sub_q = plan.sub_questions[idx]
    retries_map = dict(state.get("retries_by_subquestion", {}))
    current_retries = retries_map.get(sub_q.id, 0)

    # Gather latest findings for this sub-question
    findings_by_sq = state.get("findings_by_subquestion", {})
    sq_findings = findings_by_sq.get(sub_q.id, [])

    # Synthetic researcher output representation for review
    from src.models.schemas import ResearcherOutput, SearchStatus
    research_output = ResearcherOutput(
        sub_question_id=sub_q.id,
        findings=sq_findings,
        search_status=SearchStatus(status="success" if sq_findings else "no_results")
    )

    elapsed = time.time() - state.get("start_time_epoch", time.time())
    is_last = (idx >= total_sub_q - 1)

    review, trace = review_research_output(
        sub_question=sub_q,
        research_output=research_output,
        current_retry_count=current_retries,
        total_tokens=state.get("total_tokens", 0),
        elapsed_seconds=elapsed,
        is_last_subquestion=is_last
    )

    new_tokens = state.get("total_tokens", 0) + trace.tokens_used
    updates: Dict[str, Any] = {
        "trace_log": [trace],
        "total_tokens": new_tokens,
    }

    completed_ids = list(state.get("completed_subquestion_ids", []))

    if review.next_action == "retry_subquestion":
        retries_map[sub_q.id] = current_retries + 1
        updates["retries_by_subquestion"] = retries_map
        updates["last_reviewer_feedback"] = review.feedback
    elif review.next_action == "continue_research":
        if sub_q.id not in completed_ids:
            completed_ids.append(sub_q.id)
        updates["completed_subquestion_ids"] = completed_ids
        updates["current_subquestion_index"] = idx + 1
        updates["last_reviewer_feedback"] = None
    elif review.next_action in ("synthesize_report", "abort_budget_exceeded"):
        if sub_q.id not in completed_ids:
            completed_ids.append(sub_q.id)
        updates["completed_subquestion_ids"] = completed_ids
        updates["current_subquestion_index"] = total_sub_q  # marks all done
        updates["last_reviewer_feedback"] = None
        if review.next_action == "abort_budget_exceeded":
            updates["budget_exhausted"] = True

    if run_id := state.get("run_id"):
        redis_store.append_trace_step(run_id, trace)
        redis_store.save_run_state(
            run_id=run_id,
            status="in_progress",
            current_node="supervisor",
            completed_sq_ids=completed_ids,
            total_tokens=new_tokens,
            total_subquestions=total_sub_q
        )

    return updates


def writer_node(state: AgentState) -> Dict[str, Any]:
    """Writer Node: Synthesizes cumulative findings into fully cited ResearchReport."""
    # Idempotency check: if report already generated, skip
    if state.get("report") is not None:
        return {"status": "completed"}

    topic = state.get("research_topic", "")
    all_findings = state.get("findings", [])
    plan = state.get("plan")

    completed_ids = set(state.get("completed_subquestion_ids", []))
    unresolved = []
    if plan:
        for sq in plan.sub_questions:
            if sq.id not in completed_ids:
                unresolved.append(sq.question)

    report, trace = write_research_report(
        topic=topic,
        findings=all_findings,
        unresolved_questions=unresolved
    )

    new_tokens = state.get("total_tokens", 0) + trace.tokens_used

    if run_id := state.get("run_id"):
        redis_store.append_trace_step(run_id, trace)
        redis_store.save_final_report(run_id, report.model_dump_json())
        redis_store.save_run_state(
            run_id=run_id,
            status="completed",
            current_node="writer",
            completed_sq_ids=list(completed_ids),
            total_tokens=new_tokens,
            total_subquestions=len(plan.sub_questions) if plan else 0
        )

    return {
        "report": report,
        "trace_log": [trace],
        "total_tokens": new_tokens,
        "status": "completed"
    }


# --- CONDITIONAL ROUTER ---

def supervisor_router(state: AgentState) -> Literal["researcher", "writer"]:
    """Determines whether another researcher step is needed or to proceed to report writing."""
    plan = state.get("plan")
    if not plan or not plan.sub_questions:
        return "writer"

    idx = state.get("current_subquestion_index", 0)
    total_sub_q = len(plan.sub_questions)

    # If all sub-questions are completed or budget exhausted, proceed to writer
    if idx >= total_sub_q or state.get("budget_exhausted", False):
        return "writer"

    return "researcher"


# --- GRAPH ASSEMBLY ---

def build_research_graph(checkpointer=None):
    """Compiles the multi-agent LangGraph workflow."""
    workflow = StateGraph(AgentState)

    # Add agent specialist and supervisor nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("writer", writer_node)

    # Establish edge topology
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "supervisor")
    workflow.add_edge("researcher", "supervisor")

    workflow.add_conditional_edges(
        "supervisor",
        supervisor_router,
        {
            "researcher": "researcher",
            "writer": "writer"
        }
    )

    workflow.add_edge("writer", END)

    if checkpointer is None:
        checkpointer = get_checkpointer()

    return workflow.compile(checkpointer=checkpointer)
