import asyncio
import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, status
from fastapi.responses import StreamingResponse

from app.api.auth import check_research_rate_limit, verify_api_key
from app.api.events import event_broker
from app.graph.builder import build_research_graph
from app.models.enums import RunStatus
from app.models.schemas import (
    Report,
    ResearchRequest,
    ResearchRunResponse,
    ResearchStatusResponse,
)
from app.persistence.checkpointer import checkpointer
from app.persistence.redis_store import redis_store
from app.services.citation_validator import validate_citations

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/research", tags=["Research Workflows"])


@router.post("", response_model=ResearchRunResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_research_run(
    request: ResearchRequest,
    caller_id: str = Depends(verify_api_key),
) -> ResearchRunResponse:
    """
    Initializes a new multi-agent research workflow.
    Returns immediately with run_id while agents execute asynchronously in the background.
    """
    check_research_rate_limit(caller_id)
    run_id = f"run_{uuid.uuid4().hex[:12]}"

    initial_state = {
        "run_id": run_id,
        "question": request.question,
        "status": RunStatus.PLANNING,
        "start_time": None,  # Will be set upon execution
        "total_tokens": 0,
        "total_usd_cost": 0.0,
        "step_index": 0,
        "require_approval": request.require_approval,
        "approved": False,
        "cancelled": False,
    }

    redis_store.save_run_state(run_id, initial_state)
    event_broker.publish_event(run_id, "initialized", {"run_id": run_id, "question": request.question})

    # Dispatch background execution
    asyncio.create_task(_execute_research_workflow(run_id, request.question, request.require_approval))

    return ResearchRunResponse(
        run_id=run_id,
        status=RunStatus.QUEUED,
        message="Research run dispatched. Query status at GET /research/{id} or stream events at GET /research/{id}/events.",
    )


@router.get("/{run_id}", response_model=ResearchStatusResponse)
async def get_research_status(
    run_id: str,
    caller_id: str = Depends(verify_api_key),
) -> ResearchStatusResponse:
    """Retrieves current workflow status, progress metrics, and final report upon completion."""
    state = redis_store.get_run_state(run_id)
    if not state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Research run '{run_id}' not found.",
        )

    # Check for completed report
    final_report = None
    raw_report_json = redis_store.get_final_report(run_id)
    if raw_report_json:
        try:
            final_report = Report.model_validate_json(raw_report_json)
        except Exception:
            final_report = None

    plan_dict = state.get("plan")
    total_sq = len(plan_dict.get("sub_questions", [])) if plan_dict and isinstance(plan_dict, dict) else 0
    completed_sq = len(state.get("completed_subquestion_ids", []))

    return ResearchStatusResponse(
        run_id=run_id,
        status=state.get("status", RunStatus.QUEUED),
        progress_subquestions=completed_sq,
        total_subquestions=total_sq,
        total_tokens=state.get("total_tokens", 0),
        total_cost_usd=state.get("total_usd_cost", 0.0),
        current_node=state.get("current_node"),
        report=final_report,
        error=state.get("error"),
        partial=state.get("partial", False),
        degradation_reason=state.get("degradation_reason"),
    )


@router.get("/{run_id}/events")
@router.get("/{run_id}/stream")
async def stream_research_events(
    run_id: str,
    last_event_id: str | None = Header(None, alias="Last-Event-ID"),
    caller_id: str = Depends(verify_api_key),
):
    """
    Server-Sent Events (SSE) live progress stream.
    Supports reconnection playback via the Last-Event-ID header.
    Accessible via both /events and /stream (for frontend compatibility).
    """
    last_id_num = None
    if last_event_id:
        try:
            last_id_num = int(last_event_id)
        except ValueError:
            pass

    return StreamingResponse(
        event_broker.event_generator(run_id, last_id_num),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/{run_id}/trace")
async def get_research_trace(
    run_id: str,
    caller_id: str = Depends(verify_api_key),
) -> dict[str, Any]:
    """Retrieves full ordered step audit log with token, cost, latency, and tool invocations."""
    raw_steps = redis_store.get_trace_steps(run_id)
    total_tokens = sum(s.get("total_tokens", 0) for s in raw_steps)
    total_cost = sum(s.get("usd_cost", 0.0) for s in raw_steps)
    total_latency = sum(s.get("latency_ms", 0.0) for s in raw_steps)

    return {
        "run_id": run_id,
        "total_steps": len(raw_steps),
        "total_tokens": total_tokens,
        "total_cost_usd": round(total_cost, 6),
        "total_latency_ms": round(total_latency, 2),
        "steps": raw_steps,
    }


@router.post("/{run_id}/approve", status_code=status.HTTP_200_OK)
async def approve_research_run(
    run_id: str,
    caller_id: str = Depends(verify_api_key),
) -> dict[str, str]:
    """Human approval gate: Approves a paused research run and resumes synthesis."""
    state = redis_store.get_run_state(run_id)
    if not state:
        raise HTTPException(status_code=404, detail="Run not found")

    event_broker.publish_event(run_id, "approved", {"message": "Human approval granted; resuming synthesis."})
    asyncio.create_task(_resume_approved_workflow(run_id))

    return {"status": "approved", "message": f"Run '{run_id}' approved. Resuming synthesis."}


@router.post("/{run_id}/reject", status_code=status.HTTP_200_OK)
async def reject_research_run(
    run_id: str,
    caller_id: str = Depends(verify_api_key),
) -> dict[str, str]:
    """Human approval gate: Rejects synthesis and marks the run cancelled."""
    redis_store.save_run_state(run_id, {"status": RunStatus.CANCELLED, "message": "Rejected by reviewer"})
    event_broker.publish_event(run_id, "cancelled", {"reason": "Rejected by human reviewer"})
    return {"status": "cancelled", "message": f"Run '{run_id}' rejected and cancelled."}


@router.post("/{run_id}/cancel", status_code=status.HTTP_200_OK)
async def cancel_research_run(
    run_id: str,
    caller_id: str = Depends(verify_api_key),
) -> dict[str, str]:
    """Cancels an ongoing research workflow."""
    redis_store.save_run_state(run_id, {"status": RunStatus.CANCELLED})
    event_broker.publish_event(run_id, "cancelled", {"reason": "Explicit user cancellation"})
    return {"status": "cancelled", "message": f"Run '{run_id}' marked cancelled."}


# --- Background Execution Helpers ---


async def _execute_research_workflow(run_id: str, question: str, require_approval: bool) -> None:
    """Executes the workflow graph, records events, validates citations, and persists reports."""
    locked = redis_store.acquire_lock(run_id, timeout_seconds=60.0)
    if not locked:
        return

    try:
        import time

        curr = redis_store.get_run_state(run_id)
        if curr and curr.get("status") == RunStatus.CANCELLED:
            return

        graph = build_research_graph(checkpointer=checkpointer, interrupt_before_write=require_approval)
        config = {"configurable": {"thread_id": run_id}}

        initial_state = {
            "run_id": run_id,
            "question": question,
            "status": RunStatus.PLANNING,
            "start_time": time.time(),
            "total_tokens": 0,
            "total_usd_cost": 0.0,
            "step_index": 0,
            "require_approval": require_approval,
            "approved": False,
            "cancelled": False,
        }

        loop = asyncio.get_running_loop()
        final_state = await loop.run_in_executor(None, lambda: graph.invoke(initial_state, config=config))

        # Check if paused before write
        curr_state = graph.get_state(config)
        if curr_state.next and "write" in curr_state.next:
            redis_store.save_run_state(
                run_id,
                {
                    "run_id": run_id,
                    "status": RunStatus.AWAITING_APPROVAL,
                    "completed_subquestion_ids": curr_state.values.get("completed_subquestion_ids", []),
                    "plan": curr_state.values.get("plan").model_dump() if curr_state.values.get("plan") else None,
                },
            )
            event_broker.publish_event(run_id, "approval_requested", {"run_id": run_id})
            redis_store.release_lock(run_id)
            return

        # Complete run
        await _finalize_completed_state(run_id, final_state)

    except Exception as e:
        logger.exception(f"Error in research run {run_id}: {e}")
        redis_store.save_run_state(run_id, {"status": RunStatus.FAILED, "error": str(e)})
        event_broker.publish_event(run_id, "failed", {"error": str(e)})
    finally:
        redis_store.release_lock(run_id)


async def _resume_approved_workflow(run_id: str) -> None:
    """Resumes an approved workflow from the write interrupt."""
    redis_store.release_lock(run_id)
    locked = redis_store.acquire_lock(run_id, timeout_seconds=60.0)
    if not locked:
        logger.warning(f"Could not acquire lock to resume run {run_id}")
        return

    try:
        graph = build_research_graph(checkpointer=checkpointer, interrupt_before_write=False)
        config = {"configurable": {"thread_id": run_id}}

        loop = asyncio.get_running_loop()
        final_state = await loop.run_in_executor(None, lambda: graph.invoke(None, config=config))

        await _finalize_completed_state(run_id, final_state)

    except Exception as e:
        logger.exception(f"Error resuming run {run_id}: {e}")
        redis_store.save_run_state(run_id, {"status": RunStatus.FAILED, "error": str(e)})
    finally:
        redis_store.release_lock(run_id)


async def _finalize_completed_state(run_id: str, final_state: dict[str, Any]) -> None:
    """Validates citations, updates coverage, persists final report, and emits completion event."""
    report: Report | None = final_state.get("final_report")
    findings = final_state.get("findings", [])

    if report:
        # Validate citation grounding
        validated_report, coverage, warnings = validate_citations(
            report=report,
            verified_findings=findings,
        )

        final_status = RunStatus.COMPLETED
        if coverage < 0.70:
            final_status = RunStatus.NEEDS_REVIEW

        validated_report_json = validated_report.model_dump_json()
        redis_store.save_final_report(run_id, validated_report_json)

        redis_store.save_run_state(
            run_id,
            {
                "run_id": run_id,
                "status": final_status,
                "total_tokens": final_state.get("total_tokens", 0),
                "total_usd_cost": final_state.get("total_usd_cost", 0.0),
                "partial": final_state.get("partial", False),
                "degradation_reason": final_state.get("degradation_reason"),
            },
        )

        # Store traces
        for trace in final_state.get("trace_log", []):
            if hasattr(trace, "model_dump"):
                redis_store.append_trace_step(run_id, trace.model_dump())
            elif isinstance(trace, dict):
                redis_store.append_trace_step(run_id, trace)

        event_broker.publish_event(
            run_id,
            "completed",
            {
                "status": final_status,
                "citation_coverage": coverage,
                "warnings": warnings,
            },
        )
