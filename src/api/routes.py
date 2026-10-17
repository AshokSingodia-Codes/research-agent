import asyncio
import json
import uuid
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.models.schemas import ResearchReport, RunStatus, StepTrace
from src.graph.redis_store import redis_store
from src.services.runner import run_research_workflow_async, register_stream_queue, unregister_stream_queue

router = APIRouter()


class ResearchRequest(BaseModel):
    topic: str = Field(..., min_length=3, description="The research topic or question to investigate")
    max_subquestions: Optional[int] = Field(None, ge=1, le=8, description="Optional override for max sub-questions")


class ResearchInitResponse(BaseModel):
    run_id: str
    status: str
    message: str


class TraceSummaryResponse(BaseModel):
    run_id: str
    total_steps: int
    total_tokens: int
    total_latency_ms: float
    steps: List[StepTrace]


class ResearchStatusResponse(BaseModel):
    run_id: str
    status: str
    progress_percentage: float
    completed_subquestions: int
    total_subquestions: int
    total_tokens: int
    current_node: Optional[str] = None
    report: Optional[ResearchReport] = None
    error: Optional[str] = None


@router.post("/research", response_model=ResearchInitResponse, status_code=202)
async def start_research(request: ResearchRequest, background_tasks: BackgroundTasks):
    """
    Initiates a new multi-agent research run.
    Returns immediately with run_id while the agents execute in the background.
    """
    run_id = f"run_{uuid.uuid4().hex[:12]}"

    # Save initial queued state in Redis store
    redis_store.save_run_state(
        run_id=run_id,
        status="queued",
        current_node="initialization",
        total_tokens=0,
        total_subquestions=0
    )

    # Dispatch background execution
    background_tasks.add_task(run_research_workflow_async, run_id, request.topic)

    return ResearchInitResponse(
        run_id=run_id,
        status="queued",
        message="Research workflow dispatched. Inspect status at GET /research/{id} or stream events at GET /research/{id}/stream."
    )


@router.get("/research/{run_id}", response_model=ResearchStatusResponse)
async def get_research_status(run_id: str):
    """
    Returns real-time run progress, metrics, and the completed ResearchReport with verified citations.
    """
    status_info = redis_store.get_run_status(run_id)
    if not status_info:
        raise HTTPException(status_code=404, detail=f"Research run '{run_id}' not found")

    report_obj = None
    raw_report_json = redis_store.get_final_report(run_id)
    if raw_report_json:
        try:
            report_obj = ResearchReport.model_validate_json(raw_report_json)
        except Exception:
            report_obj = None

    return ResearchStatusResponse(
        run_id=run_id,
        status=status_info.status,
        progress_percentage=status_info.progress_percentage,
        completed_subquestions=status_info.completed_subquestions,
        total_subquestions=status_info.total_subquestions,
        total_tokens=status_info.total_tokens,
        current_node=status_info.current_node,
        report=report_obj,
        error=status_info.error
    )


@router.get("/research/{run_id}/trace", response_model=TraceSummaryResponse)
async def get_research_trace(run_id: str):
    """
    Returns the step-level audit log: which agent ran, inputs, outputs, tool calls,
    tokens, and latency per step. Interview-proof debugging trace.
    """
    traces = redis_store.get_trace_log(run_id)
    if not traces:
        # Check if the run even exists
        if not redis_store.get_run_status(run_id):
            raise HTTPException(status_code=404, detail=f"Research run '{run_id}' not found")

    total_latency = sum(t.latency_ms for t in traces)
    total_tokens = sum(t.tokens_used for t in traces)

    return TraceSummaryResponse(
        run_id=run_id,
        total_steps=len(traces),
        total_tokens=total_tokens,
        total_latency_ms=round(total_latency, 2),
        steps=traces
    )


@router.get("/research/{run_id}/stream")
async def stream_research_progress(run_id: str, request: Request):
    """
    Streams Server-Sent Events (SSE) in real-time as agent steps complete,
    preventing blind spinners.
    """
    status_info = redis_store.get_run_status(run_id)
    if not status_info:
        raise HTTPException(status_code=404, detail=f"Research run '{run_id}' not found")

    queue = register_stream_queue(run_id)

    async def event_generator():
        try:
            # Emit initial current status
            yield f"data: {json.dumps({'type': 'init', 'status': status_info.status, 'progress': status_info.progress_percentage})}\n\n"

            while True:
                if await request.is_disconnected():
                    break

                try:
                    event = await asyncio.wait_for(queue.get(), timeout=2.0)
                    if event is None:
                        # Stream closed
                        yield f"data: {json.dumps({'type': 'stream_end', 'message': 'Run execution finished'})}\n\n"
                        break
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    # Heartbeat ping to keep SSE connection alive
                    yield ": ping\n\n"
        finally:
            unregister_stream_queue(run_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
