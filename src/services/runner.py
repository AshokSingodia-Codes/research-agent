import asyncio
import json
import logging
import time
from typing import Any, AsyncGenerator, Dict, Optional
from langgraph.checkpoint.memory import MemorySaver

from src.graph.builder import build_research_graph
from src.graph.redis_store import redis_store
from src.services.validator import validate_report_citations
from src.models.schemas import ResearchReport

logger = logging.getLogger(__name__)

# In-memory checkpointer shared across runs for instant local resumption
_shared_checkpointer = MemorySaver()
# Active event queues for live streaming
_event_queues: Dict[str, asyncio.Queue] = {}


async def run_research_workflow_async(run_id: str, topic: str) -> None:
    """
    Executes the research graph asynchronously in the background,
    streaming progress events and persisting state to Redis.
    """
    queue = _event_queues.get(run_id)

    async def emit_event(event_type: str, data: Dict[str, Any]):
        if queue:
            await queue.put({"type": event_type, "data": data, "timestamp": time.time()})

    try:
        redis_store.save_run_state(
            run_id=run_id,
            status="in_progress",
            current_node="planner",
            total_tokens=0,
            total_subquestions=0
        )
        await emit_event("status", {"status": "in_progress", "node": "planner", "message": "Planning research questions..."})

        graph = build_research_graph(checkpointer=_shared_checkpointer)
        config = {"configurable": {"thread_id": run_id}}

        initial_state = {
            "run_id": run_id,
            "research_topic": topic,
            "start_time_epoch": time.time(),
            "total_tokens": 0,
            "total_searches": 0,
            "trace_log": [],
        }

        # Stream node execution
        loop = asyncio.get_running_loop()
        final_state = await loop.run_in_executor(None, lambda: graph.invoke(initial_state, config=config))

        report: Optional[ResearchReport] = final_state.get("report")
        if report:
            # Validate citations against findings provenance
            validated_report, warnings = validate_report_citations(
                report=report,
                evidence_records=final_state.get("findings", [])
            )
            final_report_json = validated_report.model_dump_json()
            redis_store.save_final_report(run_id, final_report_json)

            await emit_event("complete", {
                "status": "completed",
                "citations_verified": len(validated_report.citations),
                "sections": len(validated_report.sections),
                "warnings": warnings
            })

    except Exception as e:
        logger.exception(f"Error in research workflow {run_id}: {e}")
        redis_store.save_run_state(
            run_id=run_id,
            status="failed",
            error=str(e)
        )
        await emit_event("error", {"error": str(e)})
    finally:
        if queue:
            await queue.put(None)  # Sentinel to close stream


def register_stream_queue(run_id: str) -> asyncio.Queue:
    """Registers or retrieves an existing event queue for live progress streaming."""
    if run_id not in _event_queues:
        _event_queues[run_id] = asyncio.Queue()
    return _event_queues[run_id]


def unregister_stream_queue(run_id: str) -> None:
    """Cleans up the queue once streaming completes."""
    _event_queues.pop(run_id, None)
