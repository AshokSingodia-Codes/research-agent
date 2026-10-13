import asyncio
import logging
import time
from typing import Any

from app.graph.builder import build_research_graph
from app.graph.state import AgentState
from app.models.enums import RunStatus
from app.persistence.checkpointer import RedisCheckpointSaver
from app.persistence.redis_store import redis_store

logger = logging.getLogger(__name__)

# Shared background checkpointer instance
checkpointer = RedisCheckpointSaver()


class BackgroundWorker:
    """
    Durable Background Worker:
    - Runs research workflows asynchronously outside request handlers
    - Enforces distributed locks per run to prevent multi-worker collisions
    - Automatically discovers and resumes interrupted/running workflows on startup
    """

    def __init__(self):
        self._running_tasks: dict[str, asyncio.Task] = {}

    def start_research_task(
        self,
        run_id: str,
        question: str,
        require_approval: bool = False,
        budget_overrides: dict[str, Any] | None = None,
    ) -> None:
        # Save initial queued state
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
        redis_store.save_run_state(run_id, initial_state)

        # Dispatch background execution
        loop = asyncio.get_event_loop()
        task = loop.create_task(self.run_workflow(run_id, initial_state, require_approval))
        self._running_tasks[run_id] = task

    async def run_workflow(
        self,
        run_id: str,
        initial_state: dict[str, Any] | None = None,
        require_approval: bool = False,
    ) -> AgentState | None:
        """Executes the research graph, acquiring a distributed lock and checkpointing state."""
        # 1. Acquire distributed lock
        locked = redis_store.acquire_lock(run_id, timeout_seconds=60.0)
        if not locked:
            logger.warning(f"Failed to acquire distributed lock for {run_id}; another worker is running it.")
            return None

        try:
            graph = build_research_graph(checkpointer=checkpointer, interrupt_before_write=require_approval)
            config = {"configurable": {"thread_id": run_id}}

            state_input = initial_state
            # Check if there is already a checkpoint in Redis for resumption
            existing_checkpoint = checkpointer.get_tuple(config)
            if existing_checkpoint is not None and initial_state is None:
                # Resuming from crashed state
                state_input = None

            loop = asyncio.get_running_loop()
            final_state = await loop.run_in_executor(None, lambda: graph.invoke(state_input, config=config))

            # Check if execution paused for human approval
            curr_state = graph.get_state(config)
            if curr_state.next and "write" in curr_state.next:
                redis_store.save_run_state(
                    run_id,
                    {
                        "run_id": run_id,
                        "status": RunStatus.AWAITING_APPROVAL,
                        "message": "Graph paused: human approval required before synthesis.",
                    },
                )
                return final_state

            # If completed, persist final report and release lock
            if final_state and final_state.get("final_report"):
                report = final_state["final_report"]
                redis_store.save_final_report(run_id, report.model_dump_json())
                redis_store.save_run_state(
                    run_id,
                    {
                        "run_id": run_id,
                        "status": final_state.get("status", RunStatus.COMPLETED),
                        "total_tokens": final_state.get("total_tokens", 0),
                        "total_usd_cost": final_state.get("total_usd_cost", 0.0),
                    },
                )

            return final_state

        except Exception as e:
            logger.exception(f"Error executing run {run_id}: {e}")
            redis_store.save_run_state(
                run_id,
                {"run_id": run_id, "status": RunStatus.FAILED, "error": str(e)},
            )
            return None
        finally:
            redis_store.release_lock(run_id)
            self._running_tasks.pop(run_id, None)

    async def resume_interrupted_runs(self) -> int:
        """Finds any runs in running / incomplete state and resumes them from last checkpoint."""
        running_ids = redis_store.list_running_runs()
        resumed_count = 0
        for run_id in running_ids:
            state = redis_store.get_run_state(run_id)
            if state and state.get("status") in (
                RunStatus.PLANNING,
                RunStatus.RESEARCHING,
                RunStatus.REVIEWING,
                RunStatus.WRITING,
            ):
                logger.info(f"Resuming interrupted research run: {run_id}")
                asyncio.create_task(self.run_workflow(run_id, None, False))
                resumed_count += 1
        return resumed_count


worker = BackgroundWorker()
