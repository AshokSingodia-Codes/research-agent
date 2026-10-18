import asyncio
import json
import logging
import os
import platform
import signal
import sys
from typing import Any

from app.api.events import event_broker
from app.config import settings
from app.graph.builder import build_research_graph
from app.models.enums import RunStatus
from app.models.schemas import Report
from app.persistence.checkpointer import RedisCheckpointSaver
from app.persistence.concurrency import CrossProcessConcurrencyLimiter
from app.persistence.redis_store import redis_store
from app.persistence.stream_adapter import RedisStreamAdapter
from app.services.citation_validator import validate_citations

logger = logging.getLogger(__name__)

checkpointer = RedisCheckpointSaver()


class ResearchStreamWorker:
    """
    Production Redis Streams Research Worker:
    - Consumer group: 'research-workers'
    - Consumer ID: hostname-pid
    - XREADGROUP BLOCK via asyncio.to_thread
    - Owner-token distributed run locks with renewal heartbeat
    - Global MAX_CONCURRENT_RUNS cap via CrossProcessConcurrencyLimiter
    - Idle reclaim via XAUTOCLAIM and startup pending list scan
    - DLQ routing after 3 delivery attempts to research:jobs:dlq
    - Handles SIGINT / SIGTERM leaving in-flight unacked for resumption
    - Strict fake-mode single worker enforcement
    """

    def __init__(
        self,
        consumer_name: str | None = None,
        stream: str = "research:jobs",
        group: str = "research-workers",
        dlq_stream: str = "research:jobs:dlq",
    ):
        self.stream = stream
        self.group = group
        self.dlq_stream = dlq_stream
        self.consumer_name = consumer_name or f"{platform.node()}-{os.getpid()}"
        self.stream_adapter = RedisStreamAdapter()
        self._stop_event = asyncio.Event()
        self._stopping = False
        self._concurrency_limiter = CrossProcessConcurrencyLimiter(
            key="research:worker:semaphore",
            max_concurrent=settings.max_concurrent_runs,
            fallback_on_unsupported=True,
        )
        self._active_fake_key = "research:worker:active_fake_worker"
        self._fake_registered = False

    @property
    def client(self) -> Any:
        return redis_store.get_client()

    def register_fake_worker(self) -> None:
        """Enforces that fake mode supports ONE worker only."""
        if not settings.use_fake_redis:
            return

        client = self.client
        acquired = client.set(self._active_fake_key, self.consumer_name, nx=True, ex=15)
        if not acquired:
            curr = client.get(self._active_fake_key)
            if curr and curr != self.consumer_name:
                raise RuntimeError(
                    f"Fake mode supports ONE worker only; a second worker ({self.consumer_name}) "
                    f"refuses to start while worker '{curr}' is active. "
                    "The shared fake-redis file is racy."
                )
        self._fake_registered = True

    def unregister_fake_worker(self) -> None:
        if settings.use_fake_redis and self._fake_registered:
            try:
                curr = self.client.get(self._active_fake_key)
                if curr == self.consumer_name:
                    self.client.delete(self._active_fake_key)
            except Exception:
                pass
            self._fake_registered = False

    def stop(self) -> None:
        """Signals worker loop to stop gracefully."""
        self._stopping = True
        self._stop_event.set()

    def _setup_signal_handlers(self) -> None:
        """Configures SIGINT and SIGTERM handlers to stop gracefully."""
        def handler(sig, frame):
            logger.info(f"Worker received signal {sig}; initiating graceful stop...")
            self.stop()

        try:
            signal.signal(signal.SIGINT, handler)
            signal.signal(signal.SIGTERM, handler)
        except (ValueError, AttributeError):
            # Not in main thread or unsupported platform
            pass

    async def _heartbeat_loop(self, run_id: str, owner_token: str, stop_hb: asyncio.Event) -> bool:
        """Periodically renews the lock for run_id as long as owner_token matches."""
        interval = max(1.0, min(5.0, settings.worker_visibility_timeout / 3.0))
        client = self.client
        lock_key = f"research:lock:{run_id}"

        while not stop_hb.is_set() and not self._stop_event.is_set():
            await asyncio.sleep(interval)
            if stop_hb.is_set():
                break
            try:
                # Check owner token
                curr = client.get(lock_key)
                if curr == owner_token:
                    client.expire(lock_key, int(settings.worker_visibility_timeout))
                else:
                    logger.warning(f"Heartbeat renewal failed for run {run_id}: lock owner mismatch ({curr} != {owner_token})")
                    return False

                if settings.use_fake_redis and self._fake_registered:
                    client.expire(self._active_fake_key, 15)
            except Exception as e:
                logger.warning(f"Error during heartbeat renewal for {run_id}: {e}")
                return False
        return True

    def _get_delivery_count(self, entry_id: str) -> int:
        """Retrieves times_delivered for entry_id from consumer group pending list."""
        try:
            pending = self.client.xpending_range(self.stream, self.group, min=entry_id, max=entry_id, count=1)
            if pending and len(pending) > 0:
                item = pending[0]
                if isinstance(item, dict):
                    return item.get("times_delivered", 1)
        except Exception as e:
            logger.debug(f"Unable to read times_delivered for {entry_id}: {e}")
        return 1

    async def _move_to_dlq(self, entry_id: str, run_id: str, error_msg: str, payload: dict[str, Any]) -> None:
        """Moves a poison/failed message to dead letter queue after max retries."""
        logger.error(f"Moving job {entry_id} (run_id: {run_id}) to DLQ: {error_msg}")
        client = self.client
        dlq_data = {
            "run_id": run_id,
            "entry_id": entry_id,
            "error": error_msg,
            "payload": json.dumps(payload, default=str),
            "consumer": self.consumer_name,
            "timestamp": str(asyncio.get_event_loop().time()),
        }
        try:
            client.xadd(self.dlq_stream, dlq_data)
            client.incr("research:metrics:dlq_total")
            redis_store.save_run_state(run_id, {"status": RunStatus.FAILED, "error": error_msg})
            client.xack(self.stream, self.group, entry_id)
            client.xdel(self.stream, entry_id)
        except Exception as e:
            logger.warning(f"Failed to record DLQ entry for {entry_id}: {e}")

    async def process_job(self, entry_id: str, payload: dict[str, Any]) -> bool:
        """
        Executes a single job from the stream.
        Enforces concurrency semaphore, owner-checked locks, renewal heartbeat,
        cancellation checks, and terminal state ACKs.
        """
        run_id = payload.get("run_id")
        if not run_id:
            logger.warning(f"Discarding malformed job entry {entry_id} with missing run_id")
            self.stream_adapter.ack_job(self.stream, self.group, entry_id)
            return True

        # Check delivery count: after 3 deliveries, move to DLQ
        deliveries = self._get_delivery_count(entry_id)
        if deliveries > 3:
            await self._move_to_dlq(
                entry_id=entry_id,
                run_id=run_id,
                error_msg=f"Exceeded maximum delivery attempts ({deliveries} > 3)",
                payload=payload,
            )
            return False

        # Check if run was cancelled while queued
        client = self.client
        is_cancelled = bool(client.get(f"research:cancel:{run_id}"))
        curr_state = redis_store.get_run_state(run_id)
        if curr_state and curr_state.get("status") in (RunStatus.CANCELLED, "cancelled"):
            is_cancelled = True

        if is_cancelled:
            logger.info(f"Job {entry_id} for run {run_id} was cancelled before execution.")
            redis_store.save_run_state(run_id, {"status": RunStatus.CANCELLED, "cancelled": True})
            self.stream_adapter.ack_job(self.stream, self.group, entry_id)
            return True

        # Acquire cross-process concurrency semaphore
        slot_acquired = self._concurrency_limiter.acquire()
        if not slot_acquired:
            logger.warning(f"Global concurrency cap reached ({settings.max_concurrent_runs}); deferring job {entry_id}")
            return False

        # Acquire owner-checked distributed run lock (with short retry for immediate resume transitions)
        owner_token = None
        for _ in range(5):
            owner_token = redis_store.acquire_lock(run_id, timeout_seconds=settings.worker_visibility_timeout)
            if owner_token:
                break
            await asyncio.sleep(0.1)

        if not owner_token:
            logger.info(f"Another worker currently holds lock for run {run_id}; leaving job unacked for retry.")
            self._concurrency_limiter.release()
            return False

        stop_hb = asyncio.Event()
        hb_task = asyncio.create_task(self._heartbeat_loop(run_id, owner_token, stop_hb))

        job_type = payload.get("type", "initial")
        question = payload.get("question", "")
        require_approval = str(payload.get("require_approval", "false")).lower() in ("true", "1")

        try:
            if job_type == "resume":
                await self._execute_resume(run_id=run_id)
            else:
                await self._execute_initial(
                    run_id=run_id,
                    question=question,
                    require_approval=require_approval,
                    options=payload,
                )

            # Check if run paused at awaiting_approval: release lock BEFORE ACK so /approve cannot race it
            state_check = redis_store.get_run_state(run_id)
            if state_check and state_check.get("status") in (RunStatus.AWAITING_APPROVAL, "awaiting_approval"):
                stop_hb.set()
                if not hb_task.done():
                    hb_task.cancel()
                    try:
                        await hb_task
                    except (asyncio.CancelledError, Exception):
                        pass
                redis_store.release_lock(run_id, owner_token)

            # If execution reached this point, job completed, paused, or reached a terminal state: ACK IT!
            self.stream_adapter.ack_job(self.stream, self.group, entry_id)
            return True

        except asyncio.CancelledError:
            # Interrupted (e.g. graceful SIGINT / shutdown)
            logger.info(f"Worker execution cancelled mid-flight for run {run_id}; leaving job UNACKED for recovery.")
            raise
        except Exception as e:
            logger.exception(f"Unhandled exception processing job {entry_id} for run {run_id}: {e}")
            redis_store.save_run_state(run_id, {"status": RunStatus.FAILED, "error": str(e)})
            self.stream_adapter.ack_job(self.stream, self.group, entry_id)
            return True
        finally:
            stop_hb.set()
            if not hb_task.done():
                hb_task.cancel()
                try:
                    await hb_task
                except (asyncio.CancelledError, Exception):
                    pass
            redis_store.release_lock(run_id, owner_token)
            self._concurrency_limiter.release()

    async def _execute_initial(
        self,
        run_id: str,
        question: str,
        require_approval: bool,
        options: dict[str, Any],
    ) -> None:
        """Executes initial research workflow from plan to completion or approval interrupt."""
        import time

        graph = build_research_graph(checkpointer=checkpointer, interrupt_before_write=require_approval)
        config = {"configurable": {"thread_id": run_id}}

        existing = checkpointer.get_tuple(config)
        state_input = (
            None
            if existing
            else {
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
        )

        final_state = await graph.ainvoke(state_input, config=config)

        # Check if execution paused before write for human approval
        curr_state = await graph.aget_state(config)
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
            return

        await self._finalize_completed_state(run_id, final_state)

    async def _execute_resume(self, run_id: str) -> None:
        """Resumes an approved workflow from the write interrupt."""
        graph = build_research_graph(checkpointer=checkpointer, interrupt_before_write=False)
        config = {"configurable": {"thread_id": run_id}}

        final_state = await graph.ainvoke(None, config=config)
        await self._finalize_completed_state(run_id, final_state)

    async def _finalize_completed_state(self, run_id: str, final_state: dict[str, Any]) -> None:
        """Validates citations, updates coverage, persists final report, and emits completion event."""
        if not final_state:
            return

        status = final_state.get("status", RunStatus.COMPLETED)
        report: Report | None = final_state.get("final_report")
        findings = final_state.get("findings", [])

        if report:
            validated_report, coverage, warnings = validate_citations(
                report=report,
                verified_findings=findings,
            )
            final_status = RunStatus.COMPLETED
            if coverage < 0.70:
                final_status = RunStatus.NEEDS_REVIEW
            if final_state.get("partial"):
                final_status = RunStatus.COMPLETED  # partial is completed with partial flag

            validated_report_json = validated_report.model_dump_json()
            redis_store.save_final_report(run_id, validated_report_json)

            redis_store.save_run_state(
                run_id,
                {
                    "run_id": run_id,
                    "status": final_status,
                    "final_report": validated_report.model_dump(),
                    "total_tokens": final_state.get("total_tokens", 0),
                    "total_usd_cost": final_state.get("total_usd_cost", 0.0),
                    "completed_subquestion_ids": final_state.get("completed_subquestion_ids", []),
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
                    "run_id": run_id,
                    "status": str(final_status),
                    "citation_coverage": coverage,
                    "warnings": warnings,
                },
            )
        else:
            redis_store.save_run_state(
                run_id,
                {
                    "run_id": run_id,
                    "status": status,
                    "total_tokens": final_state.get("total_tokens", 0),
                    "total_usd_cost": final_state.get("total_usd_cost", 0.0),
                    "partial": final_state.get("partial", False),
                    "degradation_reason": final_state.get("degradation_reason"),
                },
            )
            for trace in final_state.get("trace_log", []):
                if hasattr(trace, "model_dump"):
                    redis_store.append_trace_step(run_id, trace.model_dump())
                elif isinstance(trace, dict):
                    redis_store.append_trace_step(run_id, trace)


    async def reclaim_idle_jobs(self, min_idle_ms: int | None = None) -> int:
        """
        Uses XAUTOCLAIM to reclaim abandoned pending jobs idle > visibility timeout.
        Returns number of successfully reclaimed and processed jobs.
        """
        idle_ms = min_idle_ms if min_idle_ms is not None else int(settings.worker_visibility_timeout * 1000)
        try:
            next_id, claimed, deleted = self.stream_adapter.autoclaim_jobs(
                stream=self.stream,
                group=self.group,
                consumer=self.consumer_name,
                min_idle_time=idle_ms,
                count=10,
            )
        except Exception as e:
            logger.warning(f"Error during autoclaim: {e}")
            return 0

        reclaimed_count = 0
        if claimed:
            self.client.incrby("research:metrics:worker_reclaims_total", len(claimed))
            for entry_id, payload in claimed:
                logger.info(f"Reclaimed idle job {entry_id} for run {payload.get('run_id')}")
                try:
                    await self.process_job(entry_id, payload)
                    reclaimed_count += 1
                except Exception as e:
                    logger.warning(f"Error processing reclaimed job {entry_id}: {e}")

        return reclaimed_count

    async def run_loop(self) -> None:
        """Main worker execution loop."""
        self.register_fake_worker()
        self._setup_signal_handlers()

        # Initialize stream consumer group
        self.stream_adapter.create_group(self.stream, self.group)

        # Scan for initial pending jobs on startup
        logger.info(f"Worker {self.consumer_name} started. Scanning pending jobs...")
        await self.reclaim_idle_jobs(min_idle_ms=0)

        # Background reclaim loop
        async def background_reclaimer():
            while not self._stop_event.is_set():
                await asyncio.sleep(min(5.0, settings.worker_visibility_timeout))
                if self._stop_event.is_set():
                    break
                try:
                    await self.reclaim_idle_jobs()
                except Exception as e:
                    logger.debug(f"Background reclaim check error: {e}")

        reclaim_task = asyncio.create_task(background_reclaimer())

        active_tasks: set[asyncio.Task] = set()

        try:
            while not self._stop_event.is_set():
                active_tasks = {t for t in active_tasks if not t.done()}
                if len(active_tasks) >= settings.max_concurrent_runs:
                    await asyncio.sleep(0.05)
                    continue

                # Read new jobs with short block <= 1000ms so stop_event is checked promptly
                try:
                    read_count = max(1, min(5, settings.max_concurrent_runs - len(active_tasks)))
                    jobs = await asyncio.to_thread(
                        self.stream_adapter.read_job,
                        self.stream,
                        self.group,
                        self.consumer_name,
                        count=read_count,
                        block=1000,
                    )
                except Exception as e:
                    if self._stop_event.is_set():
                        break
                    logger.warning(f"Error reading from stream {self.stream}: {e}")
                    await asyncio.sleep(1.0)
                    continue

                if self._stop_event.is_set():
                    break

                if not jobs:
                    await asyncio.sleep(0.05)
                    continue

                logger.info(f"Worker {self.consumer_name} read {len(jobs)} jobs: {[j[1].get('run_id') for j in jobs]}")

                for entry_id, payload in jobs:
                    if self._stop_event.is_set():
                        break
                    t = asyncio.create_task(self.process_job(entry_id, payload))
                    active_tasks.add(t)


        finally:
            reclaim_task.cancel()
            try:
                await reclaim_task
            except Exception:
                pass
            if active_tasks:
                await asyncio.gather(*active_tasks, return_exceptions=True)
            self.unregister_fake_worker()
            logger.info(f"Worker {self.consumer_name} stopped cleanly.")


    async def run_cli(self) -> None:
        """CLI runner entrypoint with logging configuration."""
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        logger.info(f"Starting Multi-Agent Research Worker (PID: {os.getpid()})...")
        await self.run_loop()


if __name__ == "__main__":
    consumer_arg = sys.argv[1] if len(sys.argv) > 1 else None
    worker_instance = ResearchStreamWorker(consumer_name=consumer_arg)
    asyncio.run(worker_instance.run_cli())
