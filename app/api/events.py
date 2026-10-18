import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

from app.persistence.redis_store import redis_store

logger = logging.getLogger(__name__)


class EventBroker:
    """
    Server-Sent Events (SSE) broker supporting historical event playback
    via Last-Event-ID reconnection and live streaming queues.
    """

    def __init__(self):
        self._subscribers: dict[str, list[asyncio.Queue]] = {}

    def publish_event(self, run_id: str, event_type: str, data: dict[str, Any]) -> None:
        """Publishes an event to Redis history and pushes to active live subscriber queues."""
        client = redis_store.get_client()
        seq_key = f"research:events_seq:{run_id}"
        event_id = client.incr(seq_key)

        event_obj = {
            "id": event_id,
            "event": event_type,
            "data": data,
        }

        # Store in Redis historical log
        history_key = f"research:events:{run_id}"
        client.rpush(history_key, json.dumps(event_obj, default=str))
        redis_store.sync_to_disk()

        # Push to live in-memory queues
        queues = self._subscribers.get(run_id, [])
        for q in queues:
            try:
                q.put_nowait(event_obj)
            except Exception:
                pass

    async def event_generator(
        self,
        run_id: str,
        last_event_id: int | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Streams SSE formatted events.
        Replays any missed events starting after last_event_id, then streams live events.
        """
        # 1. Replay historical events
        client = redis_store.get_client()
        history_key = f"research:events:{run_id}"
        historical_raw = client.lrange(history_key, 0, -1)

        last_id_seen = last_event_id or 0
        already_completed = False
        for item in historical_raw:
            try:
                evt = json.loads(item)
                if evt["id"] > last_id_seen:
                    last_id_seen = evt["id"]
                    yield f"id: {evt['id']}\nevent: {evt['event']}\ndata: {json.dumps(evt['data'])}\n\n"
                if evt.get("event") in ("completed", "failed", "cancelled"):
                    already_completed = True
            except Exception:
                pass

        if already_completed:
            return

        if not historical_raw and last_event_id is None:
            state = redis_store.get_run_state(run_id)
            if state:
                yield f"data: {json.dumps({'type': 'init', 'status': state.get('status', 'in_progress'), 'progress': 0})}\n\n"

        # 2. Use existing registered queue or create a new live subscriber queue
        created_queue = False
        if run_id in self._subscribers and self._subscribers[run_id]:
            queue = self._subscribers[run_id][0]
        else:
            queue = asyncio.Queue()
            if run_id not in self._subscribers:
                self._subscribers[run_id] = []
            self._subscribers[run_id].append(queue)
            created_queue = True

        try:
            while True:
                try:
                    evt = await asyncio.wait_for(queue.get(), timeout=25.0)
                    if evt is None:
                        yield f"data: {json.dumps({'type': 'stream_end', 'message': 'Run execution finished'})}\n\n"
                        break

                    if isinstance(evt, dict):
                        if "event" in evt:
                            if evt.get("id", 0) > last_id_seen:
                                last_id_seen = evt["id"]
                                yield f"id: {evt['id']}\nevent: {evt['event']}\ndata: {json.dumps(evt['data'])}\n\n"
                            if evt.get("event") in ("completed", "failed", "cancelled"):
                                break
                        elif "type" in evt:
                            yield f"data: {json.dumps(evt)}\n\n"
                            if evt.get("type") in ("complete", "failed", "cancelled"):
                                break
                except TimeoutError:
                    # Keep-alive heartbeat comment
                    yield ": keep-alive\n\n"
        finally:
            if created_queue and run_id in self._subscribers and queue in self._subscribers[run_id]:
                self._subscribers[run_id].remove(queue)

    def register_queue(self, run_id: str) -> asyncio.Queue:
        """Registers a queue for a run and returns it."""
        q = asyncio.Queue()
        if run_id not in self._subscribers:
            self._subscribers[run_id] = []
        self._subscribers[run_id].append(q)
        return q

    def unregister_queue(self, run_id: str, queue: asyncio.Queue | None = None) -> None:
        """Unregisters queue or all queues for run_id."""
        if run_id in self._subscribers:
            if queue:
                if queue in self._subscribers[run_id]:
                    self._subscribers[run_id].remove(queue)
            else:
                self._subscribers.pop(run_id, None)


event_broker = EventBroker()


def register_stream_queue(run_id: str) -> asyncio.Queue:
    """Ported from src/services/runner.py: Registers an event queue for live progress streaming."""
    return event_broker.register_queue(run_id)


def unregister_stream_queue(run_id: str, queue: asyncio.Queue | None = None) -> None:
    """Ported from src/services/runner.py: Cleans up the event queue for live progress streaming."""
    event_broker.unregister_queue(run_id, queue)
