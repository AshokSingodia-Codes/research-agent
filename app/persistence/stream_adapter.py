import logging
from typing import Any

import redis
from redis.exceptions import ResponseError

from app.persistence.redis_store import redis_store

logger = logging.getLogger(__name__)


class RedisStreamAdapter:
    """
    Adapter for Redis Streams providing resilient stream operations.
    Handles compatibility between real Redis and fakeredis.
    If stream commands like XAUTOCLAIM are missing or unsupported in the underlying client,
    it falls back to a composition of XPENDING and XCLAIM.
    """

    def __init__(self, client: Any | None = None):
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is not None:
            return self._client
        return redis_store.get_client()

    def create_group(self, stream: str, group: str, start_id: str = "0", mkstream: bool = True) -> bool:
        """Creates consumer group on the stream. Idempotent if group already exists."""
        try:
            self.client.xgroup_create(stream, group, id=start_id, mkstream=mkstream)
            return True
        except (ResponseError, redis.exceptions.RedisError) as e:
            if "BUSYGROUP" in str(e):
                return True
            raise e

    def publish_job(self, stream: str, job_data: dict[str, Any]) -> str:
        """Publishes a job entry to the stream, returning the generated entry ID."""
        return self.client.xadd(stream, job_data)

    def read_job(
        self,
        stream: str,
        group: str,
        consumer: str,
        count: int = 1,
        block: int | None = None,
    ) -> list[tuple[str, dict[str, Any]]]:
        """
        Reads pending/new jobs for consumer in group.
        Returns a normalized list of (entry_id, payload_dict).
        """
        raw = self.client.xreadgroup(group, consumer, {stream: ">"}, count=count, block=block)
        results = []
        if not raw:
            return results

        # Format: [[stream_name, [(msg_id, data), ...]]]
        for stream_entry in raw:
            messages = stream_entry[1] if len(stream_entry) > 1 else []
            for msg_id, payload in messages:
                results.append((msg_id, payload))
        return results

    def ack_job(self, stream: str, group: str, *entry_ids: str) -> int:
        """Acknowledges one or more processed jobs from the stream group."""
        return self.client.xack(stream, group, *entry_ids)

    def get_pending_summary(self, stream: str, group: str) -> dict[str, Any]:
        """Returns summary of pending messages in the consumer group."""
        try:
            return self.client.xpending(stream, group)
        except Exception as e:
            logger.warning(f"Error fetching xpending for {stream}/{group}: {e}")
            return {"pending": 0, "min": None, "max": None, "consumers": []}

    def claim_job(
        self,
        stream: str,
        group: str,
        consumer: str,
        min_idle_time: int,
        entry_ids: list[str],
    ) -> list[tuple[str, dict[str, Any]]]:
        """Transfers ownership of specific pending messages to the specified consumer."""
        return self.client.xclaim(stream, group, consumer, min_idle_time=min_idle_time, message_ids=entry_ids)

    def autoclaim_jobs(
        self,
        stream: str,
        group: str,
        consumer: str,
        min_idle_time: int = 0,
        start_id: str = "0-0",
        count: int = 10,
    ) -> tuple[str, list[tuple[str, dict[str, Any]]], list[str]]:
        """
        Attempts XAUTOCLAIM to reclaim abandoned pending jobs.
        If XAUTOCLAIM is missing or fails (ResponseError/unknown command),
        gracefully falls back to XPENDING_RANGE + XCLAIM.
        """
        if hasattr(self.client, "xautoclaim"):
            try:
                res = self.client.xautoclaim(
                    stream,
                    group,
                    consumer,
                    min_idle_time=min_idle_time,
                    start_id=start_id,
                    count=count,
                )
                # Redis returns: [next_start_id, [(id, data), ...], [deleted_ids...]]
                next_id = res[0] if len(res) > 0 else "0-0"
                claimed = res[1] if len(res) > 1 else []
                deleted = res[2] if len(res) > 2 else []
                return next_id, claimed, deleted
            except (ResponseError, NotImplementedError, Exception) as e:
                err_str = str(e).lower()
                if "unknown command" not in err_str and "not implemented" not in err_str:
                    # Reraise unexpected errors
                    raise e
                logger.info(f"xautoclaim not supported by client ({e}); using XPENDING fallback")

        # --- Fallback: XPENDING + XCLAIM ---
        return self._autoclaim_fallback(
            stream,
            group,
            consumer,
            min_idle_time=min_idle_time,
            count=count,
        )

    def _autoclaim_fallback(
        self,
        stream: str,
        group: str,
        consumer: str,
        min_idle_time: int = 0,
        count: int = 10,
    ) -> tuple[str, list[tuple[str, dict[str, Any]]], list[str]]:
        """Fallback implementation using xpending_range and xclaim."""
        claimed_messages: list[tuple[str, dict[str, Any]]] = []

        # Retrieve pending messages
        try:
            pending_entries = self.client.xpending_range(
                stream,
                group,
                min="-",
                max="+",
                count=count,
            )
        except Exception:
            pending_entries = []

        candidate_ids = []
        for entry in pending_entries:
            # entry format in redis-py: dict with 'message_id', 'idle', 'consumer'
            # or tuple (message_id, consumer, idle, deliveries)
            if isinstance(entry, dict):
                msg_id = entry.get("message_id")
                idle = entry.get("idle", 0)
            elif isinstance(entry, (list, tuple)) and len(entry) >= 3:
                msg_id = entry[0]
                idle = entry[2]
            else:
                continue

            if idle >= min_idle_time and msg_id:
                candidate_ids.append(msg_id)

        if candidate_ids:
            try:
                claimed = self.client.xclaim(
                    stream,
                    group,
                    consumer,
                    min_idle_time=min_idle_time,
                    message_ids=candidate_ids,
                )
                if claimed:
                    claimed_messages.extend(claimed)
            except Exception as e:
                logger.warning(f"Error during fallback xclaim: {e}")

        next_id = candidate_ids[-1] if candidate_ids else "0-0"
        return next_id, claimed_messages, []


stream_adapter = RedisStreamAdapter()
