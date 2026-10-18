import hashlib
import hmac
import json
import logging
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from app.config import settings
from app.persistence.redis_store import redis_store

logger = logging.getLogger(__name__)


class RedisCheckpointSaver(BaseCheckpointSaver):
    """
    Durable Redis-backed checkpointer for LangGraph workflows.
    Persists checkpoints, channel versions, and step state to Redis using secure JSON/Msgpack serialization.
    Enforces optional HMAC signatures over checkpoint blobs to prevent tampering.
    Zero pickle usage.
    """

    def __init__(self):
        super().__init__()
        self._serializer = JsonPlusSerializer().with_msgpack_allowlist([("app",)])

    def _redis_key(self, prefix: str, thread_id: str, checkpoint_ns: str = "", extra: str = "") -> str:
        parts = ["research:checkpoint", prefix, thread_id]
        if checkpoint_ns:
            parts.append(checkpoint_ns)
        if extra:
            parts.append(extra)
        return ":".join(parts)

    def _serialize_blob(self, obj: Any) -> str:
        """Serializes an object using JsonPlusSerializer and attaches an optional HMAC signature."""
        t, b = self._serializer.dumps_typed(obj)
        sig = ""
        if settings.checkpoint_hmac_key:
            sig = hmac.new(settings.checkpoint_hmac_key.encode("utf-8"), b, hashlib.sha256).hexdigest()
        payload = {
            "t": t,
            "p": b.hex(),
            "s": sig,
        }
        return json.dumps(payload)

    def _deserialize_blob(self, raw_str: str) -> Any:
        """Validates HMAC signature and deserializes an object using JsonPlusSerializer."""
        try:
            payload = json.loads(raw_str)
            b = bytes.fromhex(payload["p"])
            t = payload["t"]
            sig = payload.get("s", "")
        except Exception as e:
            raise ValueError(f"Malformed checkpoint envelope: {e}") from e

        if settings.checkpoint_hmac_key:
            expected_sig = hmac.new(settings.checkpoint_hmac_key.encode("utf-8"), b, hashlib.sha256).hexdigest()
            if not hmac.compare_digest(sig, expected_sig):
                raise ValueError("Checkpoint HMAC signature verification failed")

        return self._serializer.loads_typed((t, b))

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        client = redis_store.get_client()
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"].get("checkpoint_id")

        if not checkpoint_id:
            # Get latest checkpoint_id for this thread
            latest_id = client.get(self._redis_key("latest", thread_id, checkpoint_ns))
            if not latest_id and settings.use_fake_redis:
                redis_store.load_from_disk()
                latest_id = client.get(self._redis_key("latest", thread_id, checkpoint_ns))
            if not latest_id:
                return None
            checkpoint_id = latest_id

        raw_meta = client.get(self._redis_key("meta", thread_id, checkpoint_ns, checkpoint_id))
        if not raw_meta:
            return None

        try:
            data = self._deserialize_blob(raw_meta)
        except Exception as e:
            logger.error(f"Failed to deserialize checkpoint metadata for thread {thread_id}: {e}")
            raise

        checkpoint_data, metadata_data, parent_checkpoint_id = data

        c = checkpoint_data.copy()
        # Reconstruct channel_values from blobs
        values = {}
        for k, v in c.get("channel_versions", {}).items():
            blob_key = self._redis_key("blob", thread_id, checkpoint_ns, f"{k}:{v}")
            blob_str = client.get(blob_key)
            if blob_str:
                try:
                    values[k] = self._deserialize_blob(blob_str)
                except Exception as e:
                    logger.error(f"Failed to deserialize channel blob {k}:{v} for thread {thread_id}: {e}")
                    raise

        c["channel_values"] = values

        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            },
            checkpoint=c,
            metadata=metadata_data,
            parent_config=(
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": parent_checkpoint_id,
                    }
                }
                if parent_checkpoint_id
                else None
            ),
        )

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        if not config:
            return
        client = redis_store.get_client()
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        history_key = self._redis_key("history", thread_id, checkpoint_ns)
        ids = client.lrange(history_key, 0, (limit - 1) if limit else -1)
        for cid in ids:
            t = self.get_tuple(
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": cid,
                    }
                }
            )
            if t:
                yield t

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        client = redis_store.get_client()
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        c = checkpoint.copy()
        values: dict[str, Any] = c.pop("channel_values", {})

        # Store channel values as serialized blobs
        for k, v in new_versions.items():
            if k in values:
                blob_key = self._redis_key("blob", thread_id, checkpoint_ns, f"{k}:{v}")
                client.set(blob_key, self._serialize_blob(values[k]))

        # Store checkpoint metadata and hierarchy
        meta_key = self._redis_key("meta", thread_id, checkpoint_ns, checkpoint["id"])
        parent_id = config["configurable"].get("checkpoint_id")
        to_store = (c, get_checkpoint_metadata(config, metadata), parent_id)
        client.set(meta_key, self._serialize_blob(to_store))

        # Update latest pointer and history list
        client.set(self._redis_key("latest", thread_id, checkpoint_ns), checkpoint["id"])
        client.rpush(self._redis_key("history", thread_id, checkpoint_ns), checkpoint["id"])
        redis_store.sync_to_disk()

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        pass

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return self.get_tuple(config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        for item in self.list(config, filter=filter, before=before, limit=limit):
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return self.put(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        self.put_writes(config, writes, task_id, task_path)


checkpointer = RedisCheckpointSaver()
