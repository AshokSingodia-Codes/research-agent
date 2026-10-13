import pickle
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

from app.persistence.redis_store import redis_store


class RedisCheckpointSaver(BaseCheckpointSaver):
    """
    Durable Redis-backed checkpointer for LangGraph workflows.
    Persists checkpoints, channel versions, and step state to Redis.
    Allows resumed workflow execution from crashed nodes without replaying completed work.
    """

    def __init__(self):
        super().__init__()

    def _redis_key(self, prefix: str, thread_id: str, checkpoint_ns: str = "", extra: str = "") -> str:
        parts = ["research:checkpoint", prefix, thread_id]
        if checkpoint_ns:
            parts.append(checkpoint_ns)
        if extra:
            parts.append(extra)
        return ":".join(parts)

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        redis_store.load_from_disk()
        client = redis_store.get_client()
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"].get("checkpoint_id")

        if not checkpoint_id:
            # Get latest checkpoint_id for this thread
            latest_id = client.get(self._redis_key("latest", thread_id, checkpoint_ns))
            if not latest_id:
                return None
            checkpoint_id = latest_id

        raw_meta = client.get(self._redis_key("meta", thread_id, checkpoint_ns, checkpoint_id))
        if not raw_meta:
            return None

        data = pickle.loads(bytes.fromhex(raw_meta))
        checkpoint_data, metadata_data, parent_checkpoint_id = data

        c = checkpoint_data.copy()
        # Reconstruct channel_values from blobs
        values = {}
        for k, v in c.get("channel_versions", {}).items():
            blob_key = self._redis_key("blob", thread_id, checkpoint_ns, f"{k}:{v}")
            blob_hex = client.get(blob_key)
            if blob_hex:
                values[k] = pickle.loads(bytes.fromhex(blob_hex))

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

        # Store channel values as blobs
        for k, v in new_versions.items():
            if k in values:
                blob_key = self._redis_key("blob", thread_id, checkpoint_ns, f"{k}:{v}")
                client.set(blob_key, pickle.dumps(values[k]).hex())

        # Store checkpoint metadata and hierarchy
        meta_key = self._redis_key("meta", thread_id, checkpoint_ns, checkpoint["id"])
        parent_id = config["configurable"].get("checkpoint_id")
        to_store = (c, get_checkpoint_metadata(config, metadata), parent_id)
        client.set(meta_key, pickle.dumps(to_store).hex())

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
        # Writes tracking for parallel branching
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
