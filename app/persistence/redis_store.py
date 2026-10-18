import json
import logging
from pathlib import Path
from typing import Any

import redis

from app.config import settings

logger = logging.getLogger(__name__)

FAKEDB_FILE = Path(".fake_redis.json")


class RedisStore:
    """
    Central Redis store manager supporting real Redis and fakeredis.
    Provides run state persistence, distributed locks, TTLs, and trace storage.
    Automatically persists fake Redis snapshots across subprocesses when in test/fake mode.
    """

    def __init__(self):
        self._client: redis.Redis | None = None
        self._fake_client: Any | None = None

    def get_client(self) -> redis.Redis:
        if self._client is not None:
            return self._client

        if settings.use_fake_redis:
            if self._fake_client is None:
                try:
                    import fakeredis

                    self._fake_client = fakeredis.FakeRedis(decode_responses=True)
                    self.load_from_disk()
                except Exception:
                    self._client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
                    return self._client
            return self._fake_client

        try:
            self._client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
            self._client.ping()
        except Exception as e:
            logger.warning(f"Could not connect to Redis at {settings.redis_url} ({e}); falling back to fakeredis")
            import fakeredis

            self._fake_client = fakeredis.FakeRedis(decode_responses=True)
            self.load_from_disk()
            return self._fake_client

        return self._client

    def sync_to_disk(self) -> None:
        """Flushes in-memory fake redis state to disk for subprocess durability."""
        if not settings.use_fake_redis or self._fake_client is None:
            return
        try:
            data = {}
            for k in self._fake_client.keys():
                t = self._fake_client.type(k)
                if t == "string":
                    data[k] = ("string", self._fake_client.get(k))
                elif t == "set":
                    data[k] = ("set", list(self._fake_client.smembers(k)))
                elif t == "list":
                    data[k] = ("list", self._fake_client.lrange(k, 0, -1))
            FAKEDB_FILE.write_text(json.dumps(data), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to sync fake redis to disk: {e}")

    def load_from_disk(self) -> None:
        """Restores in-memory fake redis state from disk snapshot."""
        if not settings.use_fake_redis or self._fake_client is None or not FAKEDB_FILE.exists():
            return
        try:
            content = FAKEDB_FILE.read_text(encoding="utf-8")
            if not content.strip():
                return
            data = json.loads(content)
            for k, (t, val) in data.items():
                if t == "string":
                    self._fake_client.set(k, val)
                elif t == "set":
                    for item in val:
                        self._fake_client.sadd(k, item)
                elif t == "list":
                    self._fake_client.delete(k)
                    for item in val:
                        self._fake_client.rpush(k, item)
        except Exception as e:
            logger.warning(f"Failed to load fake redis snapshot: {e}")

    # --- Run State Operations ---

    def save_run_state(
        self,
        run_id: str,
        state_data: dict[str, Any] | None = None,
        ttl: int | None = None,
        **kwargs,
    ) -> None:
        client = self.get_client()
        key = f"research:state:{run_id}"
        merged = dict(state_data or {})
        merged.update(kwargs)
        if "run_id" not in merged:
            merged["run_id"] = run_id
        serialized = json.dumps(merged, default=str)
        if ttl:
            client.set(key, serialized, ex=ttl)
        else:
            client.set(key, serialized)

        # Track status in running runs index if in-progress
        status = merged.get("status", "")
        if hasattr(status, "value"):
            status = status.value
        if status in ("queued", "planning", "researching", "reviewing", "writing", "awaiting_approval", "in_progress"):
            client.sadd("research:running_runs", run_id)
        else:
            client.srem("research:running_runs", run_id)
        self.sync_to_disk()

    def get_run_state(self, run_id: str) -> dict[str, Any] | None:
        client = self.get_client()
        key = f"research:state:{run_id}"
        raw = client.get(key)
        if raw is None and settings.use_fake_redis:
            self.load_from_disk()
            raw = client.get(key)
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    def get_run_status(self, run_id: str) -> Any | None:
        st = self.get_run_state(run_id)
        if not st:
            return None
        from types import SimpleNamespace

        completed = len(st.get("completed_subquestion_ids", st.get("completed_sq_ids", [])))
        total_sq = st.get("total_subquestions", 0)
        progress = st.get("progress_percentage", 0.0)
        if total_sq > 0 and progress == 0.0:
            progress = round((completed / total_sq) * 100.0, 1)

        return SimpleNamespace(
            run_id=st.get("run_id", run_id),
            status=st.get("status", "unknown"),
            progress_percentage=progress,
            completed_subquestions=completed,
            total_subquestions=total_sq,
            total_tokens=st.get("total_tokens", 0),
            current_node=st.get("current_node"),
            error=st.get("error"),
        )

    def list_running_runs(self) -> list[str]:
        client = self.get_client()
        members = client.smembers("research:running_runs")
        return list(members) if members else []

    # --- Distributed Locks ---

    def acquire_lock(self, run_id: str, timeout_seconds: float = 30.0) -> bool:
        """Acquires a distributed lock for the specified run_id using SET NX PX."""
        client = self.get_client()
        lock_key = f"research:lock:{run_id}"
        px_ms = int(timeout_seconds * 1000)
        acquired = client.set(lock_key, "locked", nx=True, px=px_ms)
        self.sync_to_disk()
        return bool(acquired)

    def release_lock(self, run_id: str) -> None:
        client = self.get_client()
        lock_key = f"research:lock:{run_id}"
        client.delete(lock_key)
        self.sync_to_disk()

    # --- TTL Management ---

    def set_finished_ttl(self, run_id: str, ttl_seconds: int | None = None) -> None:
        client = self.get_client()
        ttl = ttl_seconds or settings.redis_ttl_seconds
        client.expire(f"research:state:{run_id}", ttl)
        client.expire(f"research:trace:{run_id}", ttl)
        client.expire(f"research:report:{run_id}", ttl)
        client.srem("research:running_runs", run_id)
        self.sync_to_disk()

    # --- Tracing Storage ---

    def append_trace_step(self, run_id: str, step_data: dict[str, Any]) -> None:
        client = self.get_client()
        key = f"research:trace:{run_id}"
        client.rpush(key, json.dumps(step_data, default=str))
        self.sync_to_disk()

    def get_trace_steps(self, run_id: str) -> list[dict[str, Any]]:
        client = self.get_client()
        key = f"research:trace:{run_id}"
        items = client.lrange(key, 0, -1)
        if not items and settings.use_fake_redis:
            self.load_from_disk()
            items = client.lrange(key, 0, -1)
        res = []
        for item in items:
            try:
                res.append(json.loads(item))
            except Exception:
                pass
        return res

    # --- Final Report Storage ---

    def save_final_report(self, run_id: str, report_json: str) -> None:
        client = self.get_client()
        key = f"research:report:{run_id}"
        client.set(key, report_json)
        self.set_finished_ttl(run_id)
        self.sync_to_disk()

    def get_final_report(self, run_id: str) -> str | None:
        client = self.get_client()
        key = f"research:report:{run_id}"
        val = client.get(key)
        if val is None and settings.use_fake_redis:
            self.load_from_disk()
            val = client.get(key)
        return val


redis_store = RedisStore()
