import atexit
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

import redis

from app.config import settings

logger = logging.getLogger(__name__)

FAKEDB_FILE = Path(".fake_redis.json")

LUA_RELEASE_LOCK = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
else
    return 0
end
"""


class RedisStore:
    """
    Central Redis store manager supporting real Redis and fakeredis.
    Provides run state persistence, distributed locks, TTLs, and trace storage.
    Automatically persists fake Redis snapshots across subprocesses when in test/fake mode.
    """

    def __init__(self):
        self._client: redis.Redis | None = None
        self._fake_client: Any | None = None
        self._last_sync_time: float = 0.0
        self._dirty: bool = False
        self._sync_debounce_seconds: float = 1.0
        self.write_count: int = 0
        self._owned_locks: dict[str, str] = {}

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

    def sync_to_disk(self, force: bool = False) -> None:
        """Flushes in-memory fake redis state to disk for subprocess durability with debounce."""
        if not settings.use_fake_redis or self._fake_client is None:
            return
        now = time.time()
        self._dirty = True
        if not force and (now - self._last_sync_time) < self._sync_debounce_seconds:
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
            tmp_file = FAKEDB_FILE.with_suffix(".tmp")
            tmp_file.write_text(json.dumps(data), encoding="utf-8")
            tmp_file.replace(FAKEDB_FILE)
            self._last_sync_time = now
            self._dirty = False
            self.write_count += 1
        except Exception as e:
            logger.warning(f"Failed to sync fake redis to disk: {e}")

    def flush(self) -> None:
        """Forced synchronous flush on shutdown or lifecycle termination."""
        if self._dirty:
            self.sync_to_disk(force=True)

    def load_from_disk(self) -> None:
        """Restores in-memory fake redis state from disk snapshot."""
        if not settings.use_fake_redis or self._fake_client is None or not FAKEDB_FILE.exists():
            return
        if self._dirty:
            # In-memory changes exist that have not been written to disk yet; don't clobber
            return
        try:
            content = FAKEDB_FILE.read_text(encoding="utf-8")
            if not content.strip():
                return
            data = json.loads(content)
            for k, (t, val) in data.items():
                if k.startswith("research:lock:") or k == "research:worker:active_fake_worker":
                    continue
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
        if status in (
            "queued",
            "planning",
            "researching",
            "reviewing",
            "writing",
            "awaiting_approval",
            "in_progress",
            "running",
        ):
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

    def acquire_lock(self, run_id: str, timeout_seconds: float = 30.0, token: str | None = None) -> str | None:
        """
        Acquires an owner-checked distributed lock for the specified run_id using SET NX PX.
        Stores a random token (or specified token).
        Returns the ownership token string if acquired, or None if already locked.
        """
        client = self.get_client()
        lock_key = f"research:lock:{run_id}"
        owner_token = token or uuid.uuid4().hex
        px_ms = int(timeout_seconds * 1000)
        acquired = client.set(lock_key, owner_token, nx=True, px=px_ms)
        if acquired:
            self._owned_locks[run_id] = owner_token
        self.sync_to_disk()
        return owner_token if acquired else None

    def release_lock(self, run_id: str, token: str | None = None) -> bool:
        """
        Releases the distributed lock ONLY if the token matches the lock's current owner.
        Uses Lua compare-and-delete when available, with an in-memory fallback for fakeredis.
        Returns True if released, False if token did not match or lock did not exist.
        """
        if token is None:
            token = self._owned_locks.get(run_id)

        if token is None:
            logger.warning(f"Unsafe lock release attempted for run_id '{run_id}' without token; ignored.")
            return False

        client = self.get_client()
        lock_key = f"research:lock:{run_id}"
        released = False
        try:
            res = client.eval(LUA_RELEASE_LOCK, 1, lock_key, token)
            released = bool(res == 1)
        except Exception:
            # Fallback for clients without Lua eval (e.g., fakeredis without lupa)
            current = client.get(lock_key)
            if current == token:
                client.delete(lock_key)
                released = True
            else:
                released = False

        if released:
            self._owned_locks.pop(run_id, None)

        self.sync_to_disk()
        return released

    def transition_run_status(
        self,
        run_id: str,
        from_status: Any,
        to_status: Any,
    ) -> bool:
        """
        Performs an atomic compare-and-set status transition.
        Updates state['status'] from from_status to to_status only if current status matches.
        Returns True if transitioned, False otherwise.
        """
        client = self.get_client()
        key = f"research:state:{run_id}"
        from_val = from_status.value if hasattr(from_status, "value") else str(from_status)
        to_val = to_status.value if hasattr(to_status, "value") else str(to_status)

        raw = client.get(key)
        if not raw and settings.use_fake_redis:
            self.load_from_disk()
            raw = client.get(key)
        if not raw:
            return False

        try:
            st = json.loads(raw)
        except Exception:
            return False

        curr_status = st.get("status")
        if hasattr(curr_status, "value"):
            curr_status = curr_status.value
        if curr_status != from_val:
            return False

        st["status"] = to_val
        client.set(key, json.dumps(st, default=str))
        if to_val in (
            "queued",
            "planning",
            "researching",
            "reviewing",
            "writing",
            "awaiting_approval",
            "in_progress",
            "running",
        ):
            client.sadd("research:running_runs", run_id)
        else:
            client.srem("research:running_runs", run_id)
        self.sync_to_disk()
        return True

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

    def append_trace_step(self, run_id: str, step_data: Any) -> None:
        client = self.get_client()
        key = f"research:trace:{run_id}"
        data = step_data.model_dump() if hasattr(step_data, "model_dump") else step_data
        client.rpush(key, json.dumps(data, default=str))
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
atexit.register(redis_store.flush)
