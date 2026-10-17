import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import redis
import fakeredis

from src.config import settings
from src.models.schemas import RunStatus, StepTrace


class RedisStateStore:
    """
    Manages persistent run state, progress tracking, and step-level trace logs in Redis.
    Supports real Redis and in-memory fakeredis.
    """

    def __init__(self, client: Optional[redis.Redis] = None):
        if client is not None:
            self.client = client
        elif settings.use_fake_redis:
            self.client = fakeredis.FakeRedis(decode_responses=True)
        else:
            try:
                self.client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
                self.client.ping()
            except Exception:
                # Graceful fallback to fakeredis if Redis daemon is down
                self.client = fakeredis.FakeRedis(decode_responses=True)

    def _state_key(self, run_id: str) -> str:
        return f"research:state:{run_id}"

    def _trace_key(self, run_id: str) -> str:
        return f"research:trace:{run_id}"

    def _report_key(self, run_id: str) -> str:
        return f"research:report:{run_id}"

    def save_run_state(
        self,
        run_id: str,
        status: str,
        current_node: Optional[str] = None,
        plan_dict: Optional[Dict[str, Any]] = None,
        completed_sq_ids: Optional[List[str]] = None,
        total_tokens: int = 0,
        total_subquestions: int = 0,
        error: Optional[str] = None
    ) -> None:
        """Writes current run metadata to Redis for status polling and crash recovery."""
        completed_count = len(completed_sq_ids or [])
        progress_pct = 0.0
        if total_subquestions > 0:
            progress_pct = round((completed_count / total_subquestions) * 100.0, 1)
        if status == "completed":
            progress_pct = 100.0

        state_payload = {
            "run_id": run_id,
            "status": status,
            "current_node": current_node or "",
            "progress_percentage": progress_pct,
            "completed_subquestions": completed_count,
            "total_subquestions": total_subquestions,
            "total_tokens": total_tokens,
            "error": error or "",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "plan_json": json.dumps(plan_dict) if plan_dict else "",
            "completed_sq_ids": json.dumps(completed_sq_ids or [])
        }
        self.client.hset(self._state_key(run_id), mapping=state_payload)

    def get_run_status(self, run_id: str) -> Optional[RunStatus]:
        """Retrieves real-time status and progress metrics for a run."""
        raw = self.client.hgetall(self._state_key(run_id))
        if not raw:
            return None

        return RunStatus(
            run_id=run_id,
            status=raw.get("status", "in_progress"),
            progress_percentage=float(raw.get("progress_percentage", 0.0)),
            completed_subquestions=int(raw.get("completed_subquestions", 0)),
            total_subquestions=int(raw.get("total_subquestions", 0)),
            total_tokens=int(raw.get("total_tokens", 0)),
            current_node=raw.get("current_node") or None,
            error=raw.get("error") or None
        )

    def append_trace_step(self, run_id: str, step: StepTrace) -> None:
        """Appends a single StepTrace JSON record to the run's audit trail."""
        self.client.rpush(self._trace_key(run_id), step.model_dump_json())

    def get_trace_log(self, run_id: str) -> List[StepTrace]:
        """Returns the full chronological step-level execution trace for a run."""
        items = self.client.lrange(self._trace_key(run_id), 0, -1)
        traces = []
        for raw in items:
            traces.append(StepTrace.model_validate_json(raw))
        return traces

    def save_final_report(self, run_id: str, report_json: str) -> None:
        """Stores the completed report JSON for instant retrieval."""
        self.client.set(self._report_key(run_id), report_json)

    def get_final_report(self, run_id: str) -> Optional[str]:
        """Retrieves the completed report JSON."""
        return self.client.get(self._report_key(run_id))


# Global default instance
redis_store = RedisStateStore()
