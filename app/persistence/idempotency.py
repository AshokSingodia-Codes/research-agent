import hashlib
import json
from typing import Any

from app.persistence.redis_store import redis_store


def compute_idempotency_key(run_id: str, node: str, input_data: Any) -> str:
    """
    Derives an idempotency key from (run_id, node, input_hash).
    Guarantees re-entering a completed node returns cached output without repeating LLM or tool calls.
    """
    serialized = json.dumps(input_data, sort_keys=True, default=str)
    content_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
    return f"research:idem:{run_id}:{node}:{content_hash}"


def get_idempotent_result(key: str) -> dict[str, Any] | None:
    """Retrieves cached node output if previously completed."""
    client = redis_store.get_client()
    raw = client.get(key)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def save_idempotent_result(key: str, output: dict[str, Any], ttl_seconds: int = 86400) -> None:
    """Caches completed node output under the idempotency key."""
    client = redis_store.get_client()
    serialized = json.dumps(output, default=str)
    client.set(key, serialized, ex=ttl_seconds)
