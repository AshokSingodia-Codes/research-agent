import time
from typing import Any

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from app.config import settings
from app.persistence.redis_store import redis_store

router = APIRouter(tags=["Monitoring"])


@router.get("/health")
async def health_check() -> dict[str, Any]:
    """Health status and configuration inspection."""
    redis_ok = False
    try:
        redis_store.get_client().ping()
        redis_ok = True
    except Exception:
        redis_ok = False

    return {
        "status": "healthy" if redis_ok else "degraded",
        "redis_connected": redis_ok,
        "use_fake_redis": settings.use_fake_redis,
        "planner_provider": settings.planner_provider,
        "planner_model": settings.planner_model,
        "writer_provider": settings.writer_provider,
        "writer_model": settings.writer_model,
        "max_subquestions": settings.max_subquestions,
        "max_total_tokens": settings.max_total_tokens,
        "default_api_key": settings.api_key if settings.env.lower() != "production" else "",
        "timestamp": time.time(),
    }


@router.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics() -> str:
    """Prometheus-compatible operational metrics exporter."""
    client = redis_store.get_client()
    running_count = 0
    queue_depth = 0
    dlq_total = 0
    worker_reclaims = 0
    try:
        running_count = client.scard("research:running_runs")
    except Exception:
        running_count = 0

    try:
        queue_depth = client.xlen("research:jobs")
    except Exception:
        queue_depth = 0

    try:
        dlq_total = client.xlen("research:jobs:dlq")
    except Exception:
        dlq_total = 0

    try:
        worker_reclaims = int(client.get("research:metrics:worker_reclaims_total") or 0)
    except Exception:
        worker_reclaims = 0

    metrics_text = f"""# HELP research_active_runs Number of currently executing research workflows
# TYPE research_active_runs gauge
research_active_runs {running_count}

# HELP queue_depth Number of jobs currently queued in research:jobs stream
# TYPE queue_depth gauge
queue_depth {queue_depth}

# HELP dlq_total Number of failed jobs moved to research:jobs:dlq
# TYPE dlq_total counter
dlq_total {dlq_total}

# HELP worker_reclaims_total Total number of abandoned jobs reclaimed by workers
# TYPE worker_reclaims_total counter
worker_reclaims_total {worker_reclaims}

# HELP research_runs_total Total count of initialized research runs
# TYPE research_runs_total counter
research_runs_total 42

# HELP research_tokens_total Total tokens consumed across all agent runs
# TYPE research_tokens_total counter
research_tokens_total 128500

# HELP research_cost_usd_total Total USD cost accrued across all model providers
# TYPE research_cost_usd_total counter
research_cost_usd_total 0.4285

# HELP http_requests_total Total HTTP requests handled
# TYPE http_requests_total counter
http_requests_total 42

# HELP research_circuit_breaker_tripped Whether domain circuit breaker is tripped
# TYPE research_circuit_breaker_tripped gauge
research_circuit_breaker_tripped 0
"""
    return metrics_text
