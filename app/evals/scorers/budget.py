from typing import Any

from app.config import settings


def score_budget_adherence(
    total_tokens: int,
    total_usd_cost: float,
    wall_clock_seconds: float,
    partial: bool = False,
    degradation_reason: str | None = None,
) -> dict[str, Any]:
    """
    Evaluates hard budget enforcement:
    - token_limit_exceeded: whether tokens consumed exceeded MAX_TOTAL_TOKENS
    - usd_ceiling_exceeded: whether USD cost exceeded MAX_USD_COST
    - wall_clock_exceeded: whether execution time exceeded WALL_CLOCK_TIMEOUT_SECONDS
    - budget_compliance: True if 0 limits breached and gracefully degraded when capped
    """
    token_cap = settings.max_total_tokens
    usd_cap = settings.max_usd_cost
    timeout_cap = settings.wall_clock_timeout_seconds

    tokens_ok = total_tokens <= token_cap
    usd_ok = total_usd_cost <= (usd_cap * 1.05)  # 5% buffer for in-flight settlement
    timeout_ok = wall_clock_seconds <= (timeout_cap * 1.25)

    is_compliant = tokens_ok and usd_ok and timeout_ok

    return {
        "budget_compliance": is_compliant,
        "total_tokens": total_tokens,
        "token_limit": token_cap,
        "token_limit_exceeded": not tokens_ok,
        "total_usd_cost": round(total_usd_cost, 4),
        "usd_limit": usd_cap,
        "usd_limit_exceeded": not usd_ok,
        "wall_clock_seconds": round(wall_clock_seconds, 2),
        "wall_clock_limit": timeout_cap,
        "wall_clock_exceeded": not timeout_ok,
        "partial_degradation_triggered": partial,
        "degradation_reason": degradation_reason,
    }
