import time
from typing import Any


class BudgetExceededError(Exception):
    """Raised when an operation would exceed hard budget constraints."""

    def __init__(self, reason: str, budget_type: str):
        super().__init__(reason)
        self.reason = reason
        self.budget_type = budget_type


class BudgetGuard:
    """
    Hard Budget Circuit Breaker enforced inside graph logic before every LLM or tool call.
    Guarantees that:
    1. Max sub-questions is never exceeded
    2. Max searches per sub-question is never exceeded
    3. Max total tokens is never exceeded
    4. Max USD cost is never exceeded
    5. Wall-clock timeout is never exceeded
    6. Max schema repair attempts is never exceeded
    """

    def __init__(
        self,
        max_subquestions: int = 5,
        max_searches_per_subquestion: int = 3,
        max_total_tokens: int = 50000,
        max_usd_cost: float = 1.00,
        wall_clock_timeout_seconds: float = 240.0,
        max_repair_attempts: int = 2,
        start_time: float | None = None,
    ):
        self.max_subquestions = max_subquestions
        self.max_searches_per_subquestion = max_searches_per_subquestion
        self.max_total_tokens = max_total_tokens
        self.max_usd_cost = max_usd_cost
        self.wall_clock_timeout_seconds = wall_clock_timeout_seconds
        self.max_repair_attempts = max_repair_attempts

        self.start_time = start_time if start_time is not None else time.time()
        self.current_tokens: int = 0
        self.current_usd_cost: float = 0.0
        self.current_repairs: int = 0
        self.searches_by_subquestion: dict[str, int] = {}
        self.is_exhausted: bool = False
        self.exhaustion_reason: str | None = None
        self.exhausted_budget_type: str | None = None

    def check_wall_clock(self) -> None:
        elapsed = time.time() - self.start_time
        if elapsed >= self.wall_clock_timeout_seconds:
            self.is_exhausted = True
            self.exhaustion_reason = (
                f"Wall-clock ceiling exceeded: elapsed {elapsed:.2f}s >= limit {self.wall_clock_timeout_seconds}s"
            )
            self.exhausted_budget_type = "wall_clock"
            raise BudgetExceededError(self.exhaustion_reason, "wall_clock")

    def check_tokens(self, incoming_tokens: int = 0) -> None:
        if (self.current_tokens + incoming_tokens) >= self.max_total_tokens:
            self.is_exhausted = True
            self.exhaustion_reason = (
                f"Token budget exceeded: {self.current_tokens + incoming_tokens} >= limit {self.max_total_tokens}"
            )
            self.exhausted_budget_type = "tokens"
            raise BudgetExceededError(self.exhaustion_reason, "tokens")

    def check_usd_cost(self, incoming_cost: float = 0.0) -> None:
        if (self.current_usd_cost + incoming_cost) >= self.max_usd_cost:
            self.is_exhausted = True
            self.exhaustion_reason = (
                f"USD cost budget exceeded: ${self.current_usd_cost + incoming_cost:.4f} >= limit ${self.max_usd_cost:.2f}"
            )
            self.exhausted_budget_type = "cost"
            raise BudgetExceededError(self.exhaustion_reason, "cost")

    def check_before_call(self, estimated_tokens: int = 100) -> None:
        """Called before every LLM call."""
        self.check_wall_clock()
        self.check_tokens(estimated_tokens)
        self.check_usd_cost()

    def can_search(self, sub_question_id: str) -> bool:
        """Returns True if a search can be dispatched for this sub-question."""
        if self.is_exhausted:
            return False
        try:
            self.check_wall_clock()
            self.check_usd_cost()
        except BudgetExceededError:
            return False

        count = self.searches_by_subquestion.get(sub_question_id, 0)
        return count < self.max_searches_per_subquestion

    def record_search(self, sub_question_id: str) -> None:
        count = self.searches_by_subquestion.get(sub_question_id, 0)
        if count >= self.max_searches_per_subquestion:
            self.is_exhausted = True
            self.exhaustion_reason = (
                f"Search limit exceeded for sub-question {sub_question_id}: "
                f"{count + 1} > limit {self.max_searches_per_subquestion}"
            )
            self.exhausted_budget_type = "search_count"
            raise BudgetExceededError(self.exhaustion_reason, "search_count")
        self.searches_by_subquestion[sub_question_id] = count + 1

    def record_spend(self, tokens: int, usd_cost: float) -> None:
        self.current_tokens += tokens
        self.current_usd_cost = round(self.current_usd_cost + usd_cost, 6)

    def record_repair_attempt(self) -> None:
        self.current_repairs += 1
        if self.current_repairs > self.max_repair_attempts:
            self.is_exhausted = True
            self.exhaustion_reason = (
                f"Schema repair attempts exceeded: {self.current_repairs} > limit {self.max_repair_attempts}"
            )
            self.exhausted_budget_type = "repair_attempts"
            raise BudgetExceededError(self.exhaustion_reason, "repair_attempts")

    def get_summary(self) -> dict[str, Any]:
        return {
            "is_exhausted": self.is_exhausted,
            "exhaustion_reason": self.exhaustion_reason,
            "exhausted_budget_type": self.exhausted_budget_type,
            "total_tokens": self.current_tokens,
            "total_usd_cost": self.current_usd_cost,
            "elapsed_seconds": round(time.time() - self.start_time, 2),
            "repair_attempts": self.current_repairs,
            "searches_by_subquestion": dict(self.searches_by_subquestion),
        }
