import asyncio
import logging
import threading
import time
import uuid
from typing import Any

from redis.exceptions import ResponseError

logger = logging.getLogger(__name__)

LUA_CHECK_AND_INCR = """
local key = KEYS[1]
local inc = tonumber(ARGV[1])
local cap = tonumber(ARGV[2])
local current = tonumber(redis.call('GET', key) or '0')
if (current + inc) <= cap then
    return redis.call('INCRBY', key, inc)
else
    return -1
end
"""


_fakeredis_budget_lock = threading.Lock()


def check_and_incr_redis(client: Any, key: str, inc: int, cap: int) -> int:
    """
    Atomically checks that key + inc <= cap and increments by inc.
    Returns the new value if successful, or -1 if the cap would be exceeded.
    Uses Lua script in real Redis; falls back to synchronized lock in fakeredis.
    """
    if client is None:
        return -1

    try:
        res = client.eval(LUA_CHECK_AND_INCR, 1, key, inc, cap)
        return int(res)
    except ResponseError as e:
        if "unknown command 'eval'" in str(e).lower():
            with _fakeredis_budget_lock:
                val = int(client.get(key) or 0)
                if (val + inc) > cap:
                    return -1
                return int(client.incrby(key, inc))
        raise e


class BudgetExceededError(Exception):
    """Raised when an operation would exceed hard budget constraints."""

    def __init__(self, reason: str, budget_type: str):
        super().__init__(reason)
        self.reason = reason
        self.budget_type = budget_type


BudgetExceeded = BudgetExceededError


class BudgetGuard:
    """
    Hard Budget Circuit Breaker enforced inside graph logic before every LLM or tool call.
    Features:
    1. In-process asyncio.Lock and threading.Lock for thread/task safety.
    2. Cross-process atomic Redis counters (budget:{run_id}:{tokens|usd_micros|searches}).
    3. Reserve-then-settle pattern: reserve estimated tokens before call, settle actual after.
    4. Persisted start_time so resumes get no fresh clock.
    5. Rejection triggers typed BudgetExceededError allowing graceful graph degradation.
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
        run_id: str | None = None,
        redis_client: Any | None = None,
    ):
        self.max_subquestions = max_subquestions
        self.max_searches_per_subquestion = max_searches_per_subquestion
        self.max_total_tokens = max_total_tokens
        self.max_usd_cost = max_usd_cost
        self.wall_clock_timeout_seconds = wall_clock_timeout_seconds
        self.max_repair_attempts = max_repair_attempts

        self.run_id = run_id or f"run_{uuid.uuid4().hex[:8]}"
        self._redis_client = redis_client
        self._async_lock = asyncio.Lock()
        self._sync_lock = threading.Lock()
        self._reserved_tokens: int = 0

        # Persist and retrieve run start_time in Redis so resumes get no fresh clock
        now = time.time()
        initial_st = start_time if start_time is not None else now
        client = self.client
        if client is not None:
            st_key = f"budget:{self.run_id}:start_time"
            client.set(st_key, str(initial_st), nx=True)
            stored_st = client.get(st_key)
            if stored_st:
                try:
                    self.start_time = float(stored_st)
                except (ValueError, TypeError):
                    self.start_time = initial_st
            else:
                self.start_time = initial_st

            # Sync initial expenditures from Redis if resuming
            stored_tokens = client.get(f"budget:{self.run_id}:tokens")
            stored_micros = client.get(f"budget:{self.run_id}:usd_micros")
            self.current_tokens = int(stored_tokens) if stored_tokens else 0
            self.current_usd_cost = (int(stored_micros) / 1_000_000.0) if stored_micros else 0.0
        else:
            self.start_time = initial_st
            self.current_tokens = 0
            self.current_usd_cost = 0.0

        self.current_repairs: int = 0
        self.searches_by_subquestion: dict[str, int] = {}
        self.is_exhausted: bool = False
        self.exhaustion_reason: str | None = None
        self.exhausted_budget_type: str | None = None

    @property
    def client(self) -> Any:
        if self._redis_client is not None:
            return self._redis_client
        try:
            from app.persistence.redis_store import redis_store

            return redis_store.get_client()
        except Exception:
            return None

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
        client = self.client
        if client is not None:
            val = int(client.get(f"budget:{self.run_id}:tokens") or 0)
            self.current_tokens = max(self.current_tokens, val)

        if (self.current_tokens + incoming_tokens) >= self.max_total_tokens:
            self.is_exhausted = True
            self.exhaustion_reason = (
                f"Token budget exceeded: {self.current_tokens + incoming_tokens} >= limit {self.max_total_tokens}"
            )
            self.exhausted_budget_type = "tokens"
            raise BudgetExceededError(self.exhaustion_reason, "tokens")

    def check_usd_cost(self, incoming_cost: float = 0.0) -> None:
        client = self.client
        if client is not None:
            micros = int(client.get(f"budget:{self.run_id}:usd_micros") or 0)
            self.current_usd_cost = max(self.current_usd_cost, micros / 1_000_000.0)

        if (self.current_usd_cost + incoming_cost) >= self.max_usd_cost:
            self.is_exhausted = True
            self.exhaustion_reason = (
                f"USD cost budget exceeded: ${self.current_usd_cost + incoming_cost:.4f} >= limit ${self.max_usd_cost:.2f}"
            )
            self.exhausted_budget_type = "cost"
            raise BudgetExceededError(self.exhaustion_reason, "cost")

    def reserve(self, estimated_tokens: int = 100) -> int:
        """
        Reserve estimated tokens atomically in Redis before initiating an LLM or tool call.
        Raises typed BudgetExceededError if the reservation exceeds token or wall-clock ceiling.
        """
        self.check_wall_clock()
        self.check_usd_cost()

        client = self.client
        if client is not None:
            token_key = f"budget:{self.run_id}:tokens"
            new_tokens = check_and_incr_redis(client, token_key, estimated_tokens, self.max_total_tokens)
            if new_tokens == -1:
                self.is_exhausted = True
                self.exhaustion_reason = (
                    f"Token budget exceeded: reservation of {estimated_tokens} tokens would exceed limit {self.max_total_tokens}"
                )
                self.exhausted_budget_type = "tokens"
                raise BudgetExceededError(self.exhaustion_reason, "tokens")
            with self._sync_lock:
                self._reserved_tokens += estimated_tokens
                self.current_tokens = new_tokens
        else:
            with self._sync_lock:
                if (self.current_tokens + estimated_tokens) >= self.max_total_tokens:
                    self.is_exhausted = True
                    self.exhaustion_reason = (
                        f"Token budget exceeded: {self.current_tokens + estimated_tokens} >= limit {self.max_total_tokens}"
                    )
                    self.exhausted_budget_type = "tokens"
                    raise BudgetExceededError(self.exhaustion_reason, "tokens")
                self._reserved_tokens += estimated_tokens
                self.current_tokens += estimated_tokens

        return estimated_tokens

    def settle(self, actual_tokens: int, usd_cost: float, reserved_tokens: int | None = None) -> None:
        """
        Settles actual token and USD usage against the previous reservation.
        Adjusts Redis token counter by (actual - reserved) delta, and increments USD micro-dollars.
        """
        client = self.client
        with self._sync_lock:
            res = reserved_tokens if reserved_tokens is not None else self._reserved_tokens
            token_delta = actual_tokens - res
            self._reserved_tokens = max(0, self._reserved_tokens - res)

        if client is not None:
            token_key = f"budget:{self.run_id}:tokens"
            if token_delta > 0:
                new_tokens = check_and_incr_redis(client, token_key, token_delta, self.max_total_tokens)
                if new_tokens == -1:
                    client.incrby(token_key, token_delta)
                    self.is_exhausted = True
                    self.exhaustion_reason = (
                        f"Token budget exceeded: settling {actual_tokens} tokens exceeds limit {self.max_total_tokens}"
                    )
                    self.exhausted_budget_type = "tokens"
            elif token_delta < 0:
                client.decrby(token_key, abs(token_delta))

            # Settle USD cost in micro-dollars
            cost_micros = int(round(usd_cost * 1_000_000))
            if cost_micros > 0:
                usd_key = f"budget:{self.run_id}:usd_micros"
                cap_micros = int(round(self.max_usd_cost * 1_000_000))
                new_micros = check_and_incr_redis(client, usd_key, cost_micros, cap_micros)
                if new_micros == -1:
                    client.incrby(usd_key, cost_micros)
                    self.is_exhausted = True
                    self.exhaustion_reason = (
                        f"USD cost budget exceeded: ${usd_cost:.4f} exceeds limit ${self.max_usd_cost:.2f}"
                    )
                    self.exhausted_budget_type = "cost"
                else:
                    with self._sync_lock:
                        self.current_usd_cost = round(new_micros / 1_000_000.0, 6)

            with self._sync_lock:
                final_tok = int(client.get(token_key) or 0)
                self.current_tokens = final_tok
                final_micros = int(client.get(f"budget:{self.run_id}:usd_micros") or 0)
                self.current_usd_cost = round(final_micros / 1_000_000.0, 6)
        else:
            with self._sync_lock:
                self.current_tokens = self.current_tokens + token_delta
                self.current_usd_cost = round(self.current_usd_cost + usd_cost, 6)
                if self.current_usd_cost >= self.max_usd_cost:
                    self.is_exhausted = True
                    self.exhaustion_reason = f"USD cost limit exceeded: ${self.current_usd_cost:.4f} >= limit ${self.max_usd_cost:.2f}"
                    self.exhausted_budget_type = "cost"
                elif self.current_tokens >= self.max_total_tokens:
                    self.is_exhausted = True
                    self.exhaustion_reason = f"Token budget exceeded: {self.current_tokens} >= limit {self.max_total_tokens}"
                    self.exhausted_budget_type = "tokens"

    def release_reservation(self, reserved_tokens: int | None = None) -> None:
        """Releases an unused budget reservation on failure or cancellation without recording spend."""
        self.settle(actual_tokens=0, usd_cost=0.0, reserved_tokens=reserved_tokens)

    def check_before_call(self, estimated_tokens: int = 100) -> None:
        """Reserve-then-settle check before call."""
        self.reserve(estimated_tokens)

    def can_search(self, sub_question_id: str) -> bool:
        """Returns True if a search can be dispatched for this sub-question."""
        if self.is_exhausted:
            return False
        try:
            self.check_wall_clock()
            self.check_usd_cost()
        except BudgetExceededError:
            return False

        client = self.client
        if client is not None:
            sq_searches = int(client.get(f"budget:{self.run_id}:searches:{sub_question_id}") or 0)
            return sq_searches < self.max_searches_per_subquestion
        count = self.searches_by_subquestion.get(sub_question_id, 0)
        return count < self.max_searches_per_subquestion

    def record_search(self, sub_question_id: str) -> None:
        client = self.client
        if client is not None:
            key = f"budget:{self.run_id}:searches:{sub_question_id}"
            new_count = check_and_incr_redis(client, key, 1, self.max_searches_per_subquestion)
            if new_count == -1:
                self.is_exhausted = True
                self.exhaustion_reason = (
                    f"Search limit exceeded for sub-question {sub_question_id}: "
                    f"limit {self.max_searches_per_subquestion}"
                )
                self.exhausted_budget_type = "search_count"
                raise BudgetExceededError(self.exhaustion_reason, "search_count")
            client.incrby(f"budget:{self.run_id}:searches", 1)
            self.searches_by_subquestion[sub_question_id] = new_count
        else:
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
        self.settle(actual_tokens=tokens, usd_cost=usd_cost)

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
