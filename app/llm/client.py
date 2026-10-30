import json
import logging
import time
from typing import Any, TypeVar

from pydantic import BaseModel

from app.config import settings
from app.llm.anthropic_provider import AnthropicProvider
from app.llm.base import (
    InvalidOutputError,
    LLMProvider,
    LLMResponse,
    RateLimitedError,
)
from app.llm.fake_provider import FakeProvider
from app.llm.groq_provider import GroqProvider
from app.models.schemas import StepTrace

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """
    Thin client wrapper for all LLM calls.
    Enforces:
    1. Provider resolution per agent
    2. Automatic token, cost, and latency recording into StepTrace
    3. Retry-with-repair on schema validation failure (max 2 repairs)
    4. Rate limit backoff with jitter
    """

    def __init__(self):
        self._providers: dict[str, LLMProvider] = {}
        self._override_provider: LLMProvider | None = None

    def set_override_provider(self, provider: LLMProvider | None) -> None:
        """Sets a provider override (e.g., FakeProvider for tests)."""
        self._override_provider = provider

    def get_model_for_agent(self, agent_name: str) -> str:
        """Returns the configured model string for a given agent role."""
        agent_key = agent_name.lower()
        if "planner" in agent_key:
            return settings.planner_model
        elif "researcher" in agent_key:
            return settings.researcher_model
        elif "writer" in agent_key:
            return settings.writer_model
        elif "supervisor" in agent_key or "review" in agent_key:
            return settings.supervisor_model
        elif "critic" in agent_key:
            return settings.critic_model
        return settings.groq_model

    def get_provider_for_agent(self, agent_name: str) -> tuple[LLMProvider, str]:
        """Resolves (provider_instance, model_name) for a specific agent role."""
        model = self.get_model_for_agent(agent_name)

        if self._override_provider is not None:
            return self._override_provider, model

        if settings.is_mock_llm:
            return self._get_fake_provider(), model

        agent_key = agent_name.lower()
        if "planner" in agent_key:
            p_name = settings.planner_provider.lower()
        elif "researcher" in agent_key:
            p_name = settings.researcher_provider.lower()
        elif "writer" in agent_key:
            p_name = settings.writer_provider.lower()
        elif "supervisor" in agent_key or "review" in agent_key:
            p_name = settings.supervisor_provider.lower()
        elif "critic" in agent_key:
            p_name = settings.critic_provider.lower()
        else:
            p_name = "groq"

        provider = self._get_or_create_provider(p_name)
        return provider, model

    def _get_fake_provider(self) -> FakeProvider:
        if "fake" not in self._providers:
            self._providers["fake"] = FakeProvider()
        return self._providers["fake"]  # type: ignore

    def _get_or_create_provider(self, provider_name: str) -> LLMProvider:
        if provider_name in self._providers:
            return self._providers[provider_name]

        from app.llm.openai_provider import OpenAICompatibleProvider

        if provider_name == "anthropic":
            if settings.anthropic_api_key:
                self._providers["anthropic"] = AnthropicProvider(api_key=settings.anthropic_api_key)
                return self._providers["anthropic"]
            logger.warning("Anthropic key missing; falling back...")
        elif provider_name == "groq":
            if settings.groq_api_key:
                self._providers["groq"] = GroqProvider(api_key=settings.groq_api_key)
                return self._providers["groq"]
            logger.warning("Groq key missing; falling back...")
        elif provider_name == "gemini":
            if settings.gemini_api_key:
                self._providers["gemini"] = OpenAICompatibleProvider(
                    api_key=settings.gemini_api_key,
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                    provider_name="gemini",
                )
                return self._providers["gemini"]
            logger.warning("Gemini key missing; falling back...")
        elif provider_name == "openrouter":
            if settings.openrouter_api_key:
                self._providers["openrouter"] = OpenAICompatibleProvider(
                    api_key=settings.openrouter_api_key,
                    base_url="https://openrouter.ai/api/v1",
                    provider_name="openrouter",
                )
                return self._providers["openrouter"]
            logger.warning("OpenRouter key missing; falling back...")
        elif provider_name == "grok":
            if settings.grok_api_key:
                self._providers["grok"] = OpenAICompatibleProvider(
                    api_key=settings.grok_api_key,
                    base_url="https://api.x.ai/v1",
                    provider_name="grok",
                )
                return self._providers["grok"]
            logger.warning("Grok key missing; falling back...")

        # Fallback cascade: try any available configured real provider before FakeProvider
        if settings.groq_api_key:
            return self._get_or_create_provider("groq")
        if settings.gemini_api_key:
            return self._get_or_create_provider("gemini")
        if settings.openrouter_api_key:
            return self._get_or_create_provider("openrouter")
        if settings.anthropic_api_key:
            return self._get_or_create_provider("anthropic")
        if settings.grok_api_key:
            return self._get_or_create_provider("grok")

        self._providers["fake"] = self._get_fake_provider()
        return self._providers["fake"]

    def generate_with_repair(
        self,
        agent_name: str,
        messages: list[dict[str, str]],
        output_schema: type[T],
        run_id: str = "run_default",
        parent_step_id: str | None = None,
        max_repairs: int = 2,
        budget_guard: Any | None = None,
        model: str | None = None,
    ) -> tuple[T, StepTrace]:
        """
        Executes structured generation with automatic retry-with-repair on schema validation failure.
        Max 2 repairs; all attempts count against budget and are logged in StepTrace.
        """
        provider, default_model = self.get_provider_for_agent(agent_name)
        chosen_model = model or default_model
        current_messages = list(messages)

        repair_attempts = 0
        total_tokens_in = 0
        total_tokens_out = 0
        total_cost_usd = 0.0
        start_time = time.time()
        last_error_msg = ""

        # Build provider fallback chain
        provider_chain: list[tuple[LLMProvider, str]] = [(provider, chosen_model)]
        
        # Add fallback providers based on available verified keys (only in non-override production mode)
        if self._override_provider is None:
            if provider.provider_name != "openrouter" and settings.openrouter_api_key:
                provider_chain.append((self._get_or_create_provider("openrouter"), settings.openrouter_model))
            if provider.provider_name != "groq" and settings.groq_api_key:
                provider_chain.append((self._get_or_create_provider("groq"), settings.groq_model))

        chain_idx = 0
        current_provider, current_model = provider_chain[0]

        while True:
            # Check BudgetGuard before invocation if provided
            if budget_guard is not None:
                budget_guard.check_before_call(estimated_tokens=500)

            try:
                response = self._call_with_backoff(current_provider, current_messages, output_schema, current_model)
                total_tokens_in += response.tokens_in
                total_tokens_out += response.tokens_out
                total_cost_usd += response.cost_usd

                if budget_guard is not None:
                    budget_guard.record_spend(tokens=response.total_tokens, usd_cost=response.cost_usd)

                latency_ms = (time.time() - start_time) * 1000.0
                trace = StepTrace(
                    run_id=run_id,
                    parent_step_id=parent_step_id,
                    agent=agent_name,
                    provider=current_provider.provider_name,
                    model=current_model,
                    input_data={"messages": self._redact_secrets(current_messages)},
                    output_data=response.parsed.model_dump() if hasattr(response.parsed, "model_dump") else {},
                    tokens_in=total_tokens_in,
                    tokens_out=total_tokens_out,
                    total_tokens=total_tokens_in + total_tokens_out,
                    usd_cost=round(total_cost_usd, 6),
                    latency_ms=round(latency_ms, 2),
                    retries=repair_attempts,
                    timestamp=time.time(),
                )
                return response.parsed, trace

            except RateLimitedError as rle:
                # Quota or rate limit error: try failing over to next provider in chain
                chain_idx += 1
                if chain_idx < len(provider_chain):
                    current_provider, current_model = provider_chain[chain_idx]
                    logger.warning(
                        f"Agent {agent_name} rate limit on {provider_chain[chain_idx - 1][0].provider_name}; "
                        f"failing over to {current_provider.provider_name} ({current_model})..."
                    )
                    continue
                raise rle

            except (InvalidOutputError, Exception) as e:
                repair_attempts += 1
                last_error_msg = str(e)
                logger.warning(
                    f"Agent {agent_name} output validation failed (attempt {repair_attempts}/{max_repairs + 1}): {e}"
                )

                if budget_guard is not None:
                    budget_guard.record_repair_attempt()

                if repair_attempts > max_repairs:
                    # Check if another provider in chain can succeed before aborting
                    chain_idx += 1
                    if chain_idx < len(provider_chain):
                        current_provider, current_model = provider_chain[chain_idx]
                        logger.warning(
                            f"Agent {agent_name} repair attempts exceeded on {provider_chain[chain_idx - 1][0].provider_name}; "
                            f"failing over to {current_provider.provider_name} ({current_model})..."
                        )
                        repair_attempts = 0
                        continue

                    latency_ms = (time.time() - start_time) * 1000.0
                    logger.error(
                        f"Agent {agent_name} failed after {max_repairs} repair attempts ({latency_ms:.1f}ms): {last_error_msg}"
                    )
                    raise InvalidOutputError(
                        f"Schema validation failed after {max_repairs} repair attempts: {last_error_msg}"
                    )

                # Formulate repair feedback message
                repair_instruction = (
                    f"CRITICAL ERROR: Your previous response was invalid. "
                    f"Validation error: {last_error_msg}. "
                    f"You MUST return strictly valid JSON conforming exactly to the {output_schema.__name__} schema: "
                    f"{json.dumps(output_schema.model_json_schema())}"
                )
                current_messages.append({"role": "user", "content": repair_instruction})

    def _call_with_backoff(
        self,
        provider: LLMProvider,
        messages: list[dict[str, str]],
        output_schema: type[T],
        model: str,
        max_retries: int | None = None,
    ) -> LLMResponse[T]:
        """Handles rate limits and transient errors with exponential backoff and jitter."""
        import random

        retries_limit = settings.llm_max_retries if max_retries is None else max_retries
        retries = 0
        while True:
            try:
                return provider.generate_structured(messages=messages, output_schema=output_schema, model=model)
            except RateLimitedError as rle:
                # If daily quota limit (TPD) was reached, fail over immediately to next provider
                if "tpd" in str(rle).lower() or "tokens per day" in str(rle).lower():
                    raise rle

                retries += 1
                if retries > retries_limit:
                    raise rle
                jitter = random.uniform(0.1, 0.5)
                # Cap backoff sleep to max 10.0 seconds to prevent multi-minute blocking on daily quotas
                base_sleep = min(rle.retry_after or 1.0, 10.0)
                sleep_dur = (base_sleep * (2 ** (retries - 1))) + jitter
                logger.info(f"Rate limited. Backing off for {sleep_dur:.2f}s (attempt {retries}/{retries_limit})...")
                time.sleep(sleep_dur)
            except Exception as e:
                err_lower = str(e).lower()
                if (
                    "timeout" in err_lower or "timed out" in err_lower or "connection" in err_lower
                ) and retries < retries_limit:
                    retries += 1
                    jitter = random.uniform(0.1, 0.5)
                    sleep_dur = (1.5**retries) + jitter
                    logger.warning(
                        f"Transient error ({e}). Retrying in {sleep_dur:.2f}s (attempt {retries}/{retries_limit})..."
                    )
                    time.sleep(sleep_dur)
                    continue
                raise e

    def _redact_secrets(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Redacts sensitive API keys or tokens from trace input."""
        redacted = []
        for msg in messages:
            content = msg.get("content", "")
            # Redact common key patterns
            for key in [settings.anthropic_api_key, settings.groq_api_key, settings.tavily_api_key]:
                if key and len(key) > 6:
                    content = content.replace(key, "[REDACTED_API_KEY]")
            redacted.append({"role": msg.get("role", "user"), "content": content})
        return redacted


llm_client = LLMClient()
