import copy
import json
import logging
import random
import time
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.llm.base import (
    AuthError,
    InvalidOutputError,
    LLMError,
    LLMProvider,
    LLMResponse,
    RateLimitedError,
    TimeoutError,
)
from app.llm.pricing import calculate_cost

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


def inline_schema_defs(schema: dict) -> dict:
    """Recursively inlines $defs and $ref in JSON schemas to satisfy strict tool validation."""
    defs_key = [k for k in schema.keys() if "defs" in k]
    defs = schema.get(defs_key[0], {}) if defs_key else {}

    def _walk(obj):
        if isinstance(obj, dict):
            ref_val = None
            for k in list(obj.keys()):
                if "ref" in k and isinstance(obj[k], str) and "/" in obj[k]:
                    ref_val = obj[k]
                    break
            if ref_val:
                ref_key = ref_val.split("/")[-1]
                if ref_key in defs:
                    resolved = copy.deepcopy(defs[ref_key])
                    resolved = _walk(resolved)
                    for k, v in obj.items():
                        if "ref" not in k:
                            resolved[k] = _walk(v)
                    return resolved
            return {k: _walk(v) for k, v in obj.items() if "defs" not in k}
        elif isinstance(obj, list):
            return [_walk(x) for x in obj]
        return obj

    return _walk(schema)


class OpenAICompatibleProvider(LLMProvider):
    """
    Generic OpenAI-compatible provider supporting:
    - OpenRouter (base_url="https://openrouter.ai/api/v1")
    - Google Gemini via OpenAI endpoint (base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    - Grok / xAI (base_url="https://api.x.ai/v1")
    - DeepSeek, Cerebras, Ollama, etc.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        provider_name: str = "openai_compatible",
        default_headers: dict[str, str] | None = None,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self._provider_name = provider_name
        self._default_headers = default_headers or {}
        self._client = None

    @property
    def provider_name(self) -> str:
        return self._provider_name

    def _get_client(self):
        if not self._client:
            if not self.api_key:
                raise AuthError(f"{self.provider_name.upper()}_API_KEY is missing or empty.")
            try:
                from openai import OpenAI

                kwargs: dict = {"api_key": self.api_key}
                if self.base_url:
                    kwargs["base_url"] = self.base_url
                if self._default_headers:
                    kwargs["default_headers"] = self._default_headers

                self._client = OpenAI(**kwargs)
            except Exception as e:
                raise AuthError(f"Failed to initialize {self.provider_name} client: {e}")
        return self._client

    def generate_structured(
        self,
        messages: list[dict[str, str]],
        output_schema: type[T],
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.1,
    ) -> LLMResponse[T]:
        client = self._get_client()

        tool_name = output_schema.__name__
        tool_definition = {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": f"Output schema formatted according to {tool_name}",
                "parameters": inline_schema_defs(output_schema.model_json_schema()),
            },
        }

        start_time = time.time()
        max_retries = 3
        backoff_base = 1.0

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=[tool_definition],
                    tool_choice={"type": "function", "function": {"name": tool_name}},
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                break
            except Exception as e:
                err_str = str(e).lower()
                status_code = getattr(e, "status_code", None)

                if (
                    "rate limit" in err_str
                    or "429" in err_str
                    or status_code == 429
                    or "tokens per minute" in err_str
                ):
                    if attempt < max_retries - 1:
                        sleep_time = (backoff_base * (2**attempt)) + random.uniform(0.1, 0.5)
                        logger.warning(
                            f"{self.provider_name} rate limit on attempt {attempt + 1}. Retrying in {sleep_time:.2f}s..."
                        )
                        time.sleep(sleep_time)
                        continue
                    raise RateLimitedError(f"{self.provider_name} rate limit exceeded: {e}")

                if "auth" in err_str or "unauthorized" in err_str or "key" in err_str or status_code in (401, 403):
                    raise AuthError(f"{self.provider_name} authentication failed: {e}")

                if "timeout" in err_str or status_code == 408:
                    raise TimeoutError(f"{self.provider_name} request timed out: {e}")

                raise LLMError(f"{self.provider_name} completion failed: {e}")

        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        completion_tokens = response.usage.completion_tokens if response.usage else 0
        total_tokens = response.usage.total_tokens if response.usage else prompt_tokens + completion_tokens

        cost_usd = calculate_cost(self.provider_name, model, prompt_tokens, completion_tokens)

        choice = response.choices[0]
        tool_calls = getattr(choice.message, "tool_calls", None)

        raw_json = None
        if tool_calls and len(tool_calls) > 0:
            call = tool_calls[0]
            raw_json = call.function.arguments
        elif choice.message.content:
            raw_json = choice.message.content

        if not raw_json:
            raise InvalidOutputError(f"{self.provider_name} returned empty output or missing tool call.")

        # Clean markdown code fences and whitespace
        clean_json = raw_json.strip()
        if clean_json.startswith("```"):
            lines = clean_json.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            clean_json = "\n".join(lines).strip()

        try:
            parsed_data = json.loads(clean_json)
            validated_output = output_schema.model_validate(parsed_data)
        except (json.JSONDecodeError, ValidationError) as e:
            raise InvalidOutputError(f"{self.provider_name} output validation failed: {e}", raw_output=raw_json)

        latency_ms = (time.time() - start_time) * 1000.0

        return LLMResponse(
            parsed=validated_output,
            tokens_in=prompt_tokens,
            tokens_out=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            raw_text=raw_json,
            provider=self.provider_name,
            model=model,
        )
