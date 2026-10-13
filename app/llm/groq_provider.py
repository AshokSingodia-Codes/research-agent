import json
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

T = TypeVar("T", bound=BaseModel)


class GroqProvider(LLMProvider):
    """
    Groq API provider (OpenAI-compatible) with tool-use / JSON schema forcing,
    exponential backoff with jitter on rate limits, and error mapping.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client = None

    @property
    def provider_name(self) -> str:
        return "groq"

    def _get_client(self):
        if not self._client:
            if not self.api_key:
                raise AuthError("GROQ_API_KEY is missing or empty.")
            try:
                import groq
                self._client = groq.Groq(api_key=self.api_key)
            except Exception as e:
                raise AuthError(f"Failed to initialize Groq client: {e}")
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
        import groq

        tool_name = output_schema.__name__
        tool_definition = {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": f"Output schema for {tool_name}",
                "parameters": output_schema.model_json_schema(),
            },
        }

        # Enhance system prompt to emphasize JSON schema compliance
        schema_instruction = (
            f"\nYou MUST output valid JSON complying exactly with schema {tool_name}: "
            f"{json.dumps(output_schema.model_json_schema())}"
        )
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({"role": msg["role"], "content": msg["content"]})

        if formatted_messages and formatted_messages[0]["role"] == "system":
            formatted_messages[0]["content"] += schema_instruction
        else:
            formatted_messages.insert(0, {"role": "system", "content": schema_instruction})

        t0 = time.time()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=formatted_messages,
                tools=[tool_definition],
                tool_choice={"type": "function", "function": {"name": tool_name}},
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except groq.RateLimitError as e:
            # Respect retry-after header and add jitter
            retry_after = 2.0
            if hasattr(e, "response") and e.response is not None:
                header_val = e.response.headers.get("retry-after")
                if header_val:
                    try:
                        retry_after = float(header_val)
                    except ValueError:
                        pass
            jitter = random.uniform(0.1, 0.5)
            raise RateLimitedError(f"Groq rate limit: {e}", retry_after=retry_after + jitter)
        except groq.APITimeoutError as e:
            raise TimeoutError(f"Groq API timeout: {e}")
        except groq.AuthenticationError as e:
            raise AuthError(f"Groq authentication failed: {e}")
        except Exception as e:
            raise LLMError(f"Groq API error: {e}")

        latency_ms = (time.time() - t0) * 1000.0

        choice = response.choices[0]
        raw_output = ""
        parsed_obj: T | None = None

        if choice.message.tool_calls:
            call = choice.message.tool_calls[0]
            raw_output = call.function.arguments
            try:
                parsed_dict = json.loads(raw_output)
                parsed_obj = output_schema.model_validate(parsed_dict)
            except (json.JSONDecodeError, ValidationError) as ve:
                raise InvalidOutputError(f"Failed to validate tool call output: {ve}", raw_output=raw_output)
        elif choice.message.content:
            raw_output = choice.message.content
            try:
                parsed_dict = json.loads(raw_output)
                parsed_obj = output_schema.model_validate(parsed_dict)
            except (json.JSONDecodeError, ValidationError) as ve:
                raise InvalidOutputError(f"Failed to validate JSON content: {ve}", raw_output=raw_output)
        else:
            raise InvalidOutputError("Model returned empty completion", raw_output="")

        tokens_in = response.usage.prompt_tokens if response.usage else 200
        tokens_out = response.usage.completion_tokens if response.usage else 300
        total_tokens = tokens_in + tokens_out
        cost = calculate_cost(self.provider_name, model, tokens_in, tokens_out)

        return LLMResponse(
            parsed=parsed_obj,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            total_tokens=total_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
            raw_text=raw_output,
            provider=self.provider_name,
            model=model,
        )
