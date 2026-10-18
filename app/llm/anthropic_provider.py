import json
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


class AnthropicProvider(LLMProvider):
    """
    Anthropic Claude API provider using tool-use forcing for strictly schema-valid JSON.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client = None

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def _get_client(self):
        if not self._client:
            if not self.api_key:
                raise AuthError("ANTHROPIC_API_KEY is missing or empty.")
            try:
                import anthropic

                self._client = anthropic.Anthropic(api_key=self.api_key)
            except Exception as e:
                raise AuthError(f"Failed to initialize Anthropic client: {e}")
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
        import anthropic

        # Extract system prompt if present
        system_prompt = ""
        user_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system_prompt += msg.get("content", "") + "\n"
            else:
                user_messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})

        if not user_messages:
            user_messages.append({"role": "user", "content": "Process according to instructions."})

        tool_name = output_schema.__name__
        tool_definition = {
            "name": tool_name,
            "description": f"Output schema for {tool_name}",
            "input_schema": output_schema.model_json_schema(),
        }

        t0 = time.time()
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt.strip() or "You are an AI research assistant.",
                messages=user_messages,
                tools=[tool_definition],
                tool_choice={"type": "tool", "name": tool_name},
            )
        except anthropic.RateLimitError as e:
            retry_after = 2.0
            if hasattr(e, "response") and e.response is not None:
                retry_header = e.response.headers.get("retry-after")
                if retry_header:
                    try:
                        retry_after = float(retry_header)
                    except ValueError:
                        pass
            raise RateLimitedError(f"Anthropic rate limit exceeded: {e}", retry_after=retry_after)
        except anthropic.APITimeoutError as e:
            raise TimeoutError(f"Anthropic API timeout: {e}")
        except anthropic.AuthenticationError as e:
            raise AuthError(f"Anthropic authentication failed: {e}")
        except Exception as e:
            raise LLMError(f"Anthropic API error: {e}")

        latency_ms = (time.time() - t0) * 1000.0

        # Parse tool use result
        parsed_obj: T | None = None
        raw_content = ""
        for block in response.content:
            if block.type == "tool_use" and block.name == tool_name:
                raw_content = json.dumps(block.input)
                try:
                    parsed_obj = output_schema.model_validate(block.input)
                except ValidationError as ve:
                    raise InvalidOutputError(f"Schema validation error: {ve}", raw_output=raw_content)
                break

        if parsed_obj is None:
            raise InvalidOutputError(
                f"Model failed to invoke tool {tool_name}",
                raw_output=str(response.content),
            )

        tokens_in = response.usage.input_tokens
        tokens_out = response.usage.output_tokens
        total_tokens = tokens_in + tokens_out
        cost = calculate_cost(self.provider_name, model, tokens_in, tokens_out)

        return LLMResponse(
            parsed=parsed_obj,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            total_tokens=total_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
            raw_text=raw_content,
            provider=self.provider_name,
            model=model,
        )
