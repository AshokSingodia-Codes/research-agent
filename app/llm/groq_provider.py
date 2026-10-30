import copy
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


def inline_schema_defs(schema: dict) -> dict:
    """Recursively inlines $defs and $ref in JSON schemas to satisfy Groq/LLM strict tool validation."""
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
        inlined_parameters = inline_schema_defs(output_schema.model_json_schema())
        tool_definition = {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": f"Output schema for {tool_name}",
                "parameters": inlined_parameters,
            },
        }

        # Provide casing aliases (e.g. research_findings, researchFindings, json)
        # to ensure tool calls succeed if the model normalizes name casing
        tool_definitions = [tool_definition]
        snake_name = "".join(["_" + c.lower() if c.isupper() and i > 0 else c.lower() for i, c in enumerate(tool_name)])
        camel_name = tool_name[0].lower() + tool_name[1:] if len(tool_name) > 1 else tool_name.lower()
        for alias in (snake_name, camel_name, "json", "result"):
            if alias != tool_name:
                tool_definitions.append({
                    "type": "function",
                    "function": {
                        "name": alias,
                        "description": f"Output schema alias for {tool_name}",
                        "parameters": inlined_parameters,
                    },
                })

        # Enhance system prompt to emphasize JSON schema compliance
        schema_instruction = (
            f"\nYou MUST output valid JSON complying exactly with schema {tool_name}: "
            f"{json.dumps(inlined_parameters)}"
        )
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({"role": msg["role"], "content": msg["content"]})

        if formatted_messages and formatted_messages[0]["role"] == "system":
            formatted_messages[0]["content"] += schema_instruction
        else:
            formatted_messages.insert(0, {"role": "system", "content": schema_instruction})

        t0 = time.time()
        # gpt-oss-120b supports full max_tokens; qwen on-demand has tight OTPM
        if "qwen" in model.lower():
            effective_max_tokens = min(max_tokens, 1000)
        else:
            effective_max_tokens = max_tokens

        try:
            if "gpt-oss" in model.lower():
                response = client.chat.completions.create(
                    model=model,
                    messages=formatted_messages,
                    response_format={"type": "json_object"},
                    temperature=temperature,
                    max_tokens=effective_max_tokens,
                )
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=formatted_messages,
                    tools=tool_definitions,
                    tool_choice="auto",
                    temperature=temperature,
                    max_tokens=effective_max_tokens,
                )
        except groq.RateLimitError as e:
            # Respect retry-after header and add jitter
            retry_after = 3.0
            if hasattr(e, "response") and e.response is not None:
                header_val = e.response.headers.get("retry-after")
                if header_val:
                    try:
                        retry_after = float(header_val)
                    except ValueError:
                        pass
            jitter = random.uniform(0.2, 0.6)
            raise RateLimitedError(f"Groq rate limit: {e}", retry_after=retry_after + jitter)
        except groq.APITimeoutError as e:
            raise TimeoutError(f"Groq API timeout: {e}")
        except groq.AuthenticationError as e:
            raise AuthError(f"Groq authentication failed: {e}")
        except Exception as e:
            err_msg = str(e)
            if "tool_use_failed" in err_msg or "failed to call a function" in err_msg.lower():
                # Qwen / open models sometimes misformat function call arguments; fallback to JSON mode
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=formatted_messages,
                        response_format={"type": "json_object"},
                        temperature=temperature,
                        max_tokens=effective_max_tokens,
                    )
                except Exception as inner_e:
                    raise LLMError(f"Groq JSON fallback failed: {inner_e}")
            elif "rate limit" in err_msg.lower() or "429" in err_msg or "tokens per minute" in err_msg.lower():
                raise RateLimitedError(f"Groq rate limit: {e}", retry_after=3.0)
            else:
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
            raw_output = choice.message.content.strip()
            # Clean markdown code fences if model wrapped response
            if raw_output.startswith("```"):
                lines = raw_output.splitlines()
                if lines and lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                raw_output = "\n".join(lines).strip()
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
