import logging
import os

logger = logging.getLogger(__name__)

# Fallback pricing constants (highest rate in standard catalog, overridable via env)
DEFAULT_UNKNOWN_INPUT_PRICE = 15.00  # USD per 1M tokens
DEFAULT_UNKNOWN_OUTPUT_PRICE = 75.00  # USD per 1M tokens

# Pricing in USD per 1,000,000 tokens (input_price_per_1m, output_price_per_1m)
MODEL_PRICING: dict[tuple[str, str], tuple[float, float]] = {
    # Anthropic Published Rates
    ("anthropic", "claude-3-5-sonnet-20241022"): (3.00, 15.00),
    ("anthropic", "claude-3-5-sonnet-latest"): (3.00, 15.00),
    ("anthropic", "claude-3-5-haiku-20241022"): (0.80, 4.00),
    ("anthropic", "claude-3-5-haiku-latest"): (0.80, 4.00),
    ("anthropic", "claude-3-opus-20240229"): (15.00, 75.00),
    ("anthropic", "claude-3-opus-latest"): (15.00, 75.00),
    ("anthropic", "claude-haiku-5-5"): (0.80, 4.00),
    ("anthropic", "claude-sonnet-5-5"): (3.00, 15.00),
    # Groq Published Rates
    ("groq", "llama-3.3-70b-versatile"): (0.59, 0.79),
    ("groq", "llama-3.1-70b-versatile"): (0.59, 0.79),
    ("groq", "llama-3.1-8b-instant"): (0.05, 0.08),
    ("groq", "mixtral-8x7b-32768"): (0.24, 0.24),
    ("groq", "gemma2-9b-it"): (0.20, 0.20),
    ("groq", "llama-3.2-1b-preview"): (0.04, 0.04),
    ("groq", "llama-3.2-3b-preview"): (0.06, 0.06),
    # Fake / Test Provider
    ("fake", "fake-model"): (0.0, 0.0),
}

_WARNED_UNKNOWN_MODELS: set[str] = set()
unknown_model_lookups_count: int = 0


def get_unknown_model_count() -> int:
    """Returns the total number of unknown model pricing lookups."""
    return unknown_model_lookups_count


def reset_unknown_model_count() -> None:
    """Resets the unknown model lookups counter."""
    global unknown_model_lookups_count
    unknown_model_lookups_count = 0


def get_fallback_rates() -> tuple[float, float]:
    """Returns conservative input and output pricing per 1M tokens from env or defaults."""
    try:
        in_price = float(os.environ.get("UNKNOWN_MODEL_INPUT_USD_PER_MTOK", DEFAULT_UNKNOWN_INPUT_PRICE))
    except ValueError:
        in_price = DEFAULT_UNKNOWN_INPUT_PRICE

    try:
        out_price = float(os.environ.get("UNKNOWN_MODEL_OUTPUT_USD_PER_MTOK", DEFAULT_UNKNOWN_OUTPUT_PRICE))
    except ValueError:
        out_price = DEFAULT_UNKNOWN_OUTPUT_PRICE

    return in_price, out_price


def calculate_cost(provider: str, model: str, tokens_in: int, tokens_out: int) -> float:
    """
    Calculates exact USD cost for a given provider, model, and token counts.
    Unknown models use a conservative fallback rate (> 0) and log a warning once per model.
    """
    global unknown_model_lookups_count

    p_norm = provider.lower()
    m_norm = model.lower()
    key = (p_norm, m_norm)

    if key in MODEL_PRICING:
        in_price, out_price = MODEL_PRICING[key]
    else:
        # Check partial match within the same provider
        matched = False
        in_price, out_price = (0.0, 0.0)
        for (p, m), (ip, op) in MODEL_PRICING.items():
            if p == p_norm and (m in m_norm or m_norm in m):
                in_price, out_price = ip, op
                matched = True
                break

        if not matched:
            if p_norm == "fake":
                in_price, out_price = (0.0, 0.0)
            else:
                unknown_model_lookups_count += 1
                fallback_in, fallback_out = get_fallback_rates()
                model_identifier = f"{provider}:{model}"
                if model_identifier not in _WARNED_UNKNOWN_MODELS:
                    _WARNED_UNKNOWN_MODELS.add(model_identifier)
                    logger.warning(
                        f"Unknown model '{model}' for provider '{provider}'. "
                        f"Applying conservative fallback pricing: ${fallback_in:.2f}/M in, ${fallback_out:.2f}/M out."
                    )
                in_price, out_price = fallback_in, fallback_out

    cost = (tokens_in / 1_000_000.0) * in_price + (tokens_out / 1_000_000.0) * out_price
    return round(cost, 6)
