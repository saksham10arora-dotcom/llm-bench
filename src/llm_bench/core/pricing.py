from __future__ import annotations
from dataclasses import dataclass

# (input $/1M tokens, output $/1M tokens)
# List prices only. Promotional/introductory rates are deliberately not encoded
# -- they expire, and a table that silently goes stale is worse than one that
# reports the durable number.
_PRICES: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-haiku-4-5": (1.0, 5.0),
    # OpenAI
    "gpt-4o": (2.50, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.0, 30.0),
    "gpt-3.5-turbo": (0.50, 1.50),
    # Groq (fast inference)
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
}


@dataclass
class CostEstimate:
    prompt_cost: float
    completion_cost: float
    total_cost: float


def estimate(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> CostEstimate | None:
    prices = _PRICES.get(model)
    if prices is None:
        return None
    input_rate, output_rate = prices
    prompt_cost = prompt_tokens * input_rate / 1_000_000
    completion_cost = completion_tokens * output_rate / 1_000_000
    return CostEstimate(
        prompt_cost=prompt_cost,
        completion_cost=completion_cost,
        total_cost=prompt_cost + completion_cost,
    )
