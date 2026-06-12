import pytest
from llm_bench.core.pricing import estimate, CostEstimate


def test_known_model_returns_estimate():
    result = estimate("claude-sonnet-4-6", prompt_tokens=1000, completion_tokens=500)
    assert isinstance(result, CostEstimate)
    assert result.total_cost > 0


def test_unknown_model_returns_none():
    result = estimate("gpt-99-ultra", prompt_tokens=1000, completion_tokens=500)
    assert result is None


def test_cost_math_anthropic_sonnet():
    # claude-sonnet-4-6: $3/1M input, $15/1M output
    result = estimate("claude-sonnet-4-6", prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert result.prompt_cost == pytest.approx(3.0, rel=0.001)
    assert result.completion_cost == pytest.approx(15.0, rel=0.001)
    assert result.total_cost == pytest.approx(18.0, rel=0.001)


def test_cost_math_openai_gpt4o():
    # gpt-4o: $2.50/1M input, $10/1M output
    result = estimate("gpt-4o", prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert result.prompt_cost == pytest.approx(2.50, rel=0.001)
    assert result.completion_cost == pytest.approx(10.0, rel=0.001)


def test_zero_tokens():
    result = estimate("claude-sonnet-4-6", prompt_tokens=0, completion_tokens=0)
    assert result.total_cost == 0.0
