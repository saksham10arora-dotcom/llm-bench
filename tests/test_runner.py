import asyncio
import pytest
import time
from llm_bench.core.runner import run
from llm_bench.core.adapters.mock import MockAdapter


@pytest.mark.asyncio
async def test_returns_n_results():
    adapter = MockAdapter(ttft_ms=10, token_delay_ms=2, n_tokens=5)
    results = await run(adapter, "hi", n=10, max_tokens=256, concurrency=1, warmup=0)
    assert len(results) == 10


@pytest.mark.asyncio
async def test_warmup_excluded():
    adapter = MockAdapter(ttft_ms=10, token_delay_ms=2, n_tokens=5)
    results = await run(adapter, "hi", n=10, max_tokens=256, concurrency=1, warmup=3)
    assert len(results) == 10  # warmup runs happen but are excluded from returned list


@pytest.mark.asyncio
async def test_ttft_less_than_total():
    adapter = MockAdapter(ttft_ms=50, token_delay_ms=10, n_tokens=5)
    results = await run(adapter, "hi", n=5, max_tokens=256, concurrency=1, warmup=0)
    for r in results:
        assert r.success
        assert r.ttft_ns < r.total_ns


@pytest.mark.asyncio
async def test_errors_counted():
    adapter = MockAdapter(error_message="API error")
    results = await run(adapter, "hi", n=5, max_tokens=256, concurrency=1, warmup=0)
    assert all(not r.success for r in results)
    assert all(r.error == "API error" for r in results)


@pytest.mark.asyncio
async def test_concurrency_respected():
    # 5 tokens * 20ms delay = 100ms per request sequential. With concurrency=5, should be ~100ms total.
    adapter = MockAdapter(ttft_ms=0, token_delay_ms=20, n_tokens=5)
    start = time.monotonic()
    await run(adapter, "hi", n=5, max_tokens=256, concurrency=5, warmup=0)
    elapsed = time.monotonic() - start
    # Sequential would be ~500ms, concurrent should be ~100ms. Allow 5x for CI jitter.
    assert elapsed < 1.0, f"Expected < 1.0s but took {elapsed:.2f}s"
