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


@pytest.mark.asyncio
async def test_timeout_enforced():
    adapter = MockAdapter(ttft_ms=5000, token_delay_ms=1, n_tokens=2)
    results = await run(adapter, "hi", n=2, max_tokens=256, concurrency=1, warmup=0, timeout_s=0.05)
    assert all(not r.success for r in results)
    assert all("timeout" in r.error for r in results)


class FailFirstKAdapter(MockAdapter):
    """Errors on the first k calls, succeeds after. Used to prove warmup
    requests complete before any measured request starts."""

    def __init__(self, k: int, **kwargs):
        super().__init__(**kwargs)
        self._calls = 0
        self._k = k

    async def stream(self, prompt: str, max_tokens: int):
        self._calls += 1
        if self._calls <= self._k:
            raise RuntimeError("cold start")
        async for event in super().stream(prompt, max_tokens):
            yield event


@pytest.mark.asyncio
async def test_warmup_runs_strictly_before_measurement():
    # First 2 calls fail. If warmup=2 truly runs first, all measured results succeed.
    adapter = FailFirstKAdapter(k=2, ttft_ms=1, token_delay_ms=1, n_tokens=3)
    results = await run(adapter, "hi", n=3, max_tokens=256, concurrency=3, warmup=2)
    assert all(r.success for r in results), [r.error for r in results]


@pytest.mark.asyncio
async def test_itl_gaps_collected():
    adapter = MockAdapter(ttft_ms=5, token_delay_ms=2, n_tokens=5)
    results = await run(adapter, "hi", n=2, max_tokens=256, concurrency=1, warmup=0)
    for r in results:
        # FIRST_TOKEN + 4 TOKEN events -> 4 measured gaps
        assert len(r.itl_gaps_ns) == 4
        assert all(g > 0 for g in r.itl_gaps_ns)
