from __future__ import annotations
import asyncio
import time
from .adapters.base import Adapter, EventType, RequestResult


async def _measure(adapter: Adapter, prompt: str, max_tokens: int) -> RequestResult:
    start_ns = time.monotonic_ns()
    ttft_ns: int | None = None
    total_ns: int | None = None
    completion_tokens = 0
    prompt_tokens = 0
    cached_tokens = 0
    gaps: list[int] = []
    prev_chunk_ns: int | None = None

    async for event in adapter.stream(prompt, max_tokens):
        if event.event_type == EventType.FIRST_TOKEN and ttft_ns is None:
            ttft_ns = event.ts_ns - start_ns
            prev_chunk_ns = event.ts_ns
        elif event.event_type == EventType.TOKEN:
            if prev_chunk_ns is not None:
                gaps.append(event.ts_ns - prev_chunk_ns)
            prev_chunk_ns = event.ts_ns
            completion_tokens += event.token_count
        elif event.event_type == EventType.DONE:
            total_ns = event.ts_ns - start_ns
            if event.token_count > 0:
                completion_tokens = event.token_count  # prefer API's exact count
            prompt_tokens = event.prompt_tokens
            cached_tokens = event.cached_tokens
    if ttft_ns is None:
        # Stream completed without a single content chunk (e.g. a reasoning
        # model that spent the whole token budget before emitting output).
        # There is no valid latency to measure, so this is an error, not a
        # success with ttft_ns=None -- stats must never see that.
        return RequestResult(ttft_ns=None, total_ns=None, completion_tokens=0,
                             error="no content tokens received")
    return RequestResult(
        ttft_ns=ttft_ns,
        total_ns=total_ns,
        completion_tokens=completion_tokens,
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
        itl_gaps_ns=gaps,
    )


async def _run_one(
    adapter: Adapter, prompt: str, max_tokens: int, timeout_s: float | None
) -> RequestResult:
    try:
        if timeout_s is not None:
            return await asyncio.wait_for(_measure(adapter, prompt, max_tokens), timeout_s)
        return await _measure(adapter, prompt, max_tokens)
    except asyncio.TimeoutError:
        return RequestResult(ttft_ns=None, total_ns=None, completion_tokens=0,
                             error=f"timeout after {timeout_s}s")
    except Exception as e:
        return RequestResult(ttft_ns=None, total_ns=None, completion_tokens=0, error=str(e))


async def run(
    adapter: Adapter,
    prompt: str,
    n: int,
    max_tokens: int = 256,
    concurrency: int = 1,
    warmup: int = 3,
    timeout_s: float | None = None,
) -> list[RequestResult]:
    sem = asyncio.Semaphore(concurrency)

    async def bounded() -> RequestResult:
        async with sem:
            return await _run_one(adapter, prompt, max_tokens, timeout_s)

    # Warmup is a strict phase: all warmup requests finish before measurement
    # starts, so connection setup and server-side cache effects never leak
    # into the measured distribution.
    if warmup > 0:
        await asyncio.gather(*[bounded() for _ in range(warmup)])

    results = await asyncio.gather(*[bounded() for _ in range(n)])
    return list(results)
