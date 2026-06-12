from __future__ import annotations
import asyncio
import time
from .adapters.base import Adapter, EventType, RequestResult


async def _run_one(adapter: Adapter, prompt: str, max_tokens: int) -> RequestResult:
    start_ns = time.monotonic_ns()
    ttft_ns: int | None = None
    total_ns: int | None = None
    completion_tokens = 0

    try:
        async for event in adapter.stream(prompt, max_tokens):
            if event.event_type == EventType.FIRST_TOKEN and ttft_ns is None:
                ttft_ns = event.ts_ns - start_ns
            if event.event_type in (EventType.FIRST_TOKEN, EventType.TOKEN):
                completion_tokens += event.token_count
            if event.event_type == EventType.DONE:
                total_ns = event.ts_ns - start_ns
                completion_tokens = event.token_count  # use API's final count
        return RequestResult(ttft_ns=ttft_ns, total_ns=total_ns, completion_tokens=completion_tokens)
    except Exception as e:
        return RequestResult(ttft_ns=None, total_ns=None, completion_tokens=0, error=str(e))


async def run(
    adapter: Adapter,
    prompt: str,
    n: int,
    max_tokens: int = 256,
    concurrency: int = 1,
    warmup: int = 3,
) -> list[RequestResult]:
    sem = asyncio.Semaphore(concurrency)

    async def bounded(_: int) -> RequestResult:
        async with sem:
            return await _run_one(adapter, prompt, max_tokens)

    total = warmup + n
    all_results = await asyncio.gather(*[bounded(i) for i in range(total)])
    return list(all_results[warmup:])
