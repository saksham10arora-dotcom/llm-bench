from __future__ import annotations
import time
from collections.abc import AsyncIterator
from openai import AsyncOpenAI
from .base import EventType, StreamEvent


class OpenAIAdapter:
    provider = "openai"

    def __init__(self, model: str, api_key: str, base_url: str | None = None):
        self.model = model
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def _create(self, prompt: str, max_tokens: int):
        kwargs = dict(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        try:
            return await self._client.chat.completions.create(
                **kwargs, stream_options={"include_usage": True}
            )
        except Exception as e:
            # Some OpenAI-compatible endpoints (older vLLM, some proxies)
            # reject stream_options. Fall back to chunk counting.
            if "stream_options" not in str(e):
                raise
            return await self._client.chat.completions.create(**kwargs)

    async def stream(self, prompt: str, max_tokens: int) -> AsyncIterator[StreamEvent]:
        first = True
        chunk_count = 0
        last_chunk_ns = 0
        usage_completion: int | None = None
        usage_prompt = 0
        response = await self._create(prompt, max_tokens)
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                chunk_count += 1
                last_chunk_ns = time.monotonic_ns()
                if first:
                    yield StreamEvent(EventType.FIRST_TOKEN, last_chunk_ns, 1)
                    first = False
                else:
                    yield StreamEvent(EventType.TOKEN, last_chunk_ns, 1)
            if chunk.usage:
                usage_completion = chunk.usage.completion_tokens
                usage_prompt = chunk.usage.prompt_tokens
        # Prefer exact usage; fall back to chunk count when the endpoint
        # never sent usage. Total ends at the last content chunk, not at
        # the trailing usage frame.
        yield StreamEvent(
            EventType.DONE,
            last_chunk_ns or time.monotonic_ns(),
            usage_completion if usage_completion is not None else chunk_count,
            prompt_tokens=usage_prompt,
        )
