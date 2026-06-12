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

    async def stream(self, prompt: str, max_tokens: int) -> AsyncIterator[StreamEvent]:
        first = True
        completion_tokens = 0
        response = await self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                if first:
                    yield StreamEvent(EventType.FIRST_TOKEN, time.monotonic_ns(), 1)
                    first = False
                else:
                    yield StreamEvent(EventType.TOKEN, time.monotonic_ns(), 1)
            if chunk.usage:
                completion_tokens = chunk.usage.completion_tokens
        yield StreamEvent(EventType.DONE, time.monotonic_ns(), completion_tokens)
