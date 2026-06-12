from __future__ import annotations
import time
from collections.abc import AsyncIterator
import anthropic as sdk
from .base import EventType, StreamEvent


class AnthropicAdapter:
    provider = "anthropic"

    def __init__(self, model: str, api_key: str):
        self.model = model
        self._client = sdk.AsyncAnthropic(api_key=api_key)

    async def stream(self, prompt: str, max_tokens: int) -> AsyncIterator[StreamEvent]:
        first = True
        async with self._client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        ) as s:
            async for _ in s.text_stream:
                if first:
                    yield StreamEvent(EventType.FIRST_TOKEN, time.monotonic_ns(), 1)
                    first = False
                else:
                    yield StreamEvent(EventType.TOKEN, time.monotonic_ns(), 1)
            final = await s.get_final_message()
            completion_tokens = final.usage.output_tokens
        yield StreamEvent(EventType.DONE, time.monotonic_ns(), completion_tokens)
