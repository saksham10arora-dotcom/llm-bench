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
        last_chunk_ns = 0
        async with self._client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        ) as s:
            async for _ in s.text_stream:
                last_chunk_ns = time.monotonic_ns()
                if first:
                    yield StreamEvent(EventType.FIRST_TOKEN, last_chunk_ns, 1)
                    first = False
                else:
                    yield StreamEvent(EventType.TOKEN, last_chunk_ns, 1)
            final = await s.get_final_message()
        # Total latency ends at the last content chunk, not after
        # get_final_message bookkeeping or connection teardown.
        yield StreamEvent(
            EventType.DONE,
            last_chunk_ns or time.monotonic_ns(),
            final.usage.output_tokens,
            prompt_tokens=final.usage.input_tokens,
        )
