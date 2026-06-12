from __future__ import annotations
import asyncio
import time
from collections.abc import AsyncIterator
from .base import EventType, StreamEvent


class MockAdapter:
    """Fake adapter with scripted delays. No network calls."""

    def __init__(
        self,
        provider: str = "mock",
        model: str = "mock-model",
        ttft_ms: float = 100.0,
        token_delay_ms: float = 20.0,
        n_tokens: int = 10,
        error_message: str | None = None,
    ):
        self.provider = provider
        self.model = model
        self._ttft_ms = ttft_ms
        self._token_delay_ms = token_delay_ms
        self._n_tokens = n_tokens
        self._error_message = error_message

    async def stream(self, prompt: str, max_tokens: int) -> AsyncIterator[StreamEvent]:
        if self._error_message:
            raise RuntimeError(self._error_message)
        await asyncio.sleep(self._ttft_ms / 1000)
        yield StreamEvent(EventType.FIRST_TOKEN, time.monotonic_ns(), 1)
        for _ in range(self._n_tokens - 1):
            await asyncio.sleep(self._token_delay_ms / 1000)
            yield StreamEvent(EventType.TOKEN, time.monotonic_ns(), 1)
        yield StreamEvent(EventType.DONE, time.monotonic_ns(), self._n_tokens)
