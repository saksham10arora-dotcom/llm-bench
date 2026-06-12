from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import AsyncIterator, Protocol


class EventType(Enum):
    FIRST_TOKEN = auto()
    TOKEN = auto()
    DONE = auto()  # token_count carries total completion tokens from API usage


@dataclass
class StreamEvent:
    event_type: EventType
    ts_ns: int        # time.monotonic_ns() at event time
    token_count: int  # tokens in chunk; for DONE: total completion tokens


@dataclass
class RequestResult:
    ttft_ns: int | None       # None on error
    total_ns: int | None      # None on error
    completion_tokens: int
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None


class Adapter(Protocol):
    provider: str
    model: str

    def stream(self, prompt: str, max_tokens: int) -> AsyncIterator[StreamEvent]:
        ...
