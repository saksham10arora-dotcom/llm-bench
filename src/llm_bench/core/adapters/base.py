from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import AsyncIterator, Protocol


class EventType(Enum):
    FIRST_TOKEN = auto()
    TOKEN = auto()
    DONE = auto()  # token_count carries total completion tokens from API usage


@dataclass
class StreamEvent:
    event_type: EventType
    ts_ns: int          # time.monotonic_ns() at event time; for DONE: when the last content chunk arrived
    token_count: int    # tokens in chunk; for DONE: total completion tokens
    prompt_tokens: int = 0  # only meaningful on DONE, from API usage (0 if endpoint omits usage)
    cached_tokens: int = 0  # prompt tokens served from cache (DONE only); 0 if none or unreported


@dataclass
class RequestResult:
    ttft_ns: int | None       # None on error
    total_ns: int | None      # None on error
    completion_tokens: int
    error: str | None = None
    prompt_tokens: int = 0    # exact count from API usage; 0 if unavailable
    cached_tokens: int = 0    # prompt tokens read from cache; >0 means a warm hit
    itl_gaps_ns: list[int] = field(default_factory=list)  # measured gaps between consecutive chunks

    @property
    def success(self) -> bool:
        return self.error is None

    @property
    def cache_hit(self) -> bool:
        # A request with no reported cache read is treated as a cold miss, not
        # unknown. Cold and warm are two latency regimes and averaging across
        # them hides the split, so every result must land in one bucket.
        return self.cached_tokens > 0


class Adapter(Protocol):
    provider: str
    model: str

    def stream(self, prompt: str, max_tokens: int) -> AsyncIterator[StreamEvent]:
        ...
