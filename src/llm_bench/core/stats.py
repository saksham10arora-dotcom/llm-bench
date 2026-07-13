from __future__ import annotations
import math
from dataclasses import dataclass
from .adapters.base import RequestResult


@dataclass
class Stats:
    ttft_p50: float
    ttft_p95: float
    ttft_p99: float
    ttft_mean: float
    total_p50: float
    total_p95: float
    total_p99: float
    total_mean: float
    itl_p50: float
    itl_p95: float
    itl_p99: float
    itl_mean: float
    throughput_tps: float
    error_count: int
    total_count: int


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile. Input must be sorted ascending."""
    if not values:
        return 0.0
    rank = math.ceil(p / 100 * len(values))
    return values[rank - 1]


def _mean(lst: list[float]) -> float:
    return sum(lst) / len(lst) if lst else 0.0


def split_by_cache(
    results: list[RequestResult],
) -> tuple[list[RequestResult], list[RequestResult]]:
    """Partition successful results into (cold, warm) by cache hit.

    Warm requests read prompt tokens from cache and can be dramatically faster,
    so a single blended percentile describes neither regime. Errors are dropped
    because they carry no valid latency. Compute stats on each list separately.
    """
    successful = [r for r in results if r.success]
    cold = [r for r in successful if not r.cache_hit]
    warm = [r for r in successful if r.cache_hit]
    return cold, warm


def compute(results: list[RequestResult]) -> Stats:
    successful = [r for r in results if r.success]

    ttft_ms = sorted(r.ttft_ns / 1e6 for r in successful)
    total_ms = sorted(r.total_ns / 1e6 for r in successful)

    # Throughput is decode rate: tokens generated after the first one, over
    # time spent generating them. Including TTFT would blend queueing and
    # prefill into a number that claims to be generation speed.
    decode_tokens = 0
    decode_time_s = 0.0
    for r in successful:
        if r.completion_tokens > 1 and r.total_ns > r.ttft_ns:
            decode_tokens += r.completion_tokens - 1
            decode_time_s += (r.total_ns - r.ttft_ns) / 1e9
    throughput = decode_tokens / decode_time_s if decode_time_s > 0 else 0.0

    # ITL percentiles come from every measured chunk gap across all requests,
    # not from per-request averages (averaging first would hide tail spikes).
    gaps_ms = sorted(g / 1e6 for r in successful for g in r.itl_gaps_ns)

    return Stats(
        ttft_p50=percentile(ttft_ms, 50),
        ttft_p95=percentile(ttft_ms, 95),
        ttft_p99=percentile(ttft_ms, 99),
        ttft_mean=_mean(ttft_ms),
        total_p50=percentile(total_ms, 50),
        total_p95=percentile(total_ms, 95),
        total_p99=percentile(total_ms, 99),
        total_mean=_mean(total_ms),
        itl_p50=percentile(gaps_ms, 50),
        itl_p95=percentile(gaps_ms, 95),
        itl_p99=percentile(gaps_ms, 99),
        itl_mean=_mean(gaps_ms),
        throughput_tps=throughput,
        error_count=len(results) - len(successful),
        total_count=len(results),
    )
