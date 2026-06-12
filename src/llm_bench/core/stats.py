from __future__ import annotations
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
    itl_mean: float
    itl_p99: float
    throughput_tps: float
    error_count: int
    total_count: int


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile. Input must be sorted ascending."""
    if not values:
        return 0.0
    # For p=50 with 5 elements: rank should be 3 (index 2)
    # Formula: ceil(p / 100 * len(values))
    import math
    rank = math.ceil(p / 100 * len(values))
    return values[rank - 1]


def _mean(lst: list[float]) -> float:
    return sum(lst) / len(lst) if lst else 0.0


def compute(results: list[RequestResult]) -> Stats:
    successful = [r for r in results if r.success]

    ttft_ms = sorted(r.ttft_ns / 1e6 for r in successful)
    total_ms = sorted(r.total_ns / 1e6 for r in successful)

    total_tokens = sum(r.completion_tokens for r in successful)
    total_time_s = sum(r.total_ns / 1e9 for r in successful)
    throughput = total_tokens / total_time_s if total_time_s > 0 else 0.0

    itl_values: list[float] = []
    for r in successful:
        if r.completion_tokens > 1:
            gap_ms = (r.total_ns - r.ttft_ns) / 1e6
            itl_values.append(gap_ms / (r.completion_tokens - 1))
    itl_sorted = sorted(itl_values)

    return Stats(
        ttft_p50=percentile(ttft_ms, 50),
        ttft_p95=percentile(ttft_ms, 95),
        ttft_p99=percentile(ttft_ms, 99),
        ttft_mean=_mean(ttft_ms),
        total_p50=percentile(total_ms, 50),
        total_p95=percentile(total_ms, 95),
        total_p99=percentile(total_ms, 99),
        total_mean=_mean(total_ms),
        itl_mean=_mean(itl_sorted),
        itl_p99=percentile(itl_sorted, 99),
        throughput_tps=throughput,
        error_count=len(results) - len(successful),
        total_count=len(results),
    )
