import pytest
from llm_bench.core.stats import percentile, compute, Stats
from llm_bench.core.adapters.base import RequestResult


def make_result(
    ttft_ms: float, total_ms: float, tokens: int = 10, gaps_ms: list[float] | None = None
) -> RequestResult:
    return RequestResult(
        ttft_ns=int(ttft_ms * 1e6),
        total_ns=int(total_ms * 1e6),
        completion_tokens=tokens,
        itl_gaps_ns=[int(g * 1e6) for g in (gaps_ms or [])],
    )


def make_error() -> RequestResult:
    return RequestResult(ttft_ns=None, total_ns=None, completion_tokens=0, error="timeout")


class TestPercentile:
    def test_median_odd(self):
        assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50) == 3.0

    def test_p99_large(self):
        values = [float(i) for i in range(1, 101)]  # 1..100 sorted
        assert percentile(values, 99) == 99.0

    def test_single_element(self):
        assert percentile([42.0], 99) == 42.0

    def test_ties(self):
        assert percentile([5.0, 5.0, 5.0], 50) == 5.0

    def test_empty_returns_zero(self):
        assert percentile([], 50) == 0.0


class TestCompute:
    def test_excludes_errors_from_latency(self):
        results = [make_result(100, 500), make_error()]
        s = compute(results)
        assert s.error_count == 1
        assert s.total_count == 2
        assert s.ttft_p50 == pytest.approx(100.0, rel=0.01)

    def test_throughput_is_decode_rate(self):
        # ttft=100ms, total=1000ms, 10 tokens: 9 decode tokens over 0.9s = 10 tok/s.
        # TTFT must NOT be in the denominator (that would blend prefill into decode).
        results = [make_result(100, 1000, tokens=10), make_result(100, 1000, tokens=10)]
        s = compute(results)
        assert s.throughput_tps == pytest.approx(10.0, rel=0.01)

    def test_all_errors_returns_zero_stats(self):
        results = [make_error(), make_error()]
        s = compute(results)
        assert s.error_count == 2
        assert s.ttft_p50 == 0.0
        assert s.throughput_tps == 0.0

    def test_itl_from_measured_gaps(self):
        results = [make_result(100, 200, tokens=11, gaps_ms=[10.0] * 10)]
        s = compute(results)
        assert s.itl_mean == pytest.approx(10.0, rel=0.01)
        assert s.itl_p50 == pytest.approx(10.0, rel=0.01)

    def test_itl_p99_catches_tail_spikes_across_requests(self):
        # 100 pooled gaps, 2 of them spikes (top 2%). Per-request averaging
        # would dilute them to ~12ms; pooled-gap p99 must surface 500ms.
        fast = make_result(100, 300, tokens=51, gaps_ms=[2.0] * 50)
        spiky = make_result(100, 300, tokens=51, gaps_ms=[2.0] * 48 + [500.0, 500.0])
        s = compute([fast, spiky])
        assert s.itl_p99 == pytest.approx(500.0, rel=0.01)
        assert s.itl_p50 == pytest.approx(2.0, rel=0.01)
