import pytest
from llm_bench.core.stats import percentile, compute, Stats
from llm_bench.core.adapters.base import RequestResult


def make_result(ttft_ms: float, total_ms: float, tokens: int = 10) -> RequestResult:
    return RequestResult(
        ttft_ns=int(ttft_ms * 1e6),
        total_ns=int(total_ms * 1e6),
        completion_tokens=tokens,
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

    def test_throughput(self):
        # 2 results, each 10 tokens, total 2s
        results = [make_result(100, 1000, tokens=10), make_result(100, 1000, tokens=10)]
        s = compute(results)
        assert s.throughput_tps == pytest.approx(10.0, rel=0.01)  # 20 tokens / 2s

    def test_all_errors_returns_zero_stats(self):
        results = [make_error(), make_error()]
        s = compute(results)
        assert s.error_count == 2
        assert s.ttft_p50 == 0.0
        assert s.throughput_tps == 0.0

    def test_itl_computed(self):
        # ttft=100ms, total=200ms, 11 tokens -> 10 gaps of 10ms each
        results = [make_result(100, 200, tokens=11)]
        s = compute(results)
        assert s.itl_mean == pytest.approx(10.0, rel=0.1)
