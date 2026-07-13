import pytest

from llm_bench.core.adapters.base import RequestResult
from llm_bench.core.adapters.mock import MockAdapter
from llm_bench.core.runner import run
from llm_bench.core.stats import split_by_cache


def _r(cached, err=None):
    return RequestResult(ttft_ns=100, total_ns=200, completion_tokens=5,
                         cached_tokens=cached, error=err)


def test_cache_hit_property():
    assert _r(50).cache_hit
    assert not _r(0).cache_hit


def test_cache_hit_defaults_false():
    # A result built without cache info is treated as a cold miss, not unknown.
    assert not RequestResult(ttft_ns=1, total_ns=2, completion_tokens=3).cache_hit


@pytest.mark.asyncio
async def test_runner_threads_cached_tokens():
    adapter = MockAdapter(ttft_ms=5, token_delay_ms=1, n_tokens=4, cached_tokens=64)
    results = await run(adapter, "hi", n=3, max_tokens=64, concurrency=1, warmup=0)
    assert all(r.cached_tokens == 64 for r in results)
    assert all(r.cache_hit for r in results)


def test_split_by_cache_partitions_cold_and_warm():
    cold, warm = split_by_cache([_r(0), _r(64), _r(0), _r(128)])
    assert len(cold) == 2 and len(warm) == 2
    assert all(not r.cache_hit for r in cold)
    assert all(r.cache_hit for r in warm)


def test_split_by_cache_excludes_errors():
    cold, warm = split_by_cache([_r(0), _r(0, err="boom"), _r(64)])
    assert len(cold) == 1 and len(warm) == 1


def test_split_all_cold_when_no_cache():
    cold, warm = split_by_cache([_r(0), _r(0)])
    assert len(cold) == 2 and warm == []


def test_write_json_includes_cache_split(tmp_path):
    import json

    from llm_bench.core.report import write_json

    results = [
        RequestResult(ttft_ns=100_000_000, total_ns=200_000_000,
                      completion_tokens=5, cached_tokens=0),
        RequestResult(ttft_ns=20_000_000, total_ns=120_000_000,
                      completion_tokens=5, cached_tokens=64),
    ]
    p = tmp_path / "out.json"
    write_json(p, results, {"provider": "mock"})
    data = json.loads(p.read_text())
    assert data["cache"]["cold_count"] == 1
    assert data["cache"]["warm_count"] == 1
    assert data["cache"]["cold_ttft_p99"] == 100.0  # ms
    assert data["cache"]["warm_ttft_p99"] == 20.0
    assert data["results"][1]["cached_tokens"] == 64
