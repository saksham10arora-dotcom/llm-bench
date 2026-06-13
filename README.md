# llm-bench

[![CI](https://github.com/saksham10arora-dotcom/llm-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/saksham10arora-dotcom/llm-bench/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-28%20passing-brightgreen)

**LLM API latency benchmarker with the statistical rigor of HFT latency analysis.** Measures TTFT, total latency, inter-token latency, throughput, and cost per call, and reports p50/p95/p99 instead of a single misleading average.

Most benchmarking tools report mean latency. In trading systems, reporting the mean is the mistake that gets you laughed out of the room: latency lives in the tail, and p99 is what your worst user actually feels. This tool treats LLM APIs the same way.

Works with Anthropic, OpenAI, and any OpenAI-compatible endpoint (Groq, Together, vLLM, OpenRouter). Ships with a mock provider so you can try the whole pipeline with no API key.

Built by [Saksham Arora](https://saksham10arora-dotcom.github.io).

---

## Quickstart

Requires [uv](https://docs.astral.sh/uv/).

```bash
# Run instantly, no clone, no key (mock provider)
uvx --from git+https://github.com/saksham10arora-dotcom/llm-bench llm-bench \
  --provider mock --model demo --prompt "hi" -n 20

# Or clone and develop locally
git clone https://github.com/saksham10arora-dotcom/llm-bench
cd llm-bench && uv sync && uv run llm-bench --help
```

---

## The demo that sold me on building it

A real run, 100 requests against Groq (llama-3.3-70b-versatile, free tier):

```
llm-bench  openai/llama-3.3-70b-versatile
                 (n=100)
+--------+-------+-------+-------+-------+
| Metric |   p50 |   p95 |   p99 |  mean |
+--------+-------+-------+-------+-------+
| TTFT   | 2.20s | 2.32s | 2.33s | 1.71s |
| Total  | 2.28s | 2.39s | 2.41s | 1.79s |
| ITL    |   0ms |  10ms |  13ms |   2ms |
+--------+-------+-------+-------+-------+
Throughput: 536.1 tok/s  |  Cost/call: $0.000058  |  Errors: 0/100
```

Look at TTFT. The mean (1.71s) is **lower** than the median (2.20s). That should be impossible for a normal latency distribution, and it is: this one is bimodal. The first ~30 requests returned in ~200ms, then the free tier's rate limiter kicked in and everything after waited ~2.2s. The mean blended two completely different latency regimes into one number that describes neither. A mean-only tool would have printed "1.7s, fine" and hidden the whole story. Catching exactly this is why the tool exists.

---

## What it measures

| Metric | What it is | Why it is computed this way |
|--------|-----------|-----------------------------|
| **TTFT** | Dispatch to first content chunk | What the user perceives as "it's responding." Queueing + prefill included on purpose. |
| **Total** | Dispatch to last content chunk | Trailing usage frames and connection teardown excluded. |
| **ITL** | Gap between consecutive chunks | Pooled across all requests, percentiles on raw gaps, so a single stall survives to p99. |
| **Throughput** | `(tokens - 1) / (total - TTFT)` | Pure decode rate. Including prefill would mislabel two phases as one. |
| **Cost/call** | prompt + completion tokens x price | From exact API usage counts when available, not word-count estimates. |
| **Errors** | failed / total | Counted and reported, never averaged into the latency percentiles. |

Every metric is reported at p50, p95, p99, and mean.

---

## Usage

```bash
# Benchmark Anthropic
llm-bench --provider anthropic --model claude-sonnet-4-6 \
  --prompt "Explain recursion in one sentence" -n 100 --warmup 5

# Any OpenAI-compatible endpoint (Groq, Together, vLLM, OpenRouter)
llm-bench --provider openai --model llama-3.3-70b-versatile \
  --base-url https://api.groq.com/openai/v1 \
  --prompt "Explain recursion" -n 100

# Compare two providers side-by-side (prints a delta row)
llm-bench --provider anthropic --model claude-sonnet-4-6 \
  --prompt "Explain recursion" -n 100 \
  --compare openai:gpt-4o

# Latency under load
llm-bench --provider anthropic --model claude-sonnet-4-6 \
  --prompt "Explain recursion" -n 100 --concurrency 8

# Export for further analysis
llm-bench --provider anthropic --model claude-sonnet-4-6 \
  --prompt "Explain recursion" -n 100 \
  --markdown results.md --json results.json
```

Prompts can be inlined or read from a file with `--prompt @prompt.txt`.

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--provider` | required | `anthropic`, `openai`, or `mock` |
| `--model` | required | Model ID |
| `--prompt` | required | Prompt string, or `@file.txt` to read from a file |
| `-n` | 50 | Number of measured requests |
| `--concurrency` | 1 | Max in-flight requests |
| `--max-tokens` | 256 | Max completion tokens |
| `--warmup` | 3 | Warmup requests, excluded from stats |
| `--timeout` | 30 | Per-request timeout in seconds |
| `--base-url` | None | OpenAI-compatible endpoint override |
| `--api-key` | None | API key override (defaults to env var) |
| `--markdown` | None | Write a markdown report |
| `--json` | None | Write raw per-request JSON + run metadata |
| `--compare` | None | `provider:model` for a side-by-side comparison |

### Auth

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...   # also used for Groq / Together / vLLM keys with --base-url
```

---

## Supported providers

| Provider | How | Token counts | Cost |
|----------|-----|--------------|------|
| Anthropic | native streaming SDK | exact (`usage`) | priced |
| OpenAI | streaming SDK | exact (`usage`) | priced |
| Groq / Together / vLLM / OpenRouter | OpenAI-compatible, via `--base-url` | exact, or chunk-count fallback | priced when model is known |
| Mock | scripted delays, no network | synthetic | n/a |

---

## Architecture

The core is fully decoupled from the CLI, so the runner and stats engine can be reused (a planned MCP wrapper will import them directly without touching `cli.py`).

```
src/llm_bench/
  cli.py              # Click CLI: arg parsing + wiring only, no business logic
  core/
    adapters/
      base.py         # StreamEvent / RequestResult / Adapter protocol
      anthropic.py    # native Anthropic streaming
      openai.py       # OpenAI + base_url override (Groq / vLLM / Together)
      mock.py         # scripted-delay fake adapter, zero network
    runner.py         # async runner: semaphore concurrency, warmup, timeout
    stats.py          # nearest-rank percentiles, pooled ITL, decode-rate throughput
    pricing.py        # per-model price table, cost estimate
    report.py         # rich table + markdown / JSON export
```

Data flows one direction: `cli -> runner -> adapter.stream() -> stats.compute() -> report`. Adapters only ever yield timestamped stream events; all the math lives in `stats.py`, which has zero knowledge of HTTP.

---

## Methodology

The numbers are only worth sharing if the measurement is honest. Precisely what this tool does:

- **Clock**: all timestamps come from `time.monotonic_ns()`. Wall clocks can step backwards; monotonic clocks can't.
- **TTFT**: time from request dispatch to the first content chunk. Includes connection setup, queueing, and prefill, because that is what the user waits through.
- **Total**: dispatch to the last content chunk. Trailing usage frames and connection teardown are excluded.
- **ITL**: every gap between consecutive content chunks is measured individually and pooled across all requests. Percentiles are over raw gaps, never per-request averages, which would hide exactly the tail spikes you are hunting. Note: providers stream chunks, not tokens, so a chunk may carry several tokens.
- **Throughput**: decode rate, `(completion_tokens - 1) / (total - TTFT)`. Including TTFT would blend prefill into a number that claims to be generation speed.
- **Warmup**: warmup requests fully complete before measurement begins, even under `--concurrency`. They never overlap the measured window.
- **Errors**: failed and timed-out requests are counted and reported, but excluded from latency percentiles. A failed call has no valid latency.
- **Percentiles**: nearest-rank on the raw samples. No interpolation, no smoothing.
- **Token counts**: exact counts from API usage when the endpoint provides them, chunk-count fallback otherwise. Cost is computed from exact counts, not estimates.
- **Sample size**: with n=50, "p99" is just your second-slowest request. The tool warns below n=100. For stable tails, use n=200+.
- **Compare-mode caveat**: the two targets run sequentially, not interleaved, so a network blip during one run biases that target. Treat small deltas (<20%) as noise.

---

## Development

```bash
uv sync --group dev
uv run pytest -v          # 28 tests
```

The test suite makes **no network calls**. A scripted mock adapter drives the runner with controllable delays, so percentile math, concurrency, warmup ordering, and timeout handling are all verified deterministically. The credibility-critical code (percentiles, pooled ITL) is tested against known distributions, including a case where two tail spikes hide among 98 fast samples and must surface at p99. CI runs the full suite on Python 3.11 and 3.12 plus a keyless end-to-end run on every push.

---

## Roadmap

- [ ] Publish to PyPI (`pip install llm-bench`)
- [ ] MCP server wrapper (expose `benchmark` as a tool to Claude / MCP clients)
- [ ] Interleaved compare mode (alternate requests to cancel out network drift)
- [ ] Histogram / sparkline output for distribution shape at a glance

---

## License

MIT. See [LICENSE](LICENSE).
