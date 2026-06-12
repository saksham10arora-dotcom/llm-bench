# llm-bench

LLM API latency benchmarker. Reports TTFT p50/p95/p99, total latency, inter-token latency, tokens/sec, and cost per call -- the same statistical rigor used in HFT latency analysis.

Most benchmarking tools report mean latency. Mean hides tail behavior. This one doesn't.

Built by [Saksham Arora](https://saksham10arora-dotcom.github.io).

## Install

Requires [uv](https://docs.astral.sh/uv/).

```bash
uvx --from git+https://github.com/saksham10arora-dotcom/llm-bench llm-bench
```

Or clone and run locally:

```bash
git clone https://github.com/saksham10arora-dotcom/llm-bench
cd llm-bench
uv sync
uv run llm-bench --help
```

## Try it without an API key

```bash
llm-bench --provider mock --model demo --prompt "hi" -n 20 --warmup 2
```

Runs the full pipeline against a simulated endpoint. No network, no keys.

## Usage

```bash
# Benchmark Anthropic
llm-bench --provider anthropic --model claude-sonnet-4-6 \
  --prompt "Explain recursion in one sentence" \
  -n 100 --warmup 5

# Benchmark any OpenAI-compatible endpoint (Groq, Together, vLLM, OpenRouter)
llm-bench --provider openai --model llama-3.3-70b-versatile \
  --base-url https://api.groq.com/openai/v1 \
  --prompt "Explain recursion" -n 100

# Compare two providers side-by-side
llm-bench --provider anthropic --model claude-sonnet-4-6 \
  --prompt "Explain recursion" -n 100 \
  --compare openai:gpt-4o

# Latency under load
llm-bench --provider anthropic --model claude-sonnet-4-6 \
  --prompt "Explain recursion" -n 100 --concurrency 8

# Export results
llm-bench --provider anthropic --model claude-sonnet-4-6 \
  --prompt "Explain recursion" -n 100 \
  --markdown results.md --json results.json
```

## Output

```
llm-bench  anthropic/claude-sonnet-4-6  (n=100)
+---------+-------+-------+--------+-------+
| Metric  |   p50 |   p95 |    p99 |  mean |
+---------+-------+-------+--------+-------+
| TTFT    | 320ms | 710ms |  890ms | 380ms |
| Total   |  1.2s |  1.8s |   2.1s |  1.3s |
| ITL     |  11ms |  24ms |   95ms |  14ms |
+---------+-------+-------+--------+-------+
Throughput: 87.3 tok/s  |  Cost/call: $0.000042  |  Errors: 0/100
```

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--provider` | required | `anthropic`, `openai`, or `mock` |
| `--model` | required | Model ID |
| `--prompt` | required | Prompt string (or `@file.txt`) |
| `-n` | 50 | Number of requests |
| `--concurrency` | 1 | Max in-flight requests |
| `--max-tokens` | 256 | Max completion tokens |
| `--warmup` | 3 | Warmup requests excluded from stats |
| `--timeout` | 30 | Per-request timeout in seconds |
| `--base-url` | None | OpenAI-compatible endpoint override |
| `--api-key` | None | API key override (defaults to env var) |
| `--markdown` | None | Write markdown report |
| `--json` | None | Write raw JSON results + run metadata |
| `--compare` | None | `provider:model` for side-by-side comparison |

## Auth

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...   # also used for Groq/Together/vLLM keys with --base-url
```

## Methodology

The numbers are only worth sharing if the measurement is honest. What this tool does, precisely:

- **Clock**: all timestamps come from `time.monotonic_ns()`. Wall clocks can step backwards; monotonic clocks can't.
- **TTFT**: time from request dispatch to the first content chunk. Includes connection setup, queueing, and prefill -- because that's what your user actually waits for.
- **Total**: time from dispatch to the *last content chunk*. Trailing usage frames and connection teardown are excluded.
- **ITL**: every gap between consecutive content chunks is measured individually and pooled across all requests. Percentiles are computed over raw gaps, never over per-request averages (averaging first hides exactly the tail spikes you're looking for). Note: providers stream chunks, not tokens -- a chunk may carry several tokens.
- **Throughput**: decode rate -- `(completion_tokens - 1) / (total - TTFT)`. Including TTFT would blend prefill into a number that claims to be generation speed.
- **Warmup**: warmup requests fully complete before measurement begins, even under `--concurrency`. They never overlap the measured window.
- **Errors**: failed and timed-out requests are counted and reported, but excluded from latency percentiles. A failed call has no valid latency.
- **Percentiles**: nearest-rank method on the raw samples. No interpolation, no smoothing.
- **Token counts**: exact counts from API usage data when the endpoint provides them; chunk-count fallback otherwise. Cost is computed from exact counts, not estimates.
- **Sample size**: with n=50, "p99" is just your second-slowest request. The tool warns below n=100. For stable tails, use n=200+.
- **Compare mode caveat**: the two targets run sequentially, not interleaved, so a network blip during one run biases that target. Treat small deltas (<20%) as noise.

## Why percentiles?

Average latency is the rookie metric. If 95% of requests complete in 100ms but 5% take 5s, your average looks fine while your users are suffering. p99 is what your worst case looks like. Knowing the distribution -- TTFT separately from total -- is how you debug actual perceived latency problems.
