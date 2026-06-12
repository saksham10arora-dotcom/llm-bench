# llm-bench

LLM API latency benchmarker. Reports TTFT p50/p95/p99, total latency, tokens/sec, and cost per call -- the same statistical rigor used in HFT latency analysis.

Most benchmarking tools report mean latency. Mean hides tail behavior. This one doesn't.

Built by [Saksham Arora](https://saksham10arora-dotcom.github.io).

## Install

Requires [uv](https://docs.astral.sh/uv/).

```bash
uvx --from git+https://github.com/saksham10arora/llm-bench llm-bench
```

Or clone and run locally:

```bash
git clone https://github.com/saksham10arora/llm-bench
cd llm-bench
uv sync
uv run llm-bench --help
```

## Usage

```bash
# Benchmark Anthropic
llm-bench --provider anthropic --model claude-sonnet-4-6 \
  --prompt "Explain recursion in one sentence" \
  -n 100 --warmup 5

# Benchmark OpenAI-compatible endpoint (Groq, Together, vLLM)
llm-bench --provider openai --model llama3-70b-8192 \
  --base-url https://api.groq.com/openai/v1 \
  --prompt "Explain recursion" -n 50

# Compare two providers side-by-side
llm-bench --provider anthropic --model claude-sonnet-4-6 \
  --prompt "Explain recursion" -n 50 \
  --compare openai:gpt-4o

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
| ITL     |     - |     - |   18ms |  12ms |
+---------+-------+-------+--------+-------+
Throughput: 87.3 tok/s  |  Cost/call: $0.000042  |  Errors: 0/100
```

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--provider` | required | `anthropic` or `openai` |
| `--model` | required | Model ID |
| `--prompt` | required | Prompt string (or `@file.txt`) |
| `-n` | 50 | Number of requests |
| `--concurrency` | 1 | Max in-flight requests |
| `--max-tokens` | 256 | Max completion tokens |
| `--warmup` | 3 | Warmup requests excluded from stats |
| `--base-url` | None | OpenAI-compatible endpoint override |
| `--api-key` | None | API key override (defaults to env var) |
| `--markdown` | None | Write markdown report |
| `--json` | None | Write raw JSON results |
| `--compare` | None | `provider:model` for side-by-side comparison |

## Auth

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
```

## Why percentiles?

Average latency is the rookie metric. If 95% of requests complete in 100ms but 5% take 5s, your average looks fine while your users are suffering. p99 is what your worst case looks like. Knowing the distribution -- TTFT separately from total -- is how you debug actual perceived latency problems.
