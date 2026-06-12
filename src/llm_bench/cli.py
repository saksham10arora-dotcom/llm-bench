from __future__ import annotations
import asyncio
import os
import sys
import click
from .core import runner as runner_mod, stats as stats_mod, pricing, report
from .core.adapters.anthropic import AnthropicAdapter
from .core.adapters.openai import OpenAIAdapter


def _build_adapter(provider: str, model: str, api_key: str | None, base_url: str | None):
    if provider == "anthropic":
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            click.echo("Error: ANTHROPIC_API_KEY not set. Use --api-key or set the env var.", err=True)
            sys.exit(1)
        return AnthropicAdapter(model=model, api_key=key)
    if provider == "openai":
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            click.echo("Error: OPENAI_API_KEY not set. Use --api-key or set the env var.", err=True)
            sys.exit(1)
        return OpenAIAdapter(model=model, api_key=key, base_url=base_url)
    click.echo(f"Error: unknown provider '{provider}'. Use 'anthropic' or 'openai'.", err=True)
    sys.exit(1)


def _run_and_stats(adapter, prompt, n, max_tokens, concurrency, warmup):
    results = asyncio.run(runner_mod.run(
        adapter, prompt, n=n, max_tokens=max_tokens, concurrency=concurrency, warmup=warmup
    ))
    computed = stats_mod.compute(results)
    successful = [r for r in results if r.success]
    avg_completion = (
        sum(r.completion_tokens for r in successful) // len(successful)
        if successful else 0
    )
    model_name = adapter.model if adapter is not None else ""
    cost = pricing.estimate(
        model_name,
        prompt_tokens=len(prompt.split()),
        completion_tokens=avg_completion,
    )
    errors = [r.error for r in results if not r.success]
    if errors:
        click.echo(f"Sample error: {errors[0]}", err=True)
    return results, computed, cost


@click.command()
@click.option("--provider", required=True, help="anthropic or openai")
@click.option("--model", required=True, help="Model ID")
@click.option("--prompt", "prompt_str", required=True, help="Prompt string or @file.txt to read from file")
@click.option("-n", default=50, show_default=True, help="Number of requests")
@click.option("--concurrency", default=1, show_default=True, help="Max concurrent requests")
@click.option("--max-tokens", default=256, show_default=True, help="Max completion tokens")
@click.option("--warmup", default=3, show_default=True, help="Warmup requests excluded from stats")
@click.option("--timeout", default=30, show_default=True, help="Per-request timeout (seconds)")
@click.option("--base-url", default=None, help="Override base URL (OpenAI-compatible endpoints)")
@click.option("--api-key", default=None, help="Override API key")
@click.option("--markdown", default=None, help="Write markdown report to path")
@click.option("--json", "json_path", default=None, help="Write JSON results to path")
@click.option("--compare", default=None, help="provider:model to compare against (e.g. openai:gpt-4o)")
def main(provider, model, prompt_str, n, concurrency, max_tokens, warmup, timeout, base_url, api_key, markdown, json_path, compare):
    if prompt_str.startswith("@"):
        prompt = open(prompt_str[1:]).read().strip()
    else:
        prompt = prompt_str

    if concurrency > n:
        click.echo(f"Warning: --concurrency {concurrency} > -n {n}, clamping.", err=True)
        concurrency = n

    adapter = _build_adapter(provider, model, api_key, base_url)
    results, computed, cost = _run_and_stats(adapter, prompt, n, max_tokens, concurrency, warmup)

    if compare:
        parts = compare.split(":", 1)
        if len(parts) != 2:
            click.echo("Error: --compare must be provider:model (e.g. openai:gpt-4o)", err=True)
            sys.exit(1)
        cmp_provider, cmp_model = parts
        cmp_adapter = _build_adapter(cmp_provider, cmp_model, api_key, base_url)
        _, cmp_stats, cmp_cost = _run_and_stats(cmp_adapter, prompt, n, max_tokens, concurrency, warmup)
        report.render_comparison(
            (provider, model, computed, cost),
            (cmp_provider, cmp_model, cmp_stats, cmp_cost),
            n,
        )
    else:
        report.render(provider, model, computed, cost, n)

    if markdown:
        report.write_md(markdown, provider, model, computed, cost, n)
        click.echo(f"Markdown report written to {markdown}")
    if json_path:
        report.write_json(json_path, results)
        click.echo(f"JSON results written to {json_path}")
