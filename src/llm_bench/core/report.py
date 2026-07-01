from __future__ import annotations
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
from .stats import Stats
from .pricing import CostEstimate
from .adapters.base import RequestResult

console = Console()


def _fmt_ms(val: float) -> str:
    if val >= 1000:
        return f"{val / 1000:.2f}s"
    return f"{val:.0f}ms"


def _fmt_cost(cost: CostEstimate | None) -> str:
    if cost is None:
        return "n/a"
    return f"${cost.total_cost:.6f}"


def render(provider: str, model: str, stats: Stats, cost: CostEstimate | None, n: int, task: str = "text") -> None:
    table = Table(title=f"llm-bench  {provider}/{model}  (n={n}, task={task})")
    table.add_column("Metric", style="bold cyan")
    table.add_column("p50", justify="right")
    table.add_column("p95", justify="right")
    table.add_column("p99", justify="right")
    table.add_column("mean", justify="right")

    table.add_row("TTFT", _fmt_ms(stats.ttft_p50), _fmt_ms(stats.ttft_p95), _fmt_ms(stats.ttft_p99), _fmt_ms(stats.ttft_mean))
    table.add_row("Total", _fmt_ms(stats.total_p50), _fmt_ms(stats.total_p95), _fmt_ms(stats.total_p99), _fmt_ms(stats.total_mean))
    table.add_row("ITL", _fmt_ms(stats.itl_p50), _fmt_ms(stats.itl_p95), _fmt_ms(stats.itl_p99), _fmt_ms(stats.itl_mean))

    console.print(table)
    console.print(
        f"Throughput: [bold]{stats.throughput_tps:.1f}[/bold] tok/s  |  "
        f"Cost/call: [bold]{_fmt_cost(cost)}[/bold]  |  "
        f"Errors: {stats.error_count}/{stats.total_count}"
    )


def render_comparison(
    primary: tuple[str, str, Stats, CostEstimate | None],
    compare: tuple[str, str, Stats, CostEstimate | None],
    n: int,
    task: str = "text",
) -> None:
    p_provider, p_model, p_stats, p_cost = primary
    c_provider, c_model, c_stats, c_cost = compare

    table = Table(title=f"llm-bench comparison (n={n} each, task={task})")
    table.add_column("Provider", style="bold")
    table.add_column("Model")
    table.add_column("TTFT p50", justify="right")
    table.add_column("TTFT p99", justify="right")
    table.add_column("Total p99", justify="right")
    table.add_column("tok/s", justify="right")
    table.add_column("$/call", justify="right")

    table.add_row(p_provider, p_model, _fmt_ms(p_stats.ttft_p50), _fmt_ms(p_stats.ttft_p99), _fmt_ms(p_stats.total_p99), f"{p_stats.throughput_tps:.1f}", _fmt_cost(p_cost))
    table.add_row(c_provider, c_model, _fmt_ms(c_stats.ttft_p50), _fmt_ms(c_stats.ttft_p99), _fmt_ms(c_stats.total_p99), f"{c_stats.throughput_tps:.1f}", _fmt_cost(c_cost))

    d_ttft_p50 = p_stats.ttft_p50 - c_stats.ttft_p50
    d_ttft_p99 = p_stats.ttft_p99 - c_stats.ttft_p99
    d_total_p99 = p_stats.total_p99 - c_stats.total_p99
    faster = p_provider if d_ttft_p99 < 0 else c_provider
    table.add_row(
        f"delta ({faster} faster)", "",
        f"{d_ttft_p50:+.0f}ms", f"{d_ttft_p99:+.0f}ms", f"{d_total_p99:+.0f}ms",
        "", "", style="dim"
    )

    console.print(table)


def write_md(path: str | Path, provider: str, model: str, stats: Stats, cost: CostEstimate | None, n: int, task: str = "text") -> None:
    lines = [
        f"# llm-bench: {provider}/{model}",
        "",
        f"n={n} | task={task} | errors={stats.error_count}/{stats.total_count}",
        "",
        "| Metric | p50 | p95 | p99 | mean |",
        "|--------|-----|-----|-----|------|",
        f"| TTFT | {_fmt_ms(stats.ttft_p50)} | {_fmt_ms(stats.ttft_p95)} | {_fmt_ms(stats.ttft_p99)} | {_fmt_ms(stats.ttft_mean)} |",
        f"| Total | {_fmt_ms(stats.total_p50)} | {_fmt_ms(stats.total_p95)} | {_fmt_ms(stats.total_p99)} | {_fmt_ms(stats.total_mean)} |",
        f"| ITL | {_fmt_ms(stats.itl_p50)} | {_fmt_ms(stats.itl_p95)} | {_fmt_ms(stats.itl_p99)} | {_fmt_ms(stats.itl_mean)} |",
        "",
        f"Throughput: {stats.throughput_tps:.1f} tok/s | Cost/call: {_fmt_cost(cost)}",
    ]
    Path(path).write_text("\n".join(lines))


def write_json(path: str | Path, results: list[RequestResult], meta: dict | None = None) -> None:
    data = {
        "meta": meta or {},
        "results": [
            {
                "ttft_ns": r.ttft_ns,
                "total_ns": r.total_ns,
                "completion_tokens": r.completion_tokens,
                "prompt_tokens": r.prompt_tokens,
                "error": r.error,
            }
            for r in results
        ],
    }
    Path(path).write_text(json.dumps(data, indent=2))
