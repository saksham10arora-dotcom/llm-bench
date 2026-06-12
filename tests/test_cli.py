from click.testing import CliRunner
from unittest.mock import patch, AsyncMock
from llm_bench.cli import main
from llm_bench.core.adapters.base import RequestResult


def make_results(n: int = 10):
    return [
        RequestResult(ttft_ns=int(100e6), total_ns=int(500e6), completion_tokens=20)
        for _ in range(n)
    ]


def test_cli_runs_with_mock_adapter():
    runner = CliRunner()
    with patch("llm_bench.cli._build_adapter") as mock_build, \
         patch("llm_bench.cli.runner_mod.run", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = make_results(10)
        mock_build.return_value = None
        result = runner.invoke(main, [
            "--provider", "anthropic",
            "--model", "claude-sonnet-4-6",
            "--prompt", "hi",
            "-n", "10",
        ])
    assert result.exit_code == 0, result.output


def test_cli_missing_provider_fails():
    runner = CliRunner()
    result = runner.invoke(main, ["--model", "foo", "--prompt", "hi"])
    assert result.exit_code != 0


def test_cli_compare_parses_provider_model():
    runner = CliRunner()
    with patch("llm_bench.cli._build_adapter") as mock_build, \
         patch("llm_bench.cli.runner_mod.run", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = make_results(5)
        mock_build.return_value = None
        result = runner.invoke(main, [
            "--provider", "anthropic", "--model", "claude-sonnet-4-6",
            "--prompt", "hi", "-n", "5",
            "--compare", "openai:gpt-4o",
        ])
    assert result.exit_code == 0, result.output


def test_cli_mock_provider_runs_end_to_end_without_keys():
    # Full real pipeline: MockAdapter -> runner -> stats -> report. No patching, no network.
    runner = CliRunner()
    result = runner.invoke(main, [
        "--provider", "mock", "--model", "demo",
        "--prompt", "hi", "-n", "2", "--warmup", "0",
    ])
    assert result.exit_code == 0, result.output
    assert "TTFT" in result.output


def test_cli_missing_prompt_file_fails_cleanly():
    runner = CliRunner()
    result = runner.invoke(main, [
        "--provider", "mock", "--model", "demo",
        "--prompt", "@/nonexistent/prompt.txt", "-n", "2",
    ])
    assert result.exit_code == 1
    assert "cannot read prompt file" in result.output
