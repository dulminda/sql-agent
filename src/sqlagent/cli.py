"""Command-line interface: ``sqlagent --help``."""

from __future__ import annotations

import json
from pathlib import Path

import typer
import yaml

from sqlagent.agent import AgentConfig

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
eval_app = typer.Typer(no_args_is_help=True, help="Evaluate the agent and its guardrails.")
app.add_typer(eval_app, name="eval")

CHINOOK = Path("data/chinook.sqlite")
BENCH = Path("benchmarks/chinook.jsonl")
SECURITY = Path("benchmarks/security.jsonl")


def _config(path: Path | None) -> AgentConfig:
    if path is None:
        return AgentConfig()
    data = yaml.safe_load(path.read_text()) or {}
    data.setdefault("name", path.stem)
    return AgentConfig.model_validate(data)


def _model(cfg: AgentConfig):
    from sqlagent.llm import ClaudeChat

    return ClaudeChat(model=cfg.model, effort=cfg.effort)


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


@app.command()
def data(dest: Path = typer.Option(CHINOOK, help="Where to save the Chinook database.")) -> None:
    """Download the Chinook sample database."""
    from sqlagent.eval.datasets import download_chinook

    typer.echo(f"Chinook ready at {download_chinook(dest)}")


@app.command()
def ask(
    question: str,
    db: Path = typer.Option(CHINOOK, help="SQLite database file."),
    config: Path | None = typer.Option(None, help="Agent YAML config."),
    evidence: str = typer.Option("", help="Optional domain hint, as in BIRD."),
    trace: bool = typer.Option(False, help="Print every tool call."),
) -> None:
    """Answer a question about a SQLite database (needs ANTHROPIC_API_KEY)."""
    from sqlagent.agent import SQLAgent
    from sqlagent.db import Database

    cfg = _config(config)
    res = SQLAgent(_model(cfg), Database(db), cfg).run(question, evidence)
    if trace:
        for s in res.steps:
            mark = "✓" if s.ok else "✗"
            typer.echo(f"[turn {s.turn}] {mark} {s.tool} {json.dumps(s.input)[:160]}")
            typer.echo(f"    {s.summary[:200]}")
    typer.echo(f"\nstatus: {res.status}   cost: ${res.cost_usd:.4f}   "
               f"turns: {res.turns}   tool calls: {res.tool_calls}")
    if res.sql:
        typer.echo(f"\nSQL:\n{res.sql}\n")
        typer.echo(" | ".join(res.columns))
        for row in res.rows[:20]:
            typer.echo(" | ".join("NULL" if v is None else str(v) for v in row))
        if len(res.rows) > 20:
            typer.echo(f"... {len(res.rows) - 20} more rows")
    elif res.error:
        typer.echo(f"error: {res.error}")


@app.command()
def serve(db: Path = typer.Option(CHINOOK), config: Path | None = None,
          host: str = "127.0.0.1", port: int = 8000) -> None:
    """Serve the HTTP API."""
    import os

    import uvicorn

    os.environ["SQLAGENT_DB"] = str(db)
    if config:
        os.environ["SQLAGENT_CONFIG"] = str(config)
    uvicorn.run("sqlagent.api.app:create_app", factory=True, host=host, port=port)


@eval_app.command("run")
def eval_run(
    config: list[Path] = typer.Option(..., help="One or more agent YAMLs (first = baseline)."),
    benchmark: Path = typer.Option(BENCH),
    fmt: str = typer.Option("jsonl", "--format", help="jsonl | bird | spider"),
    db_root: Path = typer.Option(Path("data"), help="Directory holding the databases."),
    limit: int | None = typer.Option(None, help="Only the first N questions (cost cap)."),
    workers: int = 4,
    out: Path = Path("reports/agent.json"),
    report: Path | None = None,
) -> None:
    """End-to-end execution accuracy, failure taxonomy and $/task (needs ANTHROPIC_API_KEY)."""
    from sqlagent.eval.datasets import load_examples
    from sqlagent.eval.report import agent_table
    from sqlagent.eval.runner import DatabasePool, run_agent_eval

    examples = load_examples(fmt, benchmark)[:limit] if limit else load_examples(fmt, benchmark)
    pool = DatabasePool(db_root)
    runs = []
    for path in config:
        cfg = _config(path)
        typer.echo(f"→ {cfg.name} on {len(examples)} questions")

        def progress(done: int, spent: float) -> None:
            if done % 10 == 0 or done == len(examples):
                typer.echo(f"  {done}/{len(examples)}  ${spent:.3f}")

        runs.append(run_agent_eval(cfg, examples, pool, lambda c=cfg: _model(c), workers,
                                   progress))
    _write(out, runs)
    table = agent_table(runs)
    if report:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(f"## Agent: {benchmark.name} ({len(examples)} questions)\n\n{table}\n")
    typer.echo(table)


@eval_app.command("security")
def eval_security(
    db: Path = typer.Option(CHINOOK),
    corpus: Path = typer.Option(SECURITY),
    benchmark: Path = typer.Option(BENCH, help="Gold SQL from here must never be blocked."),
    out: Path = Path("reports/security.json"),
    report: Path | None = None,
) -> None:
    """Attack corpus against both sandbox layers (offline, no LLM)."""
    from sqlagent.eval.datasets import load_jsonl
    from sqlagent.eval.report import security_table
    from sqlagent.eval.security import run_security_eval

    s = run_security_eval(db, corpus, [e.gold_sql for e in load_jsonl(benchmark)])
    _write(out, s)
    table = security_table(s)
    if report:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(f"## Sandbox security\n\n{table}\n")
    typer.echo(table)
    for leak in s["leaks"]:
        typer.secho(f"LEAK {leak['id']}: {leak['sql']}", fg=typer.colors.RED)


@eval_app.command("linking")
def eval_linking(
    benchmark: Path = typer.Option(BENCH),
    fmt: str = typer.Option("jsonl", "--format"),
    db_root: Path = typer.Option(Path("data")),
    out: Path = Path("reports/linking.json"),
    report: Path | None = None,
) -> None:
    """Gold-table recall of schema linking at several k (offline, no LLM)."""
    from sqlagent.eval.datasets import load_examples
    from sqlagent.eval.linking import run_linking_eval
    from sqlagent.eval.report import linking_table
    from sqlagent.eval.runner import DatabasePool

    examples = load_examples(fmt, benchmark)
    pool = DatabasePool(db_root)
    rows = run_linking_eval(examples, {d: pool.get(d) for d in {e.db_id for e in examples}})
    _write(out, rows)
    table = linking_table(rows)
    if report:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(f"## Schema linking ({len(examples)} questions)\n\n{table}\n")
    typer.echo(table)


@eval_app.command("redteam")
def eval_redteam(
    config: Path = typer.Option(..., help="Agent YAML."),
    db: Path = typer.Option(CHINOOK),
    workdir: Path = Path("data/redteam"),
    out: Path = Path("reports/redteam.json"),
) -> None:
    """Indirect prompt injection: poisoned data, live agent (needs ANTHROPIC_API_KEY)."""
    import hashlib

    from sqlagent.eval.datasets import load_jsonl
    from sqlagent.eval.redteam import build_poisoned_db
    from sqlagent.eval.runner import DatabasePool, run_agent_eval

    db_file, bench = build_poisoned_db(db, workdir)
    before = hashlib.sha256(db_file.read_bytes()).hexdigest()
    cfg = _config(config)
    run = run_agent_eval(cfg, load_jsonl(bench), DatabasePool(workdir), lambda: _model(cfg))
    run["write_attempts"] = sum(r["guard_blocks"] + r["sandbox_blocks"] for r in run["rows"])
    run["tasks_with_write_attempts"] = sum(
        (r["guard_blocks"] + r["sandbox_blocks"]) > 0 for r in run["rows"])
    run["database_unchanged"] = hashlib.sha256(db_file.read_bytes()).hexdigest() == before
    _write(out, run)
    typer.echo(f"accuracy {run['execution_accuracy']['mean']:.3f}  "
               f"write attempts {run['write_attempts']} "
               f"({run['tasks_with_write_attempts']}/{run['n']} tasks)  "
               f"database unchanged: {run['database_unchanged']}")


@eval_app.command("show")
def eval_show(results: Path, example_id: str = typer.Option(..., "--id")) -> None:
    """Print the step-by-step trace of one example from an `eval run` result file."""
    for run in json.loads(results.read_text()):
        for r in run["rows"]:
            if r["id"] != example_id:
                continue
            typer.secho(f"== {run['name']}: {r['category']} (${r['cost_usd']:.4f})", bold=True)
            typer.echo(f"Q: {r['question']}\ngold: {r['gold_sql']}\npred: {r['pred_sql']}")
            for s in r["steps"]:
                typer.echo(f"  [turn {s['turn']}] {'✓' if s['ok'] else '✗'} {s['tool']} "
                           f"{json.dumps(s['input'])[:140]}\n      {s['summary'][:160]}")


@eval_app.command("gate")
def eval_gate(
    result: list[str] = typer.Option(..., help="name=path pairs, e.g. security=reports/s.json"),
    thresholds: Path = typer.Option(Path("eval/thresholds.yaml")),
) -> None:
    """Fail (exit 1) if any metric violates its threshold. Used in CI."""
    from sqlagent.gate import check

    results = {}
    for item in result:
        name, _, path = item.partition("=")
        results[name] = json.loads(Path(path).read_text())
    failures = check(results, yaml.safe_load(thresholds.read_text()))
    for f in failures:
        typer.secho(f"FAIL {f}", fg=typer.colors.RED)
    if failures:
        raise typer.Exit(1)
    typer.secho("Quality gate passed.", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
