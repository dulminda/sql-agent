"""End-to-end agent evaluation: execution accuracy, failure taxonomy, cost and tool use."""

from __future__ import annotations

import threading
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from sqlagent.agent import AgentConfig, SQLAgent
from sqlagent.db import Database
from sqlagent.eval.datasets import Example, db_path
from sqlagent.eval.execution import FAILURE_CATEGORIES, categorize
from sqlagent.eval.stats import bootstrap_ci
from sqlagent.llm import ChatModel


class DatabasePool:
    """One Database (and cached schema) per db_id, shared across worker threads."""

    def __init__(self, db_root: str | Path, max_rows: int = 10_000) -> None:
        self.db_root, self.max_rows = Path(db_root), max_rows
        self._dbs: dict[str, Database] = {}
        self._lock = threading.Lock()

    def get(self, db_id: str) -> Database:
        with self._lock:
            if db_id not in self._dbs:
                db = Database(db_path(self.db_root, db_id), max_rows=self.max_rows)
                _ = db.schema  # introspect once, under the lock
                self._dbs[db_id] = db
            return self._dbs[db_id]


def _summary(values: list[float]) -> dict[str, float]:
    lo, hi = bootstrap_ci(values)
    return {"mean": float(np.mean(values)) if values else float("nan"), "ci_low": lo,
            "ci_high": hi}


def run_agent_eval(
    config: AgentConfig,
    examples: list[Example],
    pool: DatabasePool,
    model_factory: Callable[[], ChatModel],
    workers: int = 8,
    on_progress: Callable[[int, float], None] | None = None,
) -> dict[str, Any]:
    done, spent = 0, 0.0
    lock = threading.Lock()

    def run_one(ex: Example) -> dict[str, Any]:
        nonlocal done, spent
        db = pool.get(ex.db_id)
        gold = db.execute(ex.gold_sql)
        res = SQLAgent(model_factory(), db, config).run(ex.question, ex.evidence)
        category = categorize(res.status, res.rows if res.sql else None, gold.rows)
        with lock:
            done += 1
            spent += res.cost_usd
            if on_progress:
                on_progress(done, spent)
        return {
            "id": ex.id, "db_id": ex.db_id, "difficulty": ex.difficulty, "question": ex.question,
            "gold_sql": ex.gold_sql, "pred_sql": res.sql, "status": res.status,
            "category": category, "correct": float(category == "correct"),
            "turns": res.turns, "tool_calls": res.tool_calls, "sql_attempts": res.sql_attempts,
            "guard_blocks": res.guard_blocks, "sandbox_blocks": res.sandbox_blocks,
            "cost_usd": res.cost_usd, "latency_ms": res.latency_ms,
            "input_tokens": res.input_tokens, "output_tokens": res.output_tokens,
            "error": res.error, "steps": [s.__dict__ for s in res.steps],
        }

    with ThreadPoolExecutor(max_workers=workers) as ex_pool:
        rows = list(ex_pool.map(run_one, examples))

    correct = [r["correct"] for r in rows]
    by_difficulty = {
        d: _summary([r["correct"] for r in rows if r["difficulty"] == d])
        for d in sorted({r["difficulty"] for r in rows})
    }
    categories = Counter(r["category"] for r in rows)
    lat = np.asarray([r["latency_ms"] for r in rows]) if rows else np.zeros(1)
    return {
        "name": config.name,
        "config": config.model_dump(),
        "n": len(rows),
        "execution_accuracy": _summary(correct),
        "by_difficulty": by_difficulty,
        "failures": {c: categories.get(c, 0) for c in FAILURE_CATEGORIES},
        "cost_per_task_usd": float(np.mean([r["cost_usd"] for r in rows])) if rows else 0.0,
        "total_cost_usd": float(sum(r["cost_usd"] for r in rows)),
        "mean_tool_calls": float(np.mean([r["tool_calls"] for r in rows])) if rows else 0.0,
        "mean_turns": float(np.mean([r["turns"] for r in rows])) if rows else 0.0,
        "latency_ms": {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95))},
        "ids": [r["id"] for r in rows],
        "per_example": correct,
        "rows": rows,
    }
