"""Offline security evaluation of the two sandbox layers. Needs no LLM.

Each attack is checked three ways:

1. guard: is it rejected by the static policy check?
2. sandbox: run it straight against the read-only connection with the guard *bypassed*.
   Does SQLite refuse it?
3. integrity: is the database file byte-identical afterwards?

Benign queries (tricky but harmless, such as string literals containing ``DROP TABLE``)
must pass both layers, so the false-block rate is measured too. All attacks run against a
temporary copy of the database.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from sqlagent.db import Database, QueryError
from sqlagent.guard import check_sql


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_security_eval(db_file: str | Path, corpus: str | Path,
                      extra_benign: list[str] | None = None) -> dict[str, Any]:
    cases = [json.loads(line) for line in Path(corpus).read_text().splitlines() if line.strip()]
    for i, sql in enumerate(extra_benign or []):
        cases.append({"id": f"gold-{i:03d}", "kind": "benign", "category": "gold_sql", "sql": sql})

    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "db.sqlite"
        shutil.copy(db_file, copy)
        before = _sha256(copy)
        db = Database(copy, timeout_s=2.0)
        rows = []
        for c in cases:
            guard_ok = check_sql(c["sql"]).ok
            try:
                db.execute(c["sql"], max_rows=5)
                sandbox_ok, sandbox_error = True, ""
            except QueryError as e:
                sandbox_ok, sandbox_error = False, str(e)
            rows.append({**c, "guard_allowed": guard_ok, "sandbox_allowed": sandbox_ok,
                         "sandbox_error": sandbox_error[:120]})
        unchanged = _sha256(copy) == before

    attacks = [r for r in rows if r["kind"] == "attack"]
    benign = [r for r in rows if r["kind"] == "benign"]
    by_cat: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "guard": 0, "sandbox": 0,
                                                             "both": 0})
    for r in attacks:
        s = by_cat[r["category"]]
        s["n"] += 1
        s["guard"] += not r["guard_allowed"]
        s["sandbox"] += not r["sandbox_allowed"]
        s["both"] += not (r["guard_allowed"] and r["sandbox_allowed"])

    def rate(xs: list[dict], key: str, blocked: bool = True) -> float:
        return sum((not r[key]) if blocked else r[key] for r in xs) / max(len(xs), 1)

    return {
        "n_attacks": len(attacks),
        "n_benign": len(benign),
        "attack_block_rate": {
            "guard": rate(attacks, "guard_allowed"),
            "sandbox": rate(attacks, "sandbox_allowed"),
            "combined": sum(not (r["guard_allowed"] and r["sandbox_allowed"]) for r in attacks)
            / max(len(attacks), 1),
        },
        "benign_false_block_rate": {
            "guard": rate(benign, "guard_allowed"),
            "sandbox": rate(benign, "sandbox_allowed"),
        },
        "database_unchanged": unchanged,
        "by_category": dict(by_cat),
        "leaks": [r for r in attacks if r["guard_allowed"] and r["sandbox_allowed"]],
        "false_blocks": [r for r in benign if not (r["guard_allowed"] and r["sandbox_allowed"])],
        "rows": rows,
    }
