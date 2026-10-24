"""Layer 2 of the sandbox: a SQLite connection that cannot write, whatever SQL it is given.

Four independent controls, each enough on its own to stop a write:

1. the file is opened read-only (``mode=ro`` URI);
2. ``PRAGMA query_only = ON``;
3. an authorizer callback that denies every action except reads, at statement-prepare time;
4. a progress handler that aborts queries past a wall-clock deadline (stops runaway joins),
   plus a cap on rows fetched.
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlagent.guard import DENIED_FUNCTIONS

_ALLOWED_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}
_ALLOWED_ACTIONS.add(getattr(sqlite3, "SQLITE_RECURSIVE", 33))


def _authorizer(action: int, arg1: Any, arg2: Any, db_name: Any, trigger: Any) -> int:
    if action not in _ALLOWED_ACTIONS:
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION and str(arg2).lower() in DENIED_FUNCTIONS:
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


class QueryError(RuntimeError):
    pass


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[tuple]
    truncated: bool
    elapsed_ms: float


@dataclass
class Column:
    name: str
    type: str
    primary_key: bool
    samples: list[str] = field(default_factory=list)


@dataclass
class Table:
    name: str
    columns: list[Column]
    row_count: int
    foreign_keys: list[tuple[str, str, str]]  # (column, ref_table, ref_column)

    def render(self) -> str:
        """Compact DDL-like description: what the model sees for this table."""
        lines = [f"TABLE {self.name}  -- {self.row_count} rows"]
        for c in self.columns:
            note = " PRIMARY KEY" if c.primary_key else ""
            fk = next((f for f in self.foreign_keys if f[0] == c.name), None)
            if fk:
                note += f" REFERENCES {fk[1]}({fk[2]})"
            ex = f"  -- e.g. {', '.join(c.samples)}" if c.samples else ""
            lines.append(f"  {c.name} {c.type or 'ANY'}{note}{ex}")
        return "\n".join(lines)


def _sample(value: Any, max_chars: int = 30) -> str:
    if isinstance(value, str) and len(value) > max_chars:
        value = value[:max_chars] + "…"
    return repr(value)


def _quote(ident: str) -> str:
    return '"' + ident.replace('"', '""') + '"'


class Database:
    def __init__(
        self,
        path: str | Path,
        timeout_s: float = 5.0,
        max_rows: int = 1000,
        samples_per_column: int = 3,
    ) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.timeout_s, self.max_rows = timeout_s, max_rows
        self.samples_per_column = samples_per_column
        self._schema: dict[str, Table] | None = None

    def _connect(self, sandboxed: bool) -> sqlite3.Connection:
        conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, check_same_thread=False)
        conn.execute("PRAGMA query_only = ON")
        if sandboxed:
            conn.set_authorizer(_authorizer)
        return conn

    def execute(self, sql: str, max_rows: int | None = None) -> QueryResult:
        """Run untrusted SQL in the sandbox. A fresh connection per call keeps it thread-safe."""
        limit = max_rows or self.max_rows
        conn = self._connect(sandboxed=True)
        deadline = time.monotonic() + self.timeout_s
        conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
        start = time.perf_counter()
        try:
            cur = conn.execute(sql)
            rows = cur.fetchmany(limit + 1)
            columns = [d[0] for d in (cur.description or [])]
        except sqlite3.DatabaseError as e:
            msg = str(e)
            if msg == "interrupted":
                msg = f"query exceeded the {self.timeout_s:.0f}s time limit"
            elif "not authorized" in msg or "prohibited" in msg:
                msg = f"blocked by sandbox: {msg}"
            raise QueryError(msg) from e
        finally:
            conn.close()
        elapsed = (time.perf_counter() - start) * 1000
        return QueryResult(columns, rows[:limit], len(rows) > limit, elapsed)

    @property
    def schema(self) -> dict[str, Table]:
        if self._schema is None:
            self._schema = self._introspect()
        return self._schema

    def _introspect(self) -> dict[str, Table]:
        conn = self._connect(sandboxed=False)
        try:
            names = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )]
            tables = {}
            for name in names:
                q = _quote(name)
                cols = []
                for _, cname, ctype, _, _, pk in conn.execute(f"PRAGMA table_info({q})"):
                    samples: list[str] = []
                    if self.samples_per_column and not pk and not re.search(r"id$", cname, re.I):
                        samples = [
                            _sample(v) for (v,) in conn.execute(
                                f"SELECT DISTINCT {_quote(cname)} FROM {q} "
                                f"WHERE {_quote(cname)} IS NOT NULL LIMIT ?",
                                (self.samples_per_column,),
                            )
                        ]
                    cols.append(Column(cname, ctype, bool(pk), samples))
                fks = [(r[3], r[2], r[4]) for r in conn.execute(f"PRAGMA foreign_key_list({q})")]
                count = conn.execute(f"SELECT COUNT(*) FROM {q}").fetchone()[0]
                tables[name] = Table(name, cols, count, fks)
            return tables
        finally:
            conn.close()

    def render_schema(self, tables: list[str] | None = None) -> str:
        chosen = tables if tables is not None else list(self.schema)
        return "\n\n".join(self.schema[t].render() for t in chosen if t in self.schema)

    def table_names(self) -> list[str]:
        return list(self.schema)

    def resolve_table(self, name: str) -> str | None:
        lookup = {t.lower(): t for t in self.schema}
        return lookup.get(name.strip().strip('"`[]').lower())
