"""Layer 1 of the sandbox: a static SQL policy check, run before anything touches the database.

Policy: exactly one read-only query (SELECT / WITH / set operation). The whole syntax tree is
walked, because a write can hide inside an otherwise innocent query, for example
``WITH x AS (DELETE FROM t RETURNING *) SELECT * FROM x``. Rejections come back to the model
as tool errors with the reason, so the agent can correct itself.

This layer is about clear feedback, not the security boundary on its own: the SQLite
authorizer in ``db.py`` enforces the same policy at prepare time (defense in depth).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

logging.getLogger("sqlglot").setLevel(logging.ERROR)

_WRITE_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Create, exp.Drop, exp.Alter,
    exp.TruncateTable, exp.Attach, exp.Detach, exp.Pragma, exp.Command, exp.Transaction,
    exp.Commit, exp.Rollback, exp.Set, exp.Use, exp.Copy, exp.Into,
)
# SQLite functions that reach outside the database or alter connection state.
DENIED_FUNCTIONS = frozenset({
    "load_extension", "readfile", "writefile", "edit", "fts3_tokenizer", "sqlite_offset",
    "zipfile", "fsdir", "shell",
})


@dataclass(frozen=True)
class GuardResult:
    ok: bool
    reason: str = ""
    tables: frozenset[str] = frozenset()

    def __bool__(self) -> bool:
        return self.ok


def check_sql(sql: str, dialect: str = "sqlite") -> GuardResult:
    if not sql or not sql.strip():
        return GuardResult(False, "empty query")
    try:
        statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    except ParseError as e:
        return GuardResult(False, f"could not parse SQL: {str(e).splitlines()[0]}")
    if len(statements) != 1:
        return GuardResult(False, f"exactly one statement is allowed, got {len(statements)}")
    root = statements[0]
    if not isinstance(root, exp.Query):
        return GuardResult(False, f"only read-only queries are allowed, got {root.key.upper()}")
    for node in root.walk():
        if isinstance(node, _WRITE_NODES):
            return GuardResult(False, f"{node.key.upper()} is not allowed in a read-only query")
        if isinstance(node, exp.Func):
            name = (node.name if isinstance(node, exp.Anonymous) else node.sql_name()).lower()
            if name in DENIED_FUNCTIONS:
                return GuardResult(False, f"function {name}() is not allowed")
    tables = frozenset(t.name for t in root.find_all(exp.Table) if t.name)
    return GuardResult(True, tables=tables)


def referenced_tables(sql: str, dialect: str = "sqlite") -> frozenset[str]:
    """Base tables a query reads (CTE names excluded). Used for schema-linking evaluation."""
    try:
        root = sqlglot.parse_one(sql, read=dialect)
    except ParseError:
        return frozenset()
    ctes = {c.alias_or_name.lower() for c in root.find_all(exp.CTE)}
    return frozenset(
        t.name for t in root.find_all(exp.Table) if t.name and t.name.lower() not in ctes
    )
