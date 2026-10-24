"""Offline schema-linking evaluation: does the linker keep the tables the gold SQL needs?"""

from __future__ import annotations

from typing import Any

import numpy as np

from sqlagent.db import Database
from sqlagent.eval.datasets import Example
from sqlagent.eval.stats import bootstrap_ci
from sqlagent.guard import referenced_tables
from sqlagent.schema_linking import SchemaLinker


def run_linking_eval(
    examples: list[Example], dbs: dict[str, Database], ks: tuple[int, ...] = (1, 2, 3, 4, 6),
) -> list[dict[str, Any]]:
    linkers = {db_id: SchemaLinker(db) for db_id, db in dbs.items()}
    full_chars = {db_id: len(db.render_schema()) for db_id, db in dbs.items()}
    out = []
    for k in ks:
        for expand in (False, True):
            recall, covered, n_tables, chars = [], [], [], []
            for ex in examples:
                db = dbs[ex.db_id]
                gold = {t.lower() for t in referenced_tables(ex.gold_sql)}
                linked = linkers[ex.db_id].link(ex.question, k=k, expand_fk=expand)
                got = {t.lower() for t in linked}
                recall.append(len(gold & got) / len(gold) if gold else 1.0)
                covered.append(float(gold <= got))
                n_tables.append(len(linked))
                chars.append(len(db.render_schema(linked)) / full_chars[ex.db_id])
            lo, hi = bootstrap_ci(covered)
            out.append({
                "k": k, "fk_expansion": expand,
                "table_recall": float(np.mean(recall)),
                "all_gold_tables_covered": float(np.mean(covered)),
                "covered_ci": [lo, hi],
                "mean_tables_shown": float(np.mean(n_tables)),
                "schema_size_vs_full": float(np.mean(chars)),
            })
    return out
