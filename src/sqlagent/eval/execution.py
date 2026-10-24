"""Execution accuracy and failure taxonomy.

Execution accuracy (EX) follows BIRD's definition: a prediction is correct when its result
set equals the gold result set (``set(pred) == set(gold)``; order and duplicates ignored).
Floats are rounded to 6 places first so ``2328.6`` and ``2328.6000000000004`` match.

Every miss is put into exactly one failure category, so error analysis is a table, not
anecdotes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

FAILURE_CATEGORIES = (
    "correct",
    "no_submission",      # ran out of turns/budget, or never called submit_answer
    "model_error",        # API failure or refusal
    "execution_error",    # submitted SQL rejected by policy or failed to run
    "empty_result",       # ran, returned nothing, gold was non-empty
    "wrong_columns",      # different number of columns than gold
    "wrong_rows",         # right shape, wrong values or row set
)


def _norm(v: Any) -> Any:
    if isinstance(v, float):
        r = round(v, 6)
        return int(r) if r.is_integer() else r
    return v


def normalize(rows: Sequence[Sequence[Any]]) -> set[tuple]:
    return {tuple(_norm(v) for v in row) for row in rows}


def execution_match(pred_rows: Sequence[Sequence[Any]], gold_rows: Sequence[Sequence[Any]]) -> bool:
    return normalize(pred_rows) == normalize(gold_rows)


def categorize(status: str, pred_rows: list | None, gold_rows: list) -> str:
    if status == "model_error":
        return "model_error"
    if status == "submission_failed":
        return "execution_error"
    if status != "answered" or pred_rows is None:
        return "no_submission"
    if execution_match(pred_rows, gold_rows):
        return "correct"
    if not pred_rows and gold_rows:
        return "empty_result"
    pred_width = len(pred_rows[0]) if pred_rows else 0
    gold_width = len(gold_rows[0]) if gold_rows else 0
    if pred_rows and gold_rows and pred_width != gold_width:
        return "wrong_columns"
    return "wrong_rows"
