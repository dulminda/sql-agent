"""Benchmark loaders.

* ``jsonl``: this repo's format (``benchmarks/chinook.jsonl``), one example per line.
* ``bird``: BIRD dev (``dev.json`` + ``dev_databases/<db_id>/<db_id>.sqlite``),
  fields ``question_id, db_id, question, evidence, SQL, difficulty``.
* ``spider``: Spider (``dev.json`` + ``database/<db_id>/<db_id>.sqlite``),
  fields ``db_id, question, query``.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

CHINOOK_URL = (
    "https://raw.githubusercontent.com/lerocha/chinook-database/master/"
    "ChinookDatabase/DataSources/Chinook_Sqlite.sqlite"
)


@dataclass(frozen=True)
class Example:
    id: str
    db_id: str
    question: str
    gold_sql: str
    evidence: str = ""
    difficulty: str = "unknown"


def download_chinook(dest: str | Path) -> Path:
    dest = Path(dest)
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(CHINOOK_URL, dest)  # noqa: S310 - fixed https URL
    return dest


def load_jsonl(path: str | Path) -> list[Example]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out.append(Example(r["id"], r["db_id"], r["question"], r["gold_sql"],
                                   r.get("evidence", ""), r.get("difficulty", "unknown")))
    return out


def load_bird(questions: str | Path) -> list[Example]:
    data = json.loads(Path(questions).read_text(encoding="utf-8"))
    return [
        Example(str(r.get("question_id", i)), r["db_id"], r["question"], r["SQL"],
                r.get("evidence", ""), r.get("difficulty", "unknown"))
        for i, r in enumerate(data)
    ]


def load_spider(questions: str | Path) -> list[Example]:
    data = json.loads(Path(questions).read_text(encoding="utf-8"))
    return [Example(str(i), r["db_id"], r["question"], r["query"]) for i, r in enumerate(data)]


def load_examples(fmt: str, questions: str | Path) -> list[Example]:
    loaders = {"jsonl": load_jsonl, "bird": load_bird, "spider": load_spider}
    if fmt not in loaders:
        raise ValueError(f"unknown benchmark format {fmt!r}; expected one of {sorted(loaders)}")
    return loaders[fmt](questions)


def db_path(db_root: str | Path, db_id: str) -> Path:
    """Find ``<root>/<db_id>/<db_id>.sqlite`` (BIRD/Spider) or ``<root>/<db_id>.sqlite``."""
    root = Path(db_root)
    for candidate in (root / db_id / f"{db_id}.sqlite", root / f"{db_id}.sqlite"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no database for {db_id!r} under {root}")
