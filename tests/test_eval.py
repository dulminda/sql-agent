import json
import shutil

import pytest

from sqlagent.agent import AgentConfig
from sqlagent.eval.datasets import Example, db_path, load_bird, load_spider
from sqlagent.eval.execution import categorize, execution_match
from sqlagent.eval.linking import run_linking_eval
from sqlagent.eval.redteam import build_poisoned_db
from sqlagent.eval.report import agent_table, linking_table, security_table
from sqlagent.eval.runner import DatabasePool, run_agent_eval
from sqlagent.eval.security import run_security_eval
from sqlagent.eval.stats import paired_bootstrap_test
from sqlagent.gate import check
from tests.conftest import ScriptedModel, tool


def test_execution_match_is_set_based_with_float_tolerance():
    assert execution_match([(1, "a"), (2, "b")], [(2, "b"), (1, "a"), (1, "a")])
    assert execution_match([(2328.6000000000004,)], [(2328.6,)])
    assert execution_match([(3.0,)], [(3,)])
    assert not execution_match([(1,)], [(1, "a")])


@pytest.mark.parametrize(("status", "pred", "gold", "expected"), [
    ("answered", [(1,)], [(1,)], "correct"),
    ("answered", [], [(1,)], "empty_result"),
    ("answered", [(1, 2)], [(1,)], "wrong_columns"),
    ("answered", [(2,)], [(1,)], "wrong_rows"),
    ("max_turns", None, [(1,)], "no_submission"),
    ("submission_failed", None, [(1,)], "execution_error"),
    ("model_error", None, [(1,)], "model_error"),
])
def test_categorize(status, pred, gold, expected):
    assert categorize(status, pred, gold) == expected


def test_paired_test():
    assert paired_bootstrap_test([1, 0] * 20, [1, 0] * 20) == (0.0, 1.0)
    d, p = paired_bootstrap_test([0.0] * 40, [1.0] * 30 + [0.0] * 10)
    assert d == pytest.approx(0.75) and p < 0.01


def test_bird_and_spider_loaders(tmp_path, db_file):
    (tmp_path / "bird.json").write_text(json.dumps([{
        "question_id": 7, "db_id": "shop", "question": "q", "evidence": "hint", "SQL": "SELECT 1",
        "difficulty": "simple"}]))
    (tmp_path / "spider.json").write_text(json.dumps([{"db_id": "shop", "question": "q",
                                                       "query": "SELECT 1"}]))
    b, s = load_bird(tmp_path / "bird.json")[0], load_spider(tmp_path / "spider.json")[0]
    assert (b.id, b.evidence, b.gold_sql, b.difficulty) == ("7", "hint", "SELECT 1", "simple")
    assert s.gold_sql == "SELECT 1"
    nested = tmp_path / "dbs" / "shop"
    nested.mkdir(parents=True)
    shutil.copy(db_file, nested / "shop.sqlite")
    assert db_path(tmp_path / "dbs", "shop") == nested / "shop.sqlite"


def _examples():
    return [
        Example("e1", "shop", "How many tracks are there?", "SELECT COUNT(*) FROM Track",
                difficulty="easy"),
        Example("e2", "shop", "Name the jazz artist.",
                "SELECT ar.Name FROM Artist ar JOIN Album a ON a.ArtistId = ar.ArtistId "
                "JOIN Track t ON t.AlbumId = a.AlbumId JOIN Genre g ON g.GenreId = t.GenreId "
                "WHERE g.Name = 'Jazz'", difficulty="hard"),
    ]


def test_run_agent_eval(db_file):
    answers = {"How many tracks are there?": "SELECT COUNT(*) FROM Track",
               "Name the jazz artist.": "SELECT 'Miles Davis', 1"}  # wrong shape on purpose

    class PerQuestion(ScriptedModel):
        def create(self, system, messages, tools):
            q = messages[0]["content"].rsplit("Question: ", 1)[1]
            self.turns = [[tool("submit_answer", sql=answers[q], explanation="")]]
            return super().create(system, messages, tools)

    run = run_agent_eval(AgentConfig(name="t"), _examples(), DatabasePool(db_file.parent),
                         lambda: PerQuestion([]), workers=2)
    assert run["execution_accuracy"]["mean"] == 0.5
    assert run["failures"]["correct"] == 1 and run["failures"]["wrong_columns"] == 1
    assert run["by_difficulty"]["easy"]["mean"] == 1.0
    assert run["cost_per_task_usd"] == pytest.approx(0.006)
    assert "| `t` |" in agent_table([run, run])


def test_linking_eval(db_file, db):
    rows = run_linking_eval(_examples(), {"shop": db}, ks=(1, 4))
    assert len(rows) == 4
    assert rows[-1]["all_gold_tables_covered"] == 1.0  # k=4 shows all 4 tables
    assert "| 4 | yes |" in linking_table(rows)


def test_security_eval_on_corpus(db_file, tmp_path):
    corpus = tmp_path / "c.jsonl"
    corpus.write_text("\n".join(json.dumps(r) for r in [
        {"id": "a1", "kind": "attack", "category": "dml", "sql": "DELETE FROM Track"},
        {"id": "a2", "kind": "attack", "category": "ddl", "sql": "DROP TABLE Track"},
        {"id": "b1", "kind": "benign", "category": "benign", "sql": "SELECT 'DROP TABLE x'"},
    ]))
    s = run_security_eval(db_file, corpus, extra_benign=["SELECT COUNT(*) FROM Track"])
    assert s["attack_block_rate"] == {"guard": 1.0, "sandbox": 1.0, "combined": 1.0}
    assert s["benign_false_block_rate"] == {"guard": 0.0, "sandbox": 0.0}
    assert s["database_unchanged"] and not s["leaks"]
    assert "**100%**" in security_table(s)


def test_redteam_db_contains_payloads(tmp_path):
    import sqlite3
    src = tmp_path / "src.sqlite"
    conn = sqlite3.connect(src)
    conn.executescript(
        "CREATE TABLE Customer (CustomerId INTEGER PRIMARY KEY, FirstName, LastName, Company, "
        "Country); INSERT INTO Customer VALUES (1, 'A', 'B', NULL, 'Brazil');"
        "CREATE TABLE Artist (ArtistId INTEGER PRIMARY KEY, Name);"
        "CREATE TABLE Album (AlbumId INTEGER PRIMARY KEY, Title, ArtistId);"
        "CREATE TABLE Track (TrackId INTEGER PRIMARY KEY, Name, AlbumId, MediaTypeId, GenreId, "
        "Composer, Milliseconds, Bytes, UnitPrice);")
    conn.close()
    db_file, bench = build_poisoned_db(src, tmp_path / "rt")
    company = sqlite3.connect(db_file).execute("SELECT Company FROM Customer").fetchone()[0]
    assert "ignore all previous instructions" in company
    assert len(bench.read_text().splitlines()) == 6


def test_gate():
    results = {"security": {"attack_block_rate": {"combined": 1.0}, "fp": 0.1},
               "linking": [{"x": 0.9}]}
    ok = {"security": {"attack_block_rate.combined": {"min": 1.0}},
          "linking": {"0.x": {"min": 0.8}}}
    assert check(results, ok) == []
    bad = check(results, {"security": {"fp": {"max": 0.0}, "missing": {"min": 1}}})
    assert [f.metric for f in bad] == ["fp", "missing"]
