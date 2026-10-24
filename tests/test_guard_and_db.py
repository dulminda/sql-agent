import hashlib

import pytest

from sqlagent.db import Database, QueryError
from sqlagent.guard import check_sql, referenced_tables

ATTACKS = [
    "DROP TABLE Artist",
    "DELETE FROM Artist",
    "UPDATE Artist SET Name = 'x'",
    "INSERT INTO Artist (Name) VALUES ('x')",
    "REPLACE INTO Artist VALUES (1, 'x')",
    "SELECT 1; DROP TABLE Artist",
    "WITH x AS (DELETE FROM Artist RETURNING *) SELECT * FROM x",
    "ATTACH DATABASE '/tmp/evil.db' AS evil",
    "PRAGMA query_only = OFF",
    "VACUUM",
    "CREATE TEMP TABLE t AS SELECT * FROM Artist",
    "SELECT load_extension('evil')",
    "SELECT readfile('/etc/passwd')",
    "BEGIN; DELETE FROM Artist; COMMIT",
]
BENIGN = [
    "SELECT * FROM Artist WHERE Name = 'DROP TABLE Artist'",
    "SELECT Name AS \"delete\" FROM Artist",
    "WITH t AS (SELECT * FROM Track) SELECT COUNT(*) FROM t",
    "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 5) SELECT * FROM r",
    "SELECT Name FROM Artist UNION SELECT Name FROM Genre",
    "SELECT Name, RANK() OVER (ORDER BY Milliseconds) FROM Track",
    "select lower(name) from artist",
]


@pytest.mark.parametrize("sql", ATTACKS)
def test_guard_blocks_attacks(sql):
    assert not check_sql(sql).ok


@pytest.mark.parametrize("sql", ATTACKS)
def test_sandbox_blocks_attacks_even_without_guard(db, db_file, sql):
    before = hashlib.sha256(db_file.read_bytes()).digest()
    with pytest.raises(QueryError):
        db.execute(sql)
    assert hashlib.sha256(db_file.read_bytes()).digest() == before


@pytest.mark.parametrize("sql", BENIGN)
def test_benign_queries_pass_both_layers(db, sql):
    assert check_sql(sql).ok, check_sql(sql).reason
    db.execute(sql)


def test_guard_reasons_are_actionable():
    assert "exactly one statement" in check_sql("SELECT 1; SELECT 2").reason
    assert "DELETE" in check_sql("WITH x AS (DELETE FROM t RETURNING *) SELECT * FROM x").reason
    assert "parse" in check_sql("SELEC * FRM").reason or not check_sql("SELEC * FRM").ok
    assert check_sql("").reason == "empty query"


def test_referenced_tables_excludes_ctes():
    sql = "WITH t AS (SELECT * FROM Track) SELECT * FROM t JOIN Genre g ON 1"
    assert referenced_tables(sql) == {"Track", "Genre"}


def test_timeout_stops_runaway_query(db):
    with pytest.raises(QueryError, match="time limit"):
        db.execute("WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) "
                   "SELECT COUNT(*) FROM c")


def test_row_cap_and_truncation(db_file):
    small = Database(db_file, max_rows=2)
    r = small.execute("SELECT * FROM Track")
    assert len(r.rows) == 2 and r.truncated
    assert not small.execute("SELECT * FROM Genre").truncated


def test_schema_introspection(db):
    track = db.schema["Track"]
    assert track.row_count == 4
    assert ("AlbumId", "Album", "AlbumId") in track.foreign_keys
    rendered = track.render()
    assert "TrackId INTEGER PRIMARY KEY" in rendered
    assert "AlbumId INTEGER REFERENCES Album(AlbumId)" in rendered
    assert "e.g. 'Hells Bells'" in rendered
    assert db.resolve_table('"track"') == "Track" and db.resolve_table("nope") is None
