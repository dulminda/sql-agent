from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from sqlagent.db import Database
from sqlagent.llm import Turn


@dataclass
class Block:
    type: str
    text: str = ""
    id: str = ""
    name: str = ""
    input: dict[str, Any] = field(default_factory=dict)


def tool(name: str, **inputs: Any) -> Block:
    tool.counter += 1
    return Block("tool_use", id=f"toolu_{tool.counter}", name=name, input=inputs)


tool.counter = 0


def text(t: str) -> Block:
    return Block("text", text=t)


class ScriptedModel:
    """Plays back a fixed list of assistant turns and records every request it receives."""

    model = "claude-opus-5-5"

    def __init__(self, turns: list[list[Block]], stop_reason: str | None = None,
                 tokens: tuple[int, int] = (1000, 100)) -> None:
        self.turns, self.stop_reason, self.tokens = list(turns), stop_reason, tokens
        self.requests: list[dict] = []

    def create(self, system: str, messages: list[dict], tools: list[dict]) -> Turn:
        self.requests.append({"system": system, "messages": [dict(m) for m in messages],
                              "tools": [t["name"] for t in tools]})
        content = self.turns.pop(0) if self.turns else [text("I am done.")]
        stop = self.stop_reason or ("tool_use" if any(b.type == "tool_use" for b in content)
                                    else "end_turn")
        return Turn(content, stop, self.tokens[0], self.tokens[1], self.model)


@pytest.fixture
def db_file(tmp_path: Path) -> Path:
    """A tiny music store with the same shape as Chinook's core tables."""
    path = tmp_path / "shop.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE Artist (ArtistId INTEGER PRIMARY KEY, Name TEXT);
        CREATE TABLE Album (AlbumId INTEGER PRIMARY KEY, Title TEXT,
                            ArtistId INTEGER REFERENCES Artist(ArtistId));
        CREATE TABLE Genre (GenreId INTEGER PRIMARY KEY, Name TEXT);
        CREATE TABLE Track (TrackId INTEGER PRIMARY KEY, Name TEXT,
                            AlbumId INTEGER REFERENCES Album(AlbumId),
                            GenreId INTEGER REFERENCES Genre(GenreId), Milliseconds INTEGER);
        INSERT INTO Artist VALUES (1, 'AC/DC'), (2, 'Miles Davis'), (3, 'Nobody');
        INSERT INTO Album VALUES (1, 'Back in Black', 1), (2, 'Kind of Blue', 2),
                                 (3, 'Highway to Hell', 1);
        INSERT INTO Genre VALUES (1, 'Rock'), (2, 'Jazz');
        INSERT INTO Track VALUES (1, 'Hells Bells', 1, 1, 312000), (2, 'So What', 2, 2, 545000),
                                 (3, 'Shoot to Thrill', 1, 1, 317000),
                                 (4, 'Highway to Hell', 3, 1, 208000);
        """
    )
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def db(db_file: Path) -> Database:
    return Database(db_file, timeout_s=1.0)
