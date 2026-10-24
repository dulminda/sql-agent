"""HTTP API.

Run with ``sqlagent serve`` or
``SQLAGENT_DB=... uvicorn sqlagent.api.app:create_app --factory``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sqlagent.agent import AgentConfig, SQLAgent
from sqlagent.db import Database, QueryError
from sqlagent.guard import check_sql
from sqlagent.llm import ChatModel

log = logging.getLogger("sqlagent.api")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    evidence: str = Field(default="", max_length=2000)


class QueryRequest(BaseModel):
    sql: str = Field(min_length=1, max_length=20000)


def create_app(
    db: Database | None = None,
    config: AgentConfig | None = None,
    model_factory: Callable[[], ChatModel] | None = None,
) -> FastAPI:
    if db is None:
        db = Database(os.environ.get("SQLAGENT_DB", "data/chinook.sqlite"))
    if config is None:
        path = os.environ.get("SQLAGENT_CONFIG")
        config = AgentConfig.model_validate(yaml.safe_load(Path(path).read_text())) if path \
            else AgentConfig()
    if model_factory is None and (
        os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    ):
        from sqlagent.llm import ClaudeChat

        def model_factory() -> ChatModel:
            return ClaudeChat(model=config.model, effort=config.effort)

    app = FastAPI(title="sql-agent", version="0.1.0")

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "tables": len(db.schema), "agent": model_factory is not None,
                "config": config.name}

    @app.get("/v1/schema")
    def schema() -> dict:
        return {"schema": db.render_schema()}

    @app.post("/v1/query")
    def query(req: QueryRequest) -> dict:
        """Run SQL directly, through the same guard and sandbox the agent uses."""
        verdict = check_sql(req.sql)
        if not verdict:
            raise HTTPException(400, f"rejected by SQL policy: {verdict.reason}")
        try:
            r = db.execute(req.sql, max_rows=500)
        except QueryError as e:
            raise HTTPException(400, str(e)) from e
        return {"columns": r.columns, "rows": [list(x) for x in r.rows], "truncated": r.truncated}

    @app.post("/v1/ask")
    def ask(req: AskRequest) -> dict:
        if model_factory is None:
            raise HTTPException(503, "agent disabled: no Anthropic credentials configured")
        res = SQLAgent(model_factory(), db, config).run(req.question, req.evidence)
        if res.status == "model_error":
            log.error("agent failed: %s", res.error)
            raise HTTPException(502, f"upstream model error: {res.error}")
        return res.to_dict()

    return app
