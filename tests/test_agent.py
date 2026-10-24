import pytest

from sqlagent.agent import AgentConfig, SQLAgent
from sqlagent.llm import ModelRefusal
from tests.conftest import ScriptedModel, text, tool

Q = "How many tracks does AC/DC have?"
GOOD = ("SELECT COUNT(*) FROM Track t JOIN Album a ON a.AlbumId = t.AlbumId "
        "JOIN Artist ar ON ar.ArtistId = a.ArtistId WHERE ar.Name = 'AC/DC'")


def run(db, turns, **cfg):
    model = ScriptedModel(turns)
    return SQLAgent(model, db, AgentConfig(**cfg)).run(Q), model


def test_self_correction_after_sql_error(db):
    res, model = run(db, [
        [tool("run_sql", sql="SELECT COUNT(*) FROM Tracks")],          # typo -> error
        [tool("run_sql", sql=GOOD)],                                      # fixed
        [tool("submit_answer", sql=GOOD, explanation="joins track to artist")],
    ])
    assert res.status == "answered" and res.rows == [(3,)]
    assert res.turns == 3 and res.tool_calls == 3 and res.sql_attempts == 3
    assert [s.ok for s in res.steps] == [False, True, True]
    assert "no such table" in res.steps[0].summary
    # The error was fed back to the model as an is_error tool result.
    tool_result = model.requests[1]["messages"][-1]["content"][0]
    assert tool_result["is_error"] is True and "SQL error" in tool_result["content"]


def test_history_is_append_only(db):
    _, model = run(db, [[tool("list_tables")], [tool("submit_answer", sql=GOOD, explanation="")]])
    first, second = model.requests[0]["messages"], model.requests[1]["messages"]
    assert second[: len(first)] == first
    assert second[1]["role"] == "assistant" and second[2]["role"] == "user"


def test_guard_rejection_is_reported_and_counted(db):
    res, _ = run(db, [
        [tool("run_sql", sql="DELETE FROM Track")],
        [tool("submit_answer", sql=GOOD, explanation="")],
    ])
    assert res.status == "answered" and res.guard_blocks == 1
    assert "Rejected by SQL policy" in res.steps[0].summary


def test_parallel_tool_calls_return_in_one_message(db):
    _, model = run(db, [
        [tool("describe_table", table="Track"), tool("describe_table", table="album")],
        [tool("submit_answer", sql=GOOD, explanation="")],
    ])
    results = model.requests[1]["messages"][-1]["content"]
    assert len(results) == 2 and all(r["type"] == "tool_result" for r in results)
    assert "TABLE Album" in results[1]["content"]


def test_unknown_table_lists_alternatives(db):
    res, _ = run(db, [[tool("describe_table", table="Songs")],
                      [tool("submit_answer", sql=GOOD, explanation="")]])
    assert "Tables: Album, Artist, Genre, Track" in res.steps[0].summary


def test_nudge_then_give_up_without_submission(db):
    res, model = run(db, [[text("The answer is 3.")], [text("Still 3.")]])
    assert res.status == "no_submission" and res.turns == 2
    assert "submit_answer" in model.requests[1]["messages"][-1]["content"]


def test_max_turns(db):
    res, _ = run(db, [[tool("list_tables")]] * 5, max_turns=3)
    assert res.status == "max_turns" and res.turns == 3


def test_cost_budget(db):
    # Each scripted turn costs (1000 * 4 + 100 * 20) / 1e6 = $0.006.
    res, _ = run(db, [[tool("list_tables")]] * 10, max_cost_usd=0.01)
    assert res.status == "budget_exhausted" and res.turns == 2
    assert res.cost_usd == pytest.approx(0.012)


def test_tool_call_budget(db):
    res, _ = run(db, [[tool("list_tables")]] * 4, max_tool_calls=2, max_turns=4)
    assert res.steps[2].summary.startswith("Tool budget exhausted")


def test_direct_mode_has_no_feedback(db):
    res, model = run(db, [[tool("submit_answer", sql="SELECT COUNT(*) FROM Tracks",
                                explanation="")]], mode="direct")
    assert model.requests[0]["tools"] == ["submit_answer"]
    assert res.status == "submission_failed" and res.turns == 1


def test_direct_mode_success(db):
    res, _ = run(db, [[tool("submit_answer", sql=GOOD, explanation="")]], mode="direct")
    assert res.status == "answered" and res.rows == [(3,)]


def test_linked_schema_shows_subset(db):
    model = ScriptedModel([[tool("submit_answer", sql=GOOD, explanation="")]])
    agent = SQLAgent(model, db, AgentConfig(schema_mode="linked", link_k=1))
    res = agent.run("Which genre is the album Kind of Blue?")
    prompt = model.requests[0]["messages"][0]["content"]
    assert len(res.schema_tables) < 4 and "chosen as most relevant" in prompt
    assert "list_tables" in model.requests[0]["tools"]  # can still recover missing tables


def test_untrusted_data_is_wrapped(db):
    res, model = run(db, [[tool("run_sql", sql="SELECT Name FROM Artist")],
                          [tool("submit_answer", sql=GOOD, explanation="")]])
    out = model.requests[1]["messages"][-1]["content"][0]["content"]
    assert out.startswith("<rows>") and "AC/DC" in out
    assert "untrusted" in model.requests[0]["system"]


def test_model_refusal_is_recorded(db):
    class Refuser(ScriptedModel):
        def create(self, *a, **k):
            raise ModelRefusal("declined")

    res = SQLAgent(Refuser([]), db).run(Q)
    assert res.status == "model_error" and "declined" in res.error
