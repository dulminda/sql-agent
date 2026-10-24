import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fastapi.testclient import TestClient

from sqlagent.api.app import create_app
from sqlagent.llm import ClaudeChat, ModelRefusal, cost_usd
from tests.conftest import ScriptedModel, tool

GOOD = "SELECT COUNT(*) FROM Track"


@pytest.fixture
def client(db):
    return TestClient(create_app(
        db, model_factory=lambda: ScriptedModel([[tool("submit_answer", sql=GOOD,
                                                        explanation="count")]])))


def test_health_schema_query(client):
    assert client.get("/healthz").json()["agent"] is True
    assert "TABLE Track" in client.get("/v1/schema").json()["schema"]
    assert client.post("/v1/query", json={"sql": GOOD}).json()["rows"] == [[4]]
    r = client.post("/v1/query", json={"sql": "DROP TABLE Track"})
    assert r.status_code == 400 and "policy" in r.json()["detail"]


def test_ask(client):
    body = client.post("/v1/ask", json={"question": "How many tracks?"}).json()
    assert body["status"] == "answered" and body["rows"] == [[4]]


def test_ask_disabled_without_model(db, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    c = TestClient(create_app(db))
    assert c.post("/v1/ask", json={"question": "q"}).status_code == 503


# --- ClaudeChat against a local mock of the Messages API ---------------------------------

class _Server:
    def __init__(self):
        self.requests, self.reply = [], {}
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                outer.requests.append({"beta": self.headers.get("anthropic-beta"), "body": body})
                out = json.dumps(outer.reply).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

            def log_message(self, *a):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"


@pytest.fixture
def server(monkeypatch):
    s = _Server()
    monkeypatch.setenv("ANTHROPIC_BASE_URL", s.url)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    yield s
    s.httpd.shutdown()


def _reply(stop_reason="tool_use"):
    return {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
        "content": [
            {"type": "thinking", "thinking": "", "signature": "sig"},
            {"type": "tool_use", "id": "toolu_1", "name": "run_sql", "input": {"sql": GOOD}},
        ],
        "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": 200, "output_tokens": 50, "cache_read_input_tokens": 3000,
                  "cache_creation_input_tokens": 0},
    }


def test_claude_request_shape_and_usage(server):
    from sqlagent.agent import TOOLS

    server.reply = _reply()
    turn = ClaudeChat(effort="low", max_retries=0).create(
        "system prompt", [{"role": "user", "content": "q"}], list(TOOLS.values()))
    req = server.requests[0]
    assert req["beta"] == "server-side-fallback-2026-07-01"
    body = req["body"]
    assert body["fallbacks"] == "default" and body["output_config"] == {"effort": "low"}
    assert "tool_choice" not in body  # forced tool use is rejected by current models
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert all(t["strict"] and t["input_schema"]["additionalProperties"] is False
               for t in body["tools"])
    assert [b.type for b in turn.content] == ["thinking", "tool_use"]
    assert turn.tool_uses()[0].input == {"sql": GOOD}
    assert turn.cost_usd == pytest.approx(cost_usd("claude-opus-5-5", 200, 50, cache_read=3000))


def test_claude_refusal(server):
    server.reply = {**_reply("refusal"), "content": []}
    with pytest.raises(ModelRefusal):
        ClaudeChat(max_retries=0).create("s", [{"role": "user", "content": "q"}], [])


def test_cache_pricing():
    # 1M cache-read tokens on Opus 5.5 cost $0.20, a twentieth of fresh input.
    assert cost_usd("claude-opus-5-5", 0, 0, cache_read=1_000_000) == pytest.approx(0.20)
    assert cost_usd("claude-opus-5-5", 0, 0, cache_write=1_000_000) == pytest.approx(5.00)
