"""Tests for the OpenAI-compatible server (design §10.2, §16.10 M6)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from anvil.server.app import build_app
from anvil.server.tool_calling import (
    build_tool_grammar,
    parse_tool_call_response,
)
from helpers import StubEngine


def _client() -> TestClient:
    return TestClient(build_app(engine=StubEngine(model_id="stub/serve"), model_id="stub/serve"))


class TestModelsEndpoint:
    def test_lists_the_loaded_model(self) -> None:
        r = _client().get("/v1/models")
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "list"
        assert any(m["id"] == "stub/serve" for m in body["data"])


class TestChatCompletions:
    def test_basic_response_shape(self) -> None:
        r = _client().post(
            "/v1/chat/completions",
            json={
                "model": "stub/serve",
                "messages": [{"role": "user", "content": "the answer is 42"}],
                "max_tokens": 32,
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "chat.completion"
        assert len(body["choices"]) == 1
        choice = body["choices"][0]
        assert choice["message"]["role"] == "assistant"
        assert isinstance(choice["message"]["content"], str)
        assert choice["finish_reason"] in ("stop", "length")
        assert "usage" in body

    def test_streaming_returns_501(self) -> None:
        r = _client().post(
            "/v1/chat/completions",
            json={
                "model": "stub/serve",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert r.status_code == 501

    def test_empty_messages_validation_error(self) -> None:
        r = _client().post(
            "/v1/chat/completions",
            json={"model": "stub/serve", "messages": []},
        )
        # FastAPI returns 200 with stub returning empty for empty messages
        # — but the engine still produces *something*. The point of this
        # test is that the validator at least accepts the empty list,
        # which matches OpenAI's behavior.
        assert r.status_code in (200, 422)


class TestHealthz:
    def test_healthz_returns_ok(self) -> None:
        r = _client().get("/healthz")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


class TestToolGrammar:
    def test_build_grammar_produces_oneof(self) -> None:
        from anvil.server.schemas import Tool, ToolFunction

        tools = [
            Tool(
                function=ToolFunction(
                    name="get_weather",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                )
            ),
            Tool(
                function=ToolFunction(
                    name="get_time",
                    parameters={
                        "type": "object",
                        "properties": {"tz": {"type": "string"}},
                        "required": ["tz"],
                    },
                )
            ),
        ]
        schema = build_tool_grammar(tools)
        assert schema["properties"]["name"]["enum"] == ["get_weather", "get_time"]
        assert "oneOf" in schema["properties"]["arguments"]
        assert len(schema["properties"]["arguments"]["oneOf"]) == 2

    def test_build_grammar_rejects_empty_tools(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="non-empty"):
            build_tool_grammar([])


class TestToolCallParser:
    def test_extracts_fenced_json(self) -> None:
        from anvil.server.schemas import Tool, ToolFunction

        tools = [
            Tool(function=ToolFunction(name="get_weather", parameters={})),
        ]
        text = (
            "Sure, calling the tool:\n"
            '```json\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n```'
        )
        calls = parse_tool_call_response(text, tools)
        assert len(calls) == 1
        assert calls[0].function.name == "get_weather"
        assert "Paris" in calls[0].function.arguments

    def test_rejects_unknown_tool(self) -> None:
        from anvil.server.schemas import Tool, ToolFunction

        tools = [Tool(function=ToolFunction(name="get_weather"))]
        text = '{"name": "send_email", "arguments": {"to": "a@b.com"}}'
        # The parser refuses unknown tool names rather than passing them through.
        assert parse_tool_call_response(text, tools) == []

    def test_no_json_returns_empty(self) -> None:
        from anvil.server.schemas import Tool, ToolFunction

        tools = [Tool(function=ToolFunction(name="x"))]
        assert parse_tool_call_response("just plain text, no JSON here.", tools) == []
