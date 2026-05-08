"""Pydantic schemas for the OpenAI-compatible server (design §10.2).

The shapes here mirror the OpenAI 1.x API surface: ``messages`` are
typed dicts, ``tools`` carry JSON Schema ``parameters``, etc. We use
Pydantic so request/response bodies validate at FastAPI's edge —
malformed clients see a 422 with a structured error rather than a 500
out of the engine.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ----------------------------------------------------------- request shapes


class ChatMessage(BaseModel):
    """One entry in the chat conversation."""

    model_config = ConfigDict(extra="allow")  # tool_calls field varies

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_call_id: str | None = None


class ToolFunction(BaseModel):
    """JSON-Schema-ish description of a callable tool."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class Tool(BaseModel):
    """A single tool the model may call."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["function"] = "function"
    function: ToolFunction


class ChatCompletionRequest(BaseModel):
    """Subset of the OpenAI ``/v1/chat/completions`` request body.

    Fields not in this subset (``logit_bias`` keyed by string, audio,
    images via URL) are accepted by FastAPI's body validator with
    ``extra="ignore"`` but ignored — surfacing them silently is OK in v0
    because the user's request still produces an answer.
    """

    model_config = ConfigDict(extra="ignore")

    model: str
    messages: list[ChatMessage]
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    max_tokens: int | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    tools: list[Tool] | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None
    stream: bool = False


# ----------------------------------------------------------- response shapes


class FunctionCall(BaseModel):
    """The function the model decided to call."""

    name: str
    arguments: str  # JSON-encoded; OpenAI's convention


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class ResponseMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str | None = None
    tool_calls: list[ToolCall] | None = None


class Choice(BaseModel):
    index: int
    message: ResponseMessage
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter"] = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage = Field(default_factory=Usage)


# -------------------------------------------------------------- /v1/models


class ModelInfo(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = "anvil"


class ModelsResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelInfo]


__all__ = [
    "ChatMessage",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "Tool",
    "ToolFunction",
    "ToolCall",
    "FunctionCall",
    "ResponseMessage",
    "Choice",
    "Usage",
    "ModelsResponse",
    "ModelInfo",
]
