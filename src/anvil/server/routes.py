"""OpenAI-compatible HTTP routes (design §10.2)."""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException

from anvil.exceptions import AnvilError
from anvil.logging import get_logger
from anvil.primitives.request import Generate
from anvil.primitives.sampler import Sampler
from anvil.server.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ModelInfo,
    ModelsResponse,
    ResponseMessage,
    Usage,
)
from anvil.server.tool_calling import parse_tool_call_response

if TYPE_CHECKING:
    from anvil.engine.public import Engine

_log = get_logger(__name__)


def build_router(*, engine: Engine, model_id: str) -> APIRouter:
    """Construct a FastAPI router bound to a pre-built engine.

    The engine is shared across requests; concurrency is handled by the
    HTTP server (uvicorn workers). Per the §12 non-goal we don't ship
    request batching at the HTTP layer — that's vLLM's job. For the
    development + small-scale eval use case the simple per-request loop
    is plenty.
    """
    router = APIRouter()

    @router.get("/v1/models", response_model=ModelsResponse)
    def list_models() -> ModelsResponse:
        return ModelsResponse(data=[ModelInfo(id=model_id, created=int(time.time()))])

    @router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    def chat_completions(req: ChatCompletionRequest) -> ChatCompletionResponse:
        if req.stream:
            raise HTTPException(
                status_code=501,
                detail="streaming responses are not implemented in v0; "
                "set stream=false or use anvil.eval(...) directly.",
            )
        sampler = _sampler_from_request(req)
        # Convert ChatMessage to plain dicts for the engine's chat-template path.
        plain_messages = [_to_plain_message(m) for m in req.messages]
        gen = Generate(messages=tuple(plain_messages), sampler=sampler)

        try:
            outs = engine.generate_logprobs([gen])
        except AnvilError as exc:
            raise HTTPException(
                status_code=500, detail={"error": str(exc), "code": exc.error_code}
            ) from exc
        if not outs:
            raise HTTPException(status_code=500, detail="engine returned no output")

        gen_out = outs[0]

        # Tool calling: if the request supplied tools, try to parse the
        # generation as a tool call. xgrammar would have constrained
        # decoding upstream; without it, the parser is best-effort.
        tool_calls = None
        if req.tools and req.tool_choice in (None, "auto"):
            tool_calls = parse_tool_call_response(gen_out.text, req.tools)
            if tool_calls:
                message = ResponseMessage(content=None, tool_calls=tool_calls)
                finish = "tool_calls"
            else:
                message = ResponseMessage(content=gen_out.text)
                finish = (
                    gen_out.finish_reason
                    if gen_out.finish_reason
                    in (
                        "stop",
                        "length",
                    )
                    else "stop"
                )
        else:
            message = ResponseMessage(content=gen_out.text)
            finish = (
                gen_out.finish_reason
                if gen_out.finish_reason
                in (
                    "stop",
                    "length",
                )
                else "stop"
            )

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
            created=int(time.time()),
            model=req.model,
            choices=[
                Choice(
                    index=0,
                    message=message,
                    finish_reason=finish,
                )
            ],
            usage=Usage(
                prompt_tokens=gen_out.prompt_token_count,
                completion_tokens=len(gen_out.token_ids),
                total_tokens=gen_out.prompt_token_count + len(gen_out.token_ids),
            ),
        )

    return router


def _sampler_from_request(req: ChatCompletionRequest) -> Sampler:
    """Translate the OpenAI request shape to an Anvil ``Sampler``.

    The default OpenAI temperature is 1.0; Anvil's default is 0.0
    (greedy). We honor the request's value explicitly to match the
    OpenAI client's expectations.
    """
    stop = req.stop
    stop_tuple: tuple[str, ...] = ()
    if isinstance(stop, str):
        stop_tuple = (stop,)
    elif isinstance(stop, list):
        stop_tuple = tuple(stop)
    return Sampler(
        temperature=req.temperature,
        top_p=req.top_p,
        n=req.n,
        max_tokens=req.max_tokens or 2048,
        seed=req.seed,
        stop=stop_tuple,
        source="explicit",
    )


def _to_plain_message(m: Any) -> dict[str, Any]:
    """Strip Pydantic-specific bits so the engine's chat template sees plain dicts."""
    out: dict[str, Any] = m.model_dump(exclude_none=True)
    return out


__all__ = ["build_router"]
