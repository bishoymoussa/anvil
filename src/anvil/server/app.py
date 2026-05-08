"""FastAPI app factory for the OpenAI-compatible server (design §10.2)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import FastAPI

from anvil._version import __version__
from anvil.server.routes import build_router

if TYPE_CHECKING:
    from anvil.engine.public import Engine


def build_app(*, engine: Engine, model_id: str) -> FastAPI:
    """Construct the FastAPI app, sharing one engine across all requests.

    Args:
        engine: pre-built :class:`Engine` (callers usually obtain it via
            ``anvil.load(model_id).engine``).
        model_id: the model id to advertise on ``/v1/models`` and to echo
            back in chat-completion responses.
    """
    app = FastAPI(
        title="Anvil OpenAI-compatible server",
        version=__version__,
        description=(
            "Drop-in replacement for the OpenAI Chat Completions API. "
            "Tool calling is constrained-decoding-driven (one grammar, "
            "no per-model parser flags)."
        ),
    )
    app.include_router(build_router(engine=engine, model_id=model_id))

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "anvil_version": __version__}

    return app


__all__ = ["build_app"]
