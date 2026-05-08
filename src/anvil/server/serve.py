"""``anvil.serve`` — top-level entry point that boots the OpenAI-compatible server."""

from __future__ import annotations

from anvil.logging import get_logger

_log = get_logger(__name__)


def serve(
    *,
    model: str,
    port: int = 8000,
    host: str = "0.0.0.0",  # noqa: S104 - dev/eval server, intentional default
    tool_calling: str = "auto",
    structured_output: str = "xgrammar",
    engine: str = "auto",
    dtype: str | None = None,
    revision: str | None = None,
) -> None:
    """Run the OpenAI-compatible HTTP server (design §10.2).

    Args:
        model: HF model id or local path. Loaded once at startup.
        port: TCP port to bind.
        host: bind address. Defaults to ``0.0.0.0`` for dev convenience —
            tighten in any production deployment.
        tool_calling: ``"auto"`` (default) honors the request's ``tools``
            field via constrained decoding; ``"off"`` returns 400 if the
            client sends ``tools``.
        structured_output: ``"xgrammar"`` (default) | ``"outlines"`` |
            ``"lmfe"``. The xgrammar path lands first; the others are
            documented for plugin-equivalence and raise if the
            corresponding extra isn't installed.
        engine: backend choice; passed to :func:`anvil.engine.build_engine`.
        dtype, revision: forwarded to the engine.
    """
    del tool_calling, structured_output  # honored at request time, not boot
    import uvicorn

    from anvil.engine import build_engine
    from anvil.server.app import build_app

    eng = build_engine(
        model_id=model,
        engine=engine,  # type: ignore[arg-type]
        revision=revision,
        dtype=dtype,
    )
    app = build_app(engine=eng, model_id=model)
    _log.info("anvil serve listening on %s:%d (model=%s)", host, port, model)
    uvicorn.run(app, host=host, port=port, log_level="info")


__all__ = ["serve"]
