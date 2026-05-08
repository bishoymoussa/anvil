"""Anvil — research-first, evaluation-first inference (design §16.3).

Public API: only the names in :data:`__all__` are stable. Everything else
(``anvil._engine``, ``anvil._version``, ``anvil.config``, …) is internal and
may break in any minor release.
"""

from __future__ import annotations

# ``anvil.research`` is a submodule users import as ``from anvil import research``.
from anvil import research
from anvil._version import __version__
from anvil.manifest.schema import Manifest
from anvil.metrics.exact_match import exact_match
from anvil.metrics.pass_at_k import pass_at_k
from anvil.models import load, load_custom
from anvil.primitives.chat_template import ChatTemplate
from anvil.primitives.hidden_state_spec import HiddenStateSpec
from anvil.primitives.logits_processor import LogitsProcessor
from anvil.primitives.request import Classify, Custom, Embed, Generate, LogLikelihood
from anvil.primitives.response import (
    ClassifyResult,
    EmbedResult,
    Generation,
    Response,
)
from anvil.primitives.sampler import Sampler
from anvil.primitives.tokenization import Tokenization
from anvil.tasks import Task, eval, register_task

# Optional: serve only available if the server module's deps are present.
try:
    from anvil.server import serve  # noqa: F401
except ImportError:  # pragma: no cover - extras-dependent
    serve = None  # type: ignore[assignment]


__all__ = [
    "__version__",
    "load",
    "load_custom",
    "eval",
    "Task",
    "register_task",
    "serve",
    "ChatTemplate",
    "Tokenization",
    "Sampler",
    "LogitsProcessor",
    "HiddenStateSpec",
    "Generate",
    "LogLikelihood",
    "Embed",
    "Classify",
    "Custom",
    "Response",
    "Generation",
    "EmbedResult",
    "ClassifyResult",
    "Manifest",
    "exact_match",
    "pass_at_k",
    "research",
]
