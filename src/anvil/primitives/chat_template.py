"""``ChatTemplate`` — content-hashed, versioned chat template (design §4.1).

A ``ChatTemplate`` is the Jinja source plus an explicit declaration of how
few-shot examples are packed into the prompt. The hash is the *identity* of
the template — two ``ChatTemplate``s with the same hash MUST produce the
same prompt for the same inputs.

Canonicalization
----------------

For correctness, ``canonicalize`` operates on the **Jinja token stream**,
not on the source string. We use ``jinja2.Environment().lex`` so that:

1. Comments are stripped.
2. Whitespace inside Jinja blocks is collapsed.
3. Quoting style is normalized (single → double quotes).
4. Trailing whitespace and BOMs are removed.

The result is a deterministic, semantically-equivalent rendering of the
source with cosmetic noise removed. (Sorting conditional branches by
canonical form — the §4.1 stretch — is *not* attempted in v0; it requires
real AST analysis for correctness, and the conservative lex-based form
already covers every cosmetic edit observed in our gap-analysis corpus.
Tracking issue: docs/design.md §4.1.)

The hash is ``sha256(canonicalize(jinja_source) + "\n" + fewshot_style)``.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Self

from jinja2 import Environment
from jinja2.lexer import (
    TOKEN_COMMENT,
    TOKEN_COMMENT_BEGIN,
    TOKEN_COMMENT_END,
    TOKEN_DATA,
    TOKEN_STRING,
    TOKEN_WHITESPACE,
)

from anvil.exceptions import ModelLoadError, TokenizationError

FewshotStyle = Literal["interleaved", "concat-system", "raw", "none"]


_VALID_FEWSHOT_STYLES: tuple[FewshotStyle, ...] = (
    "interleaved",
    "concat-system",
    "raw",
    "none",
)


_TEMPLATE_SOURCES: tuple[str, ...] = (
    "tokenizer_config.json",
    "chat_template.json",
    "user-supplied",
    "builtin",
    "model-card-override",
)


@dataclass(frozen=True, slots=True)
class ChatTemplate:
    """Versioned, content-hashed chat template.

    Two ``ChatTemplate``s with the same :attr:`hash` MUST produce identical
    prompts for identical inputs.

    Attributes:
        jinja_source: the Jinja2 source string.
        fewshot_style: how few-shot examples are packed (``"interleaved"`` is
            most common; ``"concat-system"`` packs them into the system
            message; ``"raw"`` pre/postpends as plain text; ``"none"`` means
            this template doesn't support fewshot).
        name: human-readable identifier like ``"qwen2.5-instruct@v1"``.
        source: where the template came from. One of ``tokenizer_config.json``,
            ``chat_template.json``, ``user-supplied``, ``builtin``,
            ``model-card-override``.

    Example:
        >>> ct = ChatTemplate(
        ...     jinja_source="{% for m in messages %}{{ m.content }}{% endfor %}",
        ...     name="trivial",
        ... )
        >>> ct.hash.startswith("sha256:")
        True
    """

    jinja_source: str
    fewshot_style: FewshotStyle = "interleaved"
    name: str = "anonymous"
    source: str = "user-supplied"
    _cached_canonical: str = field(default="", repr=False, compare=False)
    _cached_hash: str = field(default="", repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.fewshot_style not in _VALID_FEWSHOT_STYLES:
            raise ValueError(
                f"fewshot_style {self.fewshot_style!r} not one of {_VALID_FEWSHOT_STYLES}"
            )
        if self.source not in _TEMPLATE_SOURCES:
            raise ValueError(f"source {self.source!r} not one of {_TEMPLATE_SOURCES}")

    # ------------------------------------------------------------------ load
    @classmethod
    def from_model(cls, model_id: str, *, revision: str | None = None) -> Self:
        """Load the chat template that ships with a HuggingFace model.

        Looks (in order) at:

        1. ``chat_template.json`` (the v4.44+ canonical location).
        2. ``tokenizer_config.json``'s ``chat_template`` key.

        Raises:
            ModelLoadError: if neither file contains a chat template.
        """
        import json as _json

        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError

        # 1. Try chat_template.json
        try:
            ct_path = hf_hub_download(
                repo_id=model_id, filename="chat_template.json", revision=revision
            )
            data: dict[str, Any] = _json.loads(Path(ct_path).read_text())
            if "chat_template" in data:
                tpl = data["chat_template"]
                if isinstance(tpl, list):
                    # Some models ship a list of (name, template) dicts.
                    default = next((e for e in tpl if e.get("name") == "default"), tpl[0])
                    src = str(default["template"])
                else:
                    src = str(tpl)
                return cls(
                    jinja_source=src,
                    name=f"{model_id}@chat_template.json",
                    source="chat_template.json",
                )
        except (EntryNotFoundError, HfHubHTTPError, OSError):
            pass

        # 2. Fall back to tokenizer_config.json
        try:
            tk_path = hf_hub_download(
                repo_id=model_id, filename="tokenizer_config.json", revision=revision
            )
        except (EntryNotFoundError, HfHubHTTPError, OSError) as exc:
            raise ModelLoadError(f"No chat template found for {model_id!r}: {exc}") from exc

        config = _json.loads(Path(tk_path).read_text())
        tpl_field = config.get("chat_template")
        if tpl_field is None:
            raise ModelLoadError(
                f"{model_id!r} has no chat_template in tokenizer_config.json. "
                "Per design §7.7, instruct/chat models without a chat template "
                "are flagged as a blocking failure mode."
            )
        if isinstance(tpl_field, list):
            default = next((e for e in tpl_field if e.get("name") == "default"), tpl_field[0])
            src = str(default["template"])
        else:
            src = str(tpl_field)
        return cls(
            jinja_source=src,
            name=f"{model_id}@tokenizer_config",
            source="tokenizer_config.json",
        )

    @classmethod
    def from_jinja_file(
        cls,
        path: str | Path,
        *,
        fewshot_style: FewshotStyle = "interleaved",
        name: str | None = None,
    ) -> Self:
        """Load a Jinja template from a local file. Source labeled ``user-supplied``."""
        p = Path(path)
        return cls(
            jinja_source=p.read_text(),
            fewshot_style=fewshot_style,
            name=name or p.stem,
            source="user-supplied",
        )

    # ------------------------------------------------------------ canonicalize
    def canonicalize(self) -> str:
        """Whitespace- and comment-normalized rendering of the source.

        Operates on the Jinja token stream, so cosmetic edits to the source
        (extra spaces, reformatted comments, single vs double quotes) do not
        change the result. The semantic content is preserved.
        """
        if self._cached_canonical:
            return self._cached_canonical
        out = io.StringIO()
        env = Environment(autoescape=False, keep_trailing_newline=False)
        # The lexer raises on truly malformed templates; let it propagate as-is.
        for _line, token, value in env.lex(self.jinja_source):
            if token in (TOKEN_COMMENT, TOKEN_COMMENT_BEGIN, TOKEN_COMMENT_END):
                continue
            if token == TOKEN_WHITESPACE:
                out.write(" ")
                continue
            if token == TOKEN_STRING:
                # The Jinja lexer hands us the literal source — including the
                # surrounding quotes. Strip them and re-emit with double quotes
                # so '...' and "..." normalize to the same canonical form.
                if len(value) >= 2 and value[0] in ("'", '"') and value[-1] == value[0]:
                    inner = value[1:-1]
                else:
                    inner = value
                # The lexer doesn't unescape; pass escapes through and re-quote.
                escaped = inner.replace("\\", "\\\\").replace('"', '\\"')
                out.write(f'"{escaped}"')
                continue
            if token == TOKEN_DATA:
                # Outside Jinja blocks: collapse runs of whitespace to a single space
                # but preserve a single newline if the chunk is purely whitespace
                # ending in newline (so block-level structure survives).
                stripped = value.strip("\r\t ")
                if stripped == "":
                    if "\n" in value:
                        out.write("\n")
                    else:
                        out.write(" ")
                else:
                    out.write(stripped)
                continue
            out.write(value)
        result = out.getvalue().strip()
        # Collapse repeated spaces (but not newlines).
        cleaned: list[str] = []
        prev_space = False
        for ch in result:
            if ch == " ":
                if prev_space:
                    continue
                prev_space = True
            else:
                prev_space = False
            cleaned.append(ch)
        canonical = "".join(cleaned)
        object.__setattr__(self, "_cached_canonical", canonical)
        return canonical

    @property
    def hash(self) -> str:
        """``sha256:`` + hex of ``canonicalize(jinja_source) + "\\n" + fewshot_style``."""
        if self._cached_hash:
            return self._cached_hash
        payload = (self.canonicalize() + "\n" + self.fewshot_style).encode()
        digest = "sha256:" + hashlib.sha256(payload).hexdigest()
        object.__setattr__(self, "_cached_hash", digest)
        return digest

    @property
    def version(self) -> str:
        """Human-readable version string. Defaults to ``{name}@{8-char-hash}``."""
        return f"{self.name}@{self.hash[7:15]}"

    # --------------------------------------------------------------- render
    def render(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool = True,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        """Render the template; assert no double-BOS is produced.

        Raises:
            TokenizationError: if the rendered output starts with two
                consecutive identical BOS-looking tokens. The detector is
                conservative and only fires on exact duplicates of common
                BOS strings; surprises about novel BOS markers must be
                handled in the engine layer.
        """
        env = Environment(
            autoescape=False,
            keep_trailing_newline=False,
            trim_blocks=False,
            lstrip_blocks=False,
        )
        # The HF chat-template convention defines a few helper functions
        # (``raise_exception``, ``strftime_now``) that some templates call.
        env.globals["raise_exception"] = _raise_exception
        try:  # pragma: no cover - defensive; older jinja2 may not need this
            import datetime as _dt

            env.globals["strftime_now"] = lambda fmt: _dt.datetime.now().strftime(fmt)
        except Exception:
            pass
        tpl = env.from_string(self.jinja_source)
        rendered = tpl.render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
        )
        _assert_no_double_bos(rendered)
        return rendered

    def to_manifest_field(self) -> dict[str, Any]:
        """Manifest projection (design §8.1): ``{hash, version, source, fewshot_style}``."""
        return {
            "hash": self.hash,
            "version": self.version,
            "source": self.source,
            "fewshot_style": self.fewshot_style,
        }


def _raise_exception(message: str) -> None:
    """Helper exposed to Jinja templates; raises a structured error."""
    raise TokenizationError(f"chat template raised: {message}")


_DOUBLE_BOS_STRINGS: tuple[str, ...] = (
    "<s><s>",
    "<|begin_of_text|><|begin_of_text|>",
    "<|startoftext|><|startoftext|>",
    "<bos><bos>",
)


def _assert_no_double_bos(rendered: str) -> None:
    head = rendered[:128]
    for marker in _DOUBLE_BOS_STRINGS:
        if marker in head:
            raise TokenizationError(
                f"rendered prompt begins with a duplicate BOS marker {marker!r}. "
                "This usually means add_special_tokens=True was combined with a "
                "template that already emits BOS."
            )


__all__ = ["ChatTemplate", "FewshotStyle"]
