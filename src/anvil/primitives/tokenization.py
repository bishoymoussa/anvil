"""``Tokenization`` — tokenizer plus invariants on its use (design §4.2).

The lm-evaluation-harness reproducibility crisis is partly a tokenizer crisis:
double-BOS, wrong padding side, EOS-not-stopped. ``Tokenization`` makes those
invariants explicit at construction time and asserts them when used.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Self

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from anvil.exceptions import ModelLoadError, TokenizationError

if TYPE_CHECKING:
    from .chat_template import ChatTemplate

BOSHandling = Literal["auto-prepend", "never", "from-template"]
PaddingSide = Literal["left", "right"]


@dataclass(frozen=True, slots=True)
class Tokenization:
    """A tokenizer plus assertions about how it is used.

    Attributes:
        tokenizer: a HuggingFace ``PreTrainedTokenizerBase``.
        bos_handling: ``"auto-prepend"`` (engine adds BOS), ``"never"`` (no BOS
            ever — for chat templates that emit it themselves), or
            ``"from-template"`` (defer to whatever the chat template does).
        eos_token_ids: ALL plausible terminator ids. Llama-3-Instruct has
            multiple (``128001`` and ``128009``); the engine treats any of
            them as a stop. (See KB entry ``llama3_eot_runaway``.)
        padding_side: ``"left"`` is required for batched generation in
            decoder-only models.
        add_special_tokens: ``False`` is the right default when a chat
            template is being applied (the template emits the special tokens).
        assert_no_double_bos: if True, ``encode(...)`` raises if the result
            begins with two BOS tokens.
    """

    tokenizer: PreTrainedTokenizerBase
    bos_handling: BOSHandling = "from-template"
    eos_token_ids: tuple[int, ...] = ()
    padding_side: PaddingSide = "left"
    add_special_tokens: bool = False
    assert_no_double_bos: bool = True
    _cached_hash: str = field(default="", repr=False, compare=False)

    @classmethod
    def from_model(cls, model_id: str, *, revision: str | None = None) -> Self:
        """Construct from a HF model id, inferring sensible invariants.

        EOS detection is conservative: every id that the tokenizer or its
        config marks as a possible terminator is included. Padding side is
        always set to ``"left"`` (the only correct choice for batched decoder
        generation).
        """
        try:
            tk = AutoTokenizer.from_pretrained(model_id, revision=revision)  # type: ignore[no-untyped-call]
        except Exception as exc:  # noqa: BLE001 - re-raised as ModelLoadError
            raise ModelLoadError(
                f"AutoTokenizer.from_pretrained({model_id!r}) failed: {exc}"
            ) from exc
        eos_ids = _collect_eos_ids(tk)
        return cls(
            tokenizer=tk,
            bos_handling="from-template",
            eos_token_ids=eos_ids,
            padding_side="left",
            add_special_tokens=False,
            assert_no_double_bos=True,
        )

    @property
    def hash(self) -> str:
        """Hash of the tokenizer's vocab + invariant fields.

        We hash the sorted vocab representation rather than the full state
        because ``PreTrainedTokenizerBase`` doesn't have a stable serialization
        across transformers versions. This is good enough for the manifest's
        purpose: detect drift, not authenticate weights.
        """
        if self._cached_hash:
            return self._cached_hash
        h = hashlib.sha256()
        # Vocab fingerprint — first 1k tokens by id, plus size.
        try:
            vocab = self.tokenizer.get_vocab()
            h.update(str(len(vocab)).encode())
            sample = sorted(vocab.items(), key=lambda kv: kv[1])[:1000]
            for tok, idx in sample:
                h.update(f"{idx}:{tok}\n".encode())
        except Exception:  # noqa: BLE001
            h.update(self.tokenizer.__class__.__name__.encode())
        h.update(b"\n")
        h.update(self.bos_handling.encode())
        h.update(b"\n")
        h.update(",".join(str(i) for i in sorted(self.eos_token_ids)).encode())
        h.update(b"\n")
        h.update(self.padding_side.encode())
        h.update(b"\n")
        h.update(b"1" if self.add_special_tokens else b"0")
        digest = "sha256:" + h.hexdigest()
        object.__setattr__(self, "_cached_hash", digest)
        return digest

    def encode(
        self,
        text: str,
        *,
        with_chat_template: ChatTemplate | None = None,
    ) -> list[int]:
        """Encode text or a chat-templated prompt to token ids.

        If ``with_chat_template`` is supplied the text is treated as already
        rendered by that template; we skip ``add_special_tokens`` to honor
        the template's own BOS handling, and assert no double-BOS.

        Raises:
            TokenizationError: if a double-BOS is detected and
                :attr:`assert_no_double_bos` is True.
        """
        add_specials = self.add_special_tokens
        if with_chat_template is not None:
            add_specials = False
        ids = self.tokenizer.encode(text, add_special_tokens=add_specials)
        if self.assert_no_double_bos:
            self._check_no_double_bos(ids)
        # The transformers .encode return type is List[int]; coerce defensively.
        return list(ids)

    def encode_messages(
        self,
        messages: list[dict[str, Any]],
        chat_template: ChatTemplate,
        *,
        add_generation_prompt: bool = True,
    ) -> tuple[list[int], str]:
        """Render messages through a chat template and encode the result.

        Returns:
            ``(token_ids, rendered_prompt)``.
        """
        rendered = chat_template.render(messages, add_generation_prompt=add_generation_prompt)
        ids = self.encode(rendered, with_chat_template=chat_template)
        return ids, rendered

    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str:
        """Decode token ids to text."""
        return str(self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens))

    def _check_no_double_bos(self, ids: list[int]) -> None:
        bos = self.tokenizer.bos_token_id
        if bos is None or len(ids) < 2:
            return
        if ids[0] == bos and ids[1] == bos:
            raise TokenizationError(
                f"encoded prompt begins with two BOS tokens (id={bos}). "
                "This is the classic 'add_special_tokens=True + template emits BOS' "
                "footgun. Either pass with_chat_template=... or set "
                "add_special_tokens=False."
            )

    def to_manifest_field(self) -> dict[str, Any]:
        """Manifest projection (design §8.1)."""
        return {
            "hash": self.hash,
            "bos_handling": self.bos_handling,
            "eos_token_ids": list(self.eos_token_ids),
            "padding_side": self.padding_side,
            "add_special_tokens": self.add_special_tokens,
        }


def _collect_eos_ids(tk: PreTrainedTokenizerBase) -> tuple[int, ...]:
    """Return every plausible terminator id for ``tk``.

    Includes the canonical ``eos_token_id`` plus any ids attached to common
    terminator strings like ``<|eot_id|>`` (Llama-3) and ``<|im_end|>`` (Qwen).
    Order is deterministic; duplicates removed.
    """
    candidates: list[int] = []
    if tk.eos_token_id is not None:
        candidates.append(int(tk.eos_token_id))
    extra_terminators = (
        "<|eot_id|>",
        "<|im_end|>",
        "<|end|>",
        "<end_of_turn>",
    )
    for marker in extra_terminators:
        token_id = tk.convert_tokens_to_ids(marker)
        # convert_tokens_to_ids returns the unk id if marker is unknown; filter.
        if isinstance(token_id, int) and token_id != tk.unk_token_id and token_id not in candidates:
            candidates.append(token_id)
    return tuple(candidates)


__all__ = ["Tokenization", "BOSHandling", "PaddingSide"]
