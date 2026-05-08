"""``Sampler`` — content-hashed sampling parameters (design §4.3).

Three constructors with explicit reproducibility semantics:

* :meth:`Sampler.greedy` — neutral default, **independent of**
  ``generation_config.json``. This is the opposite of vLLM v0.8.0+'s default.
* :meth:`Sampler.from_generation_config` — explicit opt-in to the model's
  on-disk defaults. ALWAYS records ``source='generation_config'`` in the
  manifest, so a reviewer can see at a glance that the run depended on
  whatever was in that file.
* Direct constructor — fully explicit; ``source='explicit'``.

The hash covers every field that affects output. ``Sampler.diff`` is the
one-liner that explains why two runs produced different numbers.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal, Self

SamplerSource = Literal["explicit", "generation_config", "greedy", "reasoning_default"]


@dataclass(frozen=True, slots=True)
class Sampler:
    """Sampling parameters with a content hash.

    Defaults are intentionally neutral: temperature=0, no penalties, no
    truncation. ``generation_config.json`` is *not* read unless you call
    :meth:`from_generation_config` explicitly. See design §1.2 and §4.3.

    Examples:
        >>> Sampler.greedy().is_argmax_invariant()
        True
        >>> Sampler.greedy(max_tokens=512).max_tokens
        512
        >>> a = Sampler.greedy(); b = Sampler(temperature=0.7, seed=1)
        >>> "temperature" in a.diff(b)
        True
    """

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    max_tokens: int = 2048
    seed: int | None = None
    stop: tuple[str, ...] = ()
    stop_token_ids: tuple[int, ...] = ()
    n: int = 1
    source: SamplerSource = "explicit"
    _cached_hash: str = field(default="", repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError(f"temperature must be ≥ 0, got {self.temperature}")
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError(f"top_p must be in [0, 1], got {self.top_p}")
        if self.top_k != -1 and self.top_k < 1:
            raise ValueError(f"top_k must be -1 (disabled) or ≥ 1, got {self.top_k}")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be ≥ 1, got {self.max_tokens}")
        if self.n < 1:
            raise ValueError(f"n must be ≥ 1, got {self.n}")

    @classmethod
    def greedy(cls, *, max_tokens: int = 2048, **overrides: Any) -> Self:
        """Neutral, deterministic default. ``source='greedy'``.

        Independent of ``generation_config.json`` — the sample-by-sample numbers
        you publish depend only on the model weights and the prompt.
        """
        kwargs: dict[str, Any] = dict(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            min_p=0.0,
            repetition_penalty=1.0,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            max_tokens=max_tokens,
            seed=None,
            n=1,
            source="greedy",
        )
        kwargs.update(overrides)
        kwargs["source"] = "greedy"
        return cls(**kwargs)

    @classmethod
    def from_generation_config(cls, model_id: str, *, revision: str | None = None) -> Self:
        """Read ``generation_config.json`` from the model and apply its defaults.

        Always records ``source='generation_config'`` in the manifest. This is
        the only way to get the model's on-disk sampling defaults; Anvil never
        applies them implicitly.

        Raises:
            ModelLoadError: if the config can't be fetched or parsed.
        """
        # Imported lazily to keep the primitives leaf-clean for static analysis;
        # the dependency is on huggingface_hub at the function level only.
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError

        from anvil.exceptions import ModelLoadError

        try:
            path = hf_hub_download(
                repo_id=model_id,
                filename="generation_config.json",
                revision=revision,
            )
        except (EntryNotFoundError, HfHubHTTPError, OSError) as exc:
            raise ModelLoadError(
                f"Could not fetch generation_config.json for {model_id!r}: {exc}"
            ) from exc
        with open(path) as fh:
            cfg: dict[str, Any] = json.load(fh)

        kwargs: dict[str, Any] = {
            "temperature": float(cfg.get("temperature", 0.0)),
            "top_p": float(cfg.get("top_p", 1.0)),
            "top_k": int(cfg.get("top_k", -1)),
            "repetition_penalty": float(cfg.get("repetition_penalty", 1.0)),
            "max_tokens": int(cfg.get("max_new_tokens", cfg.get("max_length", 2048))),
            "source": "generation_config",
        }
        # generation_config sometimes encodes top_k=0 to mean "off"; normalize.
        if kwargs["top_k"] in (0, None):
            kwargs["top_k"] = -1
        return cls(**kwargs)

    @classmethod
    def for_reasoning_model(
        cls, model_id: str, *, max_tokens: int = 32768, **overrides: Any
    ) -> Self:
        """Greedy defaults with a long generation budget for ``<think>`` blocks.

        The ``model_id`` is recorded for provenance only — the sampling itself
        is identical to :meth:`greedy` with a larger ``max_tokens``.
        """
        del model_id  # unused; kept in the signature for future per-family tuning
        kwargs: dict[str, Any] = dict(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=max_tokens,
            source="reasoning_default",
        )
        kwargs.update(overrides)
        kwargs["source"] = "reasoning_default"
        return cls(**kwargs)

    def _hash_payload(self) -> dict[str, Any]:
        """Fields that affect output, normalized for stable hashing."""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "repetition_penalty": self.repetition_penalty,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "stop": list(self.stop),
            "stop_token_ids": list(self.stop_token_ids),
            "n": self.n,
        }

    @property
    def hash(self) -> str:
        """``sha256:`` prefix + hex digest of the canonical hash payload."""
        if self._cached_hash:
            return self._cached_hash
        encoded = json.dumps(self._hash_payload(), sort_keys=True, separators=(",", ":")).encode()
        digest = "sha256:" + hashlib.sha256(encoded).hexdigest()
        # Cache on the frozen dataclass via object.__setattr__.
        object.__setattr__(self, "_cached_hash", digest)
        return digest

    def diff(self, other: Sampler) -> dict[str, tuple[Any, Any]]:
        """Fields where ``self != other``, as ``{field: (self, other)}``.

        ``source`` is a label, not an output-affecting field, so it is excluded
        from the diff. If you want to see source differences, compare manifests.
        """
        a, b = asdict(self), asdict(other)
        out: dict[str, tuple[Any, Any]] = {}
        for k in a.keys() | b.keys():
            if k in {"source", "_cached_hash"}:
                continue
            if a.get(k) != b.get(k):
                out[k] = (a.get(k), b.get(k))
        return out

    def is_argmax_invariant(self) -> bool:
        """True if this sampler picks the argmax token unconditionally.

        Used by the engine to short-circuit logits-processor batching when
        every active processor is argmax-invariant too.
        """
        return (
            self.temperature == 0.0
            and self.top_k in (-1, 1)
            and self.repetition_penalty == 1.0
            and self.presence_penalty == 0.0
            and self.frequency_penalty == 0.0
        )

    def with_overrides(self, **overrides: Any) -> Self:
        """Return a copy with select fields replaced; recomputes hash on access."""
        return replace(self, _cached_hash="", **overrides)

    def to_manifest_field(self) -> dict[str, Any]:
        """Manifest-shaped projection (design §8.1)."""
        return {
            "hash": self.hash,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "repetition_penalty": self.repetition_penalty,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "stop": list(self.stop),
            "stop_token_ids": list(self.stop_token_ids),
            "n": self.n,
            "source": self.source,
        }


__all__ = ["Sampler", "SamplerSource"]
