"""``HiddenStateSpec`` — declarative request for activation capture (design §4.5).

Implementing this in user space is the answer to vLLM RFC #33118. The spec
is a *plan*, not a tensor — it tells the engine which layers and positions
to copy back to host memory async.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

PositionLiteral = Literal["all", "last", "first", "image_tokens", "text_tokens"]
PositionSpec = PositionLiteral | tuple[int, ...]


_DTYPE_BYTES: dict[torch.dtype, int] = {
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.float64: 8,
    torch.int32: 4,
    torch.int64: 8,
    torch.int16: 2,
    torch.int8: 1,
}


@dataclass(frozen=True, slots=True)
class HiddenStateSpec:
    """A plan to capture per-layer activations during a forward pass.

    Attributes:
        layers: layer indices to capture. Negative indices supported (``-1`` is
            the last layer). Order is preserved in the output.
        positions: which token positions to keep. Either a literal
            (``"all"``, ``"last"``, ``"first"``, ``"image_tokens"``,
            ``"text_tokens"``) or an explicit tuple of indices.
        dtype: cast on copy. ``None`` keeps the model's native dtype.
        pin_memory: copy to pinned host memory for faster device→host transfer.

    Example:
        >>> spec = HiddenStateSpec(layers=(-1,), positions="last")
        >>> spec.estimate_bytes(seq_len=128, hidden_dim=4096) > 0
        True
    """

    layers: tuple[int, ...]
    positions: PositionSpec = "last"
    dtype: torch.dtype | None = None
    pin_memory: bool = True
    _cached_hash: str = field(default="", repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("HiddenStateSpec.layers must be non-empty")
        if isinstance(self.positions, tuple) and not all(
            isinstance(p, int) for p in self.positions
        ):
            raise TypeError("positions tuple must contain int indices")

    def _positions_count(self, seq_len: int) -> int:
        """How many positions per layer the engine will keep, for ``estimate_bytes``."""
        if isinstance(self.positions, tuple):
            return len(self.positions)
        if self.positions in ("last", "first"):
            return 1
        if self.positions == "all":
            return seq_len
        if self.positions in ("image_tokens", "text_tokens"):
            # Best-effort estimate; real count depends on the input.
            return seq_len
        raise ValueError(f"unknown positions spec: {self.positions!r}")

    def estimate_bytes(self, *, seq_len: int, hidden_dim: int) -> int:
        """Approximate VRAM cost of this capture for a single sample.

        The engine refuses captures whose total estimate exceeds a configured
        budget rather than letting them OOM mid-batch.
        """
        per_value = _DTYPE_BYTES.get(self.dtype or torch.float32, 4)
        return len(self.layers) * self._positions_count(seq_len) * hidden_dim * per_value

    @property
    def hash(self) -> str:
        if self._cached_hash:
            return self._cached_hash
        positions_repr: Any
        if isinstance(self.positions, tuple):
            positions_repr = list(self.positions)
        else:
            positions_repr = self.positions
        payload = {
            "layers": list(self.layers),
            "positions": positions_repr,
            "dtype": str(self.dtype) if self.dtype is not None else None,
            "pin_memory": self.pin_memory,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        digest = "sha256:" + hashlib.sha256(encoded).hexdigest()
        object.__setattr__(self, "_cached_hash", digest)
        return digest

    def to_manifest_field(self) -> dict[str, Any]:
        """Manifest-shaped projection."""
        return {
            "hash": self.hash,
            "layers": list(self.layers),
            "positions": (
                list(self.positions) if isinstance(self.positions, tuple) else self.positions
            ),
            "dtype": str(self.dtype) if self.dtype is not None else None,
            "pin_memory": self.pin_memory,
        }


__all__ = ["HiddenStateSpec", "PositionSpec", "PositionLiteral"]
