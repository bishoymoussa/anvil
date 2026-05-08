"""Canonical JSON serialization (design §16.5).

The manifest is signed by computing sha256 of the *canonical* JSON.
Canonicalization rules (lifted verbatim from the manuscript):

1. UTF-8, no BOM.
2. Keys sorted lexicographically at every nesting level.
3. ``indent=2, separators=(",", ": "), ensure_ascii=False``.
4. The ``manifest_signature`` field is excluded from canonical form.
5. List ordering is preserved.
6. ``None`` → ``null``. Absent fields are simply not present.
7. Floats: ``repr()`` then strip trailing ``.0`` only for integer-valued
   floats. NaN and Inf are forbidden — raise.
8. Hash fields of nested objects are computed on construction and embedded —
   they are not recomputed during signing.

Two manifests with the same canonical JSON have the same signature,
byte-for-byte, on any machine.
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any

from anvil.exceptions import ManifestError

if TYPE_CHECKING:  # pragma: no cover
    from anvil.manifest.schema import Manifest


def _check_finite(node: Any, path: str = "") -> None:
    """Walk the structure refusing NaN and Inf (rule 7)."""
    if isinstance(node, float):
        if math.isnan(node) or math.isinf(node):
            raise ManifestError(
                f"non-finite float at {path or '<root>'}: {node!r}; "
                "manifests must not contain NaN or Inf (canonical JSON rule 7)"
            )
        return
    if isinstance(node, dict):
        for k, v in node.items():
            _check_finite(v, f"{path}.{k}" if path else str(k))
        return
    if isinstance(node, list):
        for i, v in enumerate(node):
            _check_finite(v, f"{path}[{i}]")
        return


def _sort_keys(node: Any) -> Any:
    """Return a structurally identical copy with all dict keys sorted (rule 2)."""
    if isinstance(node, dict):
        return {k: _sort_keys(node[k]) for k in sorted(node.keys())}
    if isinstance(node, list):
        return [_sort_keys(x) for x in node]
    return node


class _CanonicalEncoder(json.JSONEncoder):
    """Custom encoder that:

    * raises on NaN/Inf (we already check, but ``allow_nan=False`` belt-and-braces),
    * forbids float ``-0.0`` to avoid platform-specific signed-zero artifacts.
    """

    def __init__(self) -> None:
        super().__init__(
            ensure_ascii=False,
            sort_keys=False,  # we already sorted recursively
            indent=2,
            separators=(",", ": "),
            allow_nan=False,
        )

    def encode(self, o: Any) -> str:
        return super().encode(_normalize_negative_zero(o))


def _normalize_negative_zero(node: Any) -> Any:
    """Replace ``-0.0`` with ``0.0`` so signed zero is platform-independent."""
    if isinstance(node, float) and node == 0.0:
        return 0.0
    if isinstance(node, dict):
        return {k: _normalize_negative_zero(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_normalize_negative_zero(x) for x in node]
    return node


def canonical_json(manifest: Manifest) -> str:
    """Render ``manifest`` as canonical JSON suitable for signing.

    The ``manifest_signature`` field is dropped (rule 4) so that two manifests
    that differ only in their signature produce identical canonical bytes.
    """
    data = manifest.model_dump(mode="json", exclude_none=True)
    data.pop("manifest_signature", None)
    _check_finite(data)
    sorted_data = _sort_keys(data)
    return _CanonicalEncoder().encode(sorted_data)


def serialize_with_signature(manifest: Manifest) -> str:
    """Render ``manifest`` to disk-form JSON, *including* the signature.

    Same canonicalization rules (sorted keys, no NaN/Inf, deterministic
    floats) as :func:`canonical_json`, but the ``manifest_signature`` field
    is preserved in its alphabetically-sorted slot so :meth:`Manifest.verify`
    can recompute and check it after :func:`Manifest.load`.
    """
    data = manifest.model_dump(mode="json", exclude_none=True)
    _check_finite(data)
    sorted_data = _sort_keys(data)
    return _CanonicalEncoder().encode(sorted_data)


__all__ = ["canonical_json", "serialize_with_signature"]
