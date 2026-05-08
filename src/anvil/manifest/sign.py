"""Manifest signing (design §8.3).

Pure functions that compute and apply the sha256 signature over a manifest's
canonical JSON. Signing **excludes** the ``manifest_signature`` field from
the bytes that are hashed (canonical-JSON rule 4) so signing an
already-signed manifest is idempotent.

The :class:`anvil.manifest.schema.Manifest` exposes ``.sign()`` as a thin
facade over :func:`sign`.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anvil.manifest.schema import Manifest


_SIG_PREFIX = "sha256:"


def compute_signature(manifest: Manifest) -> str:
    """Return the sha256 signature of ``manifest``'s canonical JSON.

    The returned string includes the ``sha256:`` prefix.
    """
    encoded = manifest.canonical_json().encode("utf-8")
    return _SIG_PREFIX + hashlib.sha256(encoded).hexdigest()


def sign(manifest: Manifest) -> Manifest:
    """Return a copy of ``manifest`` with ``manifest_signature`` set.

    Idempotent: signing an already-signed manifest produces the same
    signature, because :func:`compute_signature` operates on the canonical
    bytes (which exclude the signature field).
    """
    sig = compute_signature(manifest)
    return manifest.model_copy(update={"manifest_signature": sig})


__all__ = ["sign", "compute_signature"]
