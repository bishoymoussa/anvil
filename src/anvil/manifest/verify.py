"""Manifest verification (design §8.3).

Verification is the inverse of :func:`anvil.manifest.sign.sign`: recompute
the canonical-JSON sha256 and check it matches the manifest's stored
``manifest_signature``. An unsigned manifest verifies to ``False``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from anvil.manifest.sign import compute_signature

if TYPE_CHECKING:
    from anvil.manifest.schema import Manifest


def verify(manifest: Manifest) -> bool:
    """Return True iff ``manifest.manifest_signature`` matches the recomputed signature.

    Returns False (rather than raising) for unsigned manifests so calling
    code can branch on the boolean. Use :func:`verify_or_raise` to fail loud.
    """
    if not manifest.manifest_signature:
        return False
    return compute_signature(manifest) == manifest.manifest_signature


def verify_or_raise(manifest: Manifest) -> None:
    """Verify the signature; raise :class:`ManifestError` if the check fails."""
    from anvil.exceptions import ManifestError

    if not manifest.manifest_signature:
        raise ManifestError(
            "manifest has no signature; cannot verify. Did you save() it without signing?"
        )
    expected = compute_signature(manifest)
    if expected != manifest.manifest_signature:
        raise ManifestError(
            f"manifest signature mismatch: stored {manifest.manifest_signature}, "
            f"recomputed {expected}. The manifest was tampered with or its canonical "
            "JSON serialization changed (e.g. anvil version bump that altered the schema)."
        )


__all__ = ["verify", "verify_or_raise"]
