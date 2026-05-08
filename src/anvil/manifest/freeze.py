"""Frozen-config rerun spec (design §8.3).

``anvil manifest strip-caas`` produces a copy of a manifest with the CaaS
log cleared, so the user can re-run with the *original* user-supplied
config (rejecting any auto-fixes that were applied during the original
run). Frozen mode (§8.2) is the default for benchmark uploads; this
helper is what lets a reviewer reject an auto-fixed run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anvil.manifest.schema import Manifest


def strip_caas(manifest: Manifest) -> Manifest:
    """Return a copy of ``manifest`` with the CaaS log cleared.

    The returned manifest is **unsigned** — the caller should re-run and
    sign the new result, not rebadge the stripped manifest as authoritative.
    """
    return manifest.model_copy(
        update={
            "caas_log": [],
            "manifest_signature": "",
        }
    )


__all__ = ["strip_caas"]
