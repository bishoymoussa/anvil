"""``anvil doctor`` — environment diagnosis (design §10.3, §16.10 M6, §16.12).

Per the §16.12 agent-decisions list: rich-table by default, ``--json`` for
machine readers. Each check is a small dataclass-like record with a
``status`` (``ok | warn | fail``) and a one-line ``message``. The
overall exit code is non-zero if any ``fail`` check fires; ``warn`` is
informational.

The checks v0 ships are tilted at the failures the gap analysis cites:

* CUDA driver vs torch wheel mismatch (§1.2 / KB ``cuda_libcudart_version_mismatch``).
* HF_TOKEN presence (relevant when the user is about to run a gated
  model like Llama-3 — we don't validate the token by hitting the Hub
  unless ``--validate-hf`` is passed).
* vLLM optional-extra installed.
* Disk space at ``$HF_HOME``.
* GPU + driver + CUDA-runtime versions, when present.
* Python version vs the project's ``requires-python``.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from dataclasses import dataclass, field
from typing import Any, Literal

from anvil._version import __version__

CheckStatus = Literal["ok", "warn", "fail"]


@dataclass
class Check:
    """One environment-diagnosis result."""

    name: str
    status: CheckStatus
    message: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "detail": self.detail,
        }


# ------------------------------------------------------------- individual checks


def _check_python_version() -> Check:
    major, minor = sys.version_info[:2]
    msg = f"Python {major}.{minor}.{sys.version_info[2]}"
    if (major, minor) < (3, 11):
        return Check(
            name="python",
            status="fail",
            message=f"{msg} — Anvil requires Python ≥ 3.11",
            detail={"version": f"{major}.{minor}.{sys.version_info[2]}"},
        )
    return Check(name="python", status="ok", message=msg)


def _check_anvil_version() -> Check:
    return Check(
        name="anvil",
        status="ok",
        message=f"anvil {__version__}",
        detail={"version": __version__},
    )


def _check_torch() -> Check:
    try:
        import torch
    except ImportError:
        return Check(
            name="torch",
            status="fail",
            message="torch is not installed",
        )
    return Check(
        name="torch",
        status="ok",
        message=f"torch {torch.__version__}",
        detail={"version": torch.__version__, "cuda_built_with": torch.version.cuda},
    )


def _check_cuda() -> Check:
    """Detect a CUDA driver / torch-wheel mismatch (design §1.2)."""
    try:
        import torch
    except ImportError:
        return Check(name="cuda", status="warn", message="torch missing; cannot probe CUDA")
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            available = torch.cuda.is_available()
        except (RuntimeError, OSError) as exc:
            return Check(
                name="cuda",
                status="fail",
                message=f"CUDA probe failed: {exc}",
                detail={"torch_cuda": torch.version.cuda},
            )
    if not available:
        return Check(
            name="cuda",
            status="warn",
            message=(
                "CUDA not available (CPU-only mode). If you have a GPU, this "
                "usually means the torch wheel is built against a different "
                "CUDA major version than the host driver — see KB "
                "cuda_libcudart_version_mismatch."
            ),
            detail={"torch_cuda": torch.version.cuda},
        )
    devices = []
    try:
        for i in range(torch.cuda.device_count()):
            devices.append(torch.cuda.get_device_name(i))
    except (RuntimeError, OSError) as exc:  # pragma: no cover
        return Check(
            name="cuda",
            status="warn",
            message=f"CUDA available but device probe failed: {exc}",
            detail={"torch_cuda": torch.version.cuda},
        )
    return Check(
        name="cuda",
        status="ok",
        message=f"CUDA {torch.version.cuda} — {len(devices)} GPU(s): {', '.join(devices)}",
        detail={"torch_cuda": torch.version.cuda, "devices": devices},
    )


def _check_hf_token() -> Check:
    if os.environ.get("HF_TOKEN"):
        return Check(
            name="hf_token",
            status="ok",
            message="HF_TOKEN is set",
        )
    return Check(
        name="hf_token",
        status="warn",
        message=(
            "HF_TOKEN is not set; gated models (Llama-3, Mistral-Instruct, …) "
            "will fail to download. Set with `export HF_TOKEN=...`."
        ),
    )


def _check_vllm() -> Check:
    try:
        import vllm
    except ImportError:
        return Check(
            name="vllm",
            status="warn",
            message=(
                "vLLM is not installed; --engine=auto falls back to HF slow path. "
                'Install with `uv pip install -e ".[vllm]"` to enable.'
            ),
        )
    return Check(
        name="vllm",
        status="ok",
        message=f"vllm {getattr(vllm, '__version__', '?')}",
        detail={"version": getattr(vllm, "__version__", None)},
    )


def _check_transformers() -> Check:
    try:
        import transformers
    except ImportError:
        return Check(name="transformers", status="fail", message="transformers is not installed")
    return Check(
        name="transformers",
        status="ok",
        message=f"transformers {transformers.__version__}",
        detail={"version": transformers.__version__},
    )


def _check_hf_home_disk() -> Check:
    """Disk space at ``$HF_HOME`` (defaults to ``~/.cache/huggingface``)."""
    hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    try:
        usage = shutil.disk_usage(hf_home if os.path.exists(hf_home) else os.path.dirname(hf_home))
    except OSError as exc:
        return Check(
            name="hf_home_disk",
            status="warn",
            message=f"could not stat HF_HOME ({hf_home}): {exc}",
        )
    free_gb = usage.free / (1024**3)
    if free_gb < 5.0:
        return Check(
            name="hf_home_disk",
            status="fail",
            message=f"only {free_gb:.1f} GB free at {hf_home}; large models won't download",
            detail={"hf_home": hf_home, "free_gb": round(free_gb, 1)},
        )
    if free_gb < 50.0:
        return Check(
            name="hf_home_disk",
            status="warn",
            message=f"{free_gb:.1f} GB free at {hf_home} (may be tight for 70B models)",
            detail={"hf_home": hf_home, "free_gb": round(free_gb, 1)},
        )
    return Check(
        name="hf_home_disk",
        status="ok",
        message=f"{free_gb:.0f} GB free at {hf_home}",
        detail={"hf_home": hf_home, "free_gb": round(free_gb, 1)},
    )


def _check_kb_loads() -> Check:
    """Sanity: the shipped CaaS KB validates."""
    try:
        from anvil.caas import load_kb

        entries = load_kb()
    except Exception as exc:  # noqa: BLE001
        return Check(
            name="caas_kb",
            status="fail",
            message=f"KB load failed: {exc}",
        )
    if len(entries) < 15:
        return Check(
            name="caas_kb",
            status="warn",
            message=f"KB has {len(entries)} entries (<15 expected per §16.7)",
        )
    return Check(
        name="caas_kb",
        status="ok",
        message=f"CaaS KB loads {len(entries)} entries",
        detail={"entries": len(entries)},
    )


def _check_fast_paths() -> Check:
    """Confirm the §5.2 fast-path roster registered correctly."""
    from anvil.models.registry import _FAST_PATHS

    expected = {"LlamaForCausalLM", "Qwen2ForCausalLM", "MistralForCausalLM"}
    missing = expected - set(_FAST_PATHS)
    if missing:
        return Check(
            name="fast_paths",
            status="warn",
            message=f"fast-path families missing: {sorted(missing)}",
        )
    return Check(
        name="fast_paths",
        status="ok",
        message=f"{len(_FAST_PATHS)} fast-path families registered",
        detail={"families": sorted(_FAST_PATHS)},
    )


def _check_os() -> Check:
    return Check(
        name="os",
        status="ok",
        message=f"{platform.system()} {platform.release()} ({platform.machine()})",
        detail={
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
    )


# -------------------------------------------------------- top-level entrypoint

ALL_CHECKS: tuple[Any, ...] = (
    _check_anvil_version,
    _check_python_version,
    _check_os,
    _check_torch,
    _check_cuda,
    _check_transformers,
    _check_vllm,
    _check_hf_token,
    _check_hf_home_disk,
    _check_kb_loads,
    _check_fast_paths,
)


def run_all_checks() -> list[Check]:
    """Execute every diagnosis in :data:`ALL_CHECKS`. Fail-soft per check."""
    out: list[Check] = []
    for fn in ALL_CHECKS:
        try:
            out.append(fn())
        except Exception as exc:  # noqa: BLE001
            out.append(
                Check(
                    name=fn.__name__.removeprefix("_check_"),
                    status="fail",
                    message=f"check raised: {exc}",
                )
            )
    return out


def render_table(checks: list[Check]) -> str:
    """Format a list of checks as a rich-style table string.

    We deliberately emit plain text rather than depending on ``rich`` for
    the doctor output — the CLI command uses ``rich.Console`` for color
    when stdout is a TTY, but the helper here is testable without rich.
    """
    if not checks:
        return "(no checks)"
    name_w = max(len(c.name) for c in checks) + 2
    status_w = 6
    lines = [f"{'name':<{name_w}}{'status':<{status_w}}message"]
    lines.append("-" * (name_w + status_w + 60))
    for c in checks:
        lines.append(f"{c.name:<{name_w}}{c.status:<{status_w}}{c.message}")
    return "\n".join(lines)


def to_json(checks: list[Check]) -> str:
    """Machine-readable JSON output for ``--json``."""
    return json.dumps([c.to_dict() for c in checks], indent=2)


def overall_status(checks: list[Check]) -> CheckStatus:
    """The worst status across all checks."""
    if any(c.status == "fail" for c in checks):
        return "fail"
    if any(c.status == "warn" for c in checks):
        return "warn"
    return "ok"


__all__ = [
    "Check",
    "ALL_CHECKS",
    "run_all_checks",
    "render_table",
    "to_json",
    "overall_status",
]
