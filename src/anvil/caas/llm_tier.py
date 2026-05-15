"""CaaS LLM tier — fallback when the rule engine + KB cannot match (design §7.4).

When no KB entry matches a captured error, ``propose_fix`` sends the error
context to a small coder model (configurable, defaults to
``claude-haiku-4-5-20251001`` via the Anthropic API) and asks it to propose
one of the allowed action types. The response is parsed and returned as an
:class:`~anvil.caas.actions.Action`.

Safety constraints (design §7.4, §11.3):
- The LLM is **never** given write access to the filesystem. It returns a
  structured action object; ``apply_action`` enforces the allow-list.
- Requires the ``ANTHROPIC_API_KEY`` env var (or ``OPENAI_API_KEY`` +
  ``ANVIL_LLM_TIER_BASE_URL`` for any OpenAI-compatible endpoint).
- Disabled entirely when ``ANVIL_LLM_TIER_DISABLED=1`` is set.
- ``requires_user_consent`` is always ``True`` on LLM-proposed actions
  (ci mode will therefore refuse them unless the user pre-authorises).
"""

from __future__ import annotations

import json
import os
from typing import Any

from anvil.caas.actions import Action
from anvil.logging import get_logger

_log = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a diagnostic assistant for Anvil, an LLM evaluation library.
A run has failed with the error shown below. Your job is to propose exactly
ONE corrective action from the allowed action types.

Allowed action types and their required fields:
  set_engine_flag   — {"type": "set_engine_flag",   "flag": "<name>", "value": <json_value>,  "confidence": 0.0-1.0, "rationale": "..."}
  unset_engine_flag — {"type": "unset_engine_flag", "flag": "<name>",                          "confidence": 0.0-1.0, "rationale": "..."}
  set_sampling_param — {"type": "set_sampling_param", "name": "<name>", "value": <json_value>, "confidence": 0.0-1.0, "rationale": "..."}
  none              — {"type": "none", "confidence": 1.0, "rationale": "Cannot fix: <reason>"}

Rules:
- Return ONLY valid JSON matching one of the schemas above. No markdown fences.
- "confidence" must be a float in [0.0, 1.0].
- "rationale" must be a one-sentence human-readable explanation.
- If you cannot identify a safe fix, return the "none" action.
- Do NOT suggest installing packages or modifying files.
"""


def propose_fix(
    *,
    error: str,
    model_id: str | None = None,
    extra_context: dict[str, Any] | None = None,
    timeout: int = 30,
) -> Action | None:
    """Ask an LLM to propose a fix for an unmatched CaaS error.

    Args:
        error: the captured error string passed to the rule engine.
        model_id: override the LLM model (default: ``claude-haiku-4-5-20251001``
            via Anthropic, or the value of ``ANVIL_LLM_TIER_MODEL``).
        extra_context: optional dict of additional context (e.g. engine_args,
            model architecture) appended to the user message.
        timeout: HTTP timeout in seconds.

    Returns:
        An :class:`~anvil.caas.actions.Action` with ``requires_user_consent=True``,
        or ``None`` if the LLM tier is disabled or the call fails.
    """
    if os.environ.get("ANVIL_LLM_TIER_DISABLED", "0") == "1":
        _log.debug("CaaS LLM tier disabled via ANVIL_LLM_TIER_DISABLED=1")
        return None

    effective_model = model_id or os.environ.get(
        "ANVIL_LLM_TIER_MODEL", "claude-haiku-4-5-20251001"
    )

    ctx_lines = [f"Error:\n{error}"]
    if extra_context:
        ctx_lines.append("Context:\n" + json.dumps(extra_context, indent=2, default=str))
    user_message = "\n\n".join(ctx_lines)

    raw = _call_llm(
        system=_SYSTEM_PROMPT,
        user=user_message,
        model=effective_model,
        timeout=timeout,
    )
    if raw is None:
        return None

    return _parse_action(raw)


def _call_llm(*, system: str, user: str, model: str, timeout: int) -> str | None:
    """Call the LLM and return the raw text response, or None on failure."""
    base_url = os.environ.get("ANVIL_LLM_TIER_BASE_URL")

    # Anthropic SDK path.
    if not base_url and os.environ.get("ANTHROPIC_API_KEY"):
        return _call_anthropic(system=system, user=user, model=model, timeout=timeout)

    # OpenAI-compatible path (also works with local servers).
    if base_url or os.environ.get("OPENAI_API_KEY"):
        return _call_openai_compat(
            system=system, user=user, model=model, base_url=base_url, timeout=timeout
        )

    _log.warning(
        "CaaS LLM tier: no API key found (set ANTHROPIC_API_KEY or OPENAI_API_KEY). "
        "Skipping LLM fallback."
    )
    return None


def _call_anthropic(*, system: str, user: str, model: str, timeout: int) -> str | None:
    try:
        import anthropic  # type: ignore[import-not-found]
    except ImportError:
        _log.warning("CaaS LLM tier: 'anthropic' package not installed. pip install anthropic")
        return None

    try:
        client = anthropic.Anthropic(timeout=timeout)
        msg = client.messages.create(
            model=model,
            max_tokens=256,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return str(msg.content[0].text)
    except Exception as exc:  # noqa: BLE001
        _log.warning("CaaS LLM tier: Anthropic call failed: %s", exc)
        return None


def _call_openai_compat(
    *, system: str, user: str, model: str, base_url: str | None, timeout: int
) -> str | None:
    try:
        import openai  # type: ignore[import-not-found]
    except ImportError:
        _log.warning("CaaS LLM tier: 'openai' package not installed. pip install openai")
        return None

    try:
        kwargs: dict[str, Any] = {"timeout": timeout}
        if base_url:
            kwargs["base_url"] = base_url
        client = openai.OpenAI(**kwargs)
        resp = client.chat.completions.create(
            model=model,
            max_tokens=256,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""
    except Exception as exc:  # noqa: BLE001
        _log.warning("CaaS LLM tier: OpenAI call failed: %s", exc)
        return None


def _parse_action(raw: str) -> Action | None:
    """Parse the LLM's JSON response into an :class:`Action`."""
    text = raw.strip()
    # Strip markdown fences if the model added them despite instructions.
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        _log.warning("CaaS LLM tier: could not parse JSON response: %s\nRaw: %r", exc, raw)
        return None

    action_type = data.get("type")
    if action_type not in {"set_engine_flag", "unset_engine_flag", "set_sampling_param", "none"}:
        _log.warning("CaaS LLM tier: unknown action type %r", action_type)
        return None

    if action_type == "none":
        return None

    confidence = float(data.get("confidence", 0.5))
    return Action(
        type=action_type,
        kb_entry_id="llm_tier",
        flag=data.get("flag"),
        name=data.get("name"),
        value=data.get("value"),
        confidence=confidence,
        rationale=data.get("rationale", "LLM-proposed fix"),
        requires_user_consent=True,
    )


__all__ = ["propose_fix"]
