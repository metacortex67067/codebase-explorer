"""
Module summarization via LLM.

Sends only a compact description (docstring + signatures), not full bodies, to
save tokens and keep the signal high. Malformed JSON from the model degrades
gracefully to using the raw text as the summary rather than crashing.
"""
from __future__ import annotations

import json
import time
from typing import Optional

from src.core.models import Module, ModuleSummary
from src.llm.client import LLMClient, LLMConfigError, LLMError, default_llm_client
from src.llm.prompts import (
    SUMMARIZE_MODULE_SYSTEM,
    SUMMARIZE_MODULE_USER_TEMPLATE,
)

# Indexing summarises every module with one LLM call each, fired back-to-back.
# Free cloud tiers (e.g. Groq) rate-limit such bursts, which would otherwise
# turn into "(summary unavailable)" placeholders at random. We retry only the
# transient cases (rate limit / 429 / timeout) with exponential backoff;
# real misconfiguration (LLMConfigError) is re-raised immediately so we don't
# stall when there is genuinely no working backend.
_MAX_ATTEMPTS = 4
_BASE_DELAY_SECONDS = 2.0
_TRANSIENT_MARKERS = (
    "rate limit", "rate_limit", "ratelimit", "429",
    "too many requests", "timeout", "timed out", "overloaded",
    "temporarily", "503", "502",
)


def _is_transient(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(marker in msg for marker in _TRANSIENT_MARKERS)


def _complete_with_retry(client: LLMClient, system: str, user: str) -> str:
    """Call the LLM, retrying transient failures with exponential backoff."""
    last_exc: Optional[Exception] = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            return client.complete(system=system, user=user)
        except LLMConfigError:
            raise  # not transient -- no key / unknown provider
        except LLMError as exc:
            last_exc = exc
            if attempt == _MAX_ATTEMPTS - 1 or not _is_transient(exc):
                raise
            time.sleep(_BASE_DELAY_SECONDS * (2 ** attempt))
    assert last_exc is not None  # unreachable, but keeps type-checkers happy
    raise last_exc


def summarize_module(
    module: Module,
    llm_client: Optional[LLMClient] = None,
) -> ModuleSummary:
    """Return a ModuleSummary for the given Module."""
    client = llm_client or default_llm_client

    user_message = SUMMARIZE_MODULE_USER_TEMPLATE.format(
        qualified_name=module.qualified_name,
        docstring=module.docstring or "(no module docstring)",
        functions=_format_functions(module),
        classes=_format_classes(module),
    )

    raw = _complete_with_retry(
        client,
        system=SUMMARIZE_MODULE_SYSTEM,
        user=user_message,
    )

    summary, responsibilities = _parse_summary_json(raw)
    return ModuleSummary(
        module_path=module.relative_path,
        qualified_name=module.qualified_name,
        summary=summary,
        key_responsibilities=responsibilities,
    )


def _format_functions(module: Module) -> str:
    if not module.functions:
        return "(none)"
    lines = []
    for fn in module.functions:
        prefix = "async def" if fn.is_async else "def"
        sig = f"{prefix} {fn.name}({', '.join(fn.parameters)})"
        doc = f" -- {fn.docstring}" if fn.docstring else ""
        lines.append(f"- {sig}{doc}")
    return "\n".join(lines)


def _format_classes(module: Module) -> str:
    if not module.classes:
        return "(none)"
    lines = []
    for cls in module.classes:
        base_str = f"({', '.join(cls.bases)})" if cls.bases else ""
        header = f"- class {cls.name}{base_str}"
        if cls.docstring:
            header += f" -- {cls.docstring}"
        lines.append(header)
        for m in cls.methods:
            prefix = "async def" if m.is_async else "def"
            lines.append(f"    {prefix} {m.name}({', '.join(m.parameters)})")
    return "\n".join(lines)


def _parse_summary_json(raw: str) -> tuple[str, list[str]]:
    """Parse the LLM response. Falls back gracefully on malformed JSON."""
    raw = raw.strip()
    # Models sometimes wrap JSON in ```json fences despite instructions;
    # strip a leading/trailing fence if present.
    if raw.startswith("```"):
        raw = raw.strip("`")
        # After stripping backticks, the language tag (e.g. "json\n") may
        # remain at the start.
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        # Last-resort fallback: keep whatever text we got as the summary.
        return raw or "(empty response)", []

    summary = str(obj.get("summary", "")).strip() or "(empty summary)"
    responsibilities_raw = obj.get("key_responsibilities") or []
    if not isinstance(responsibilities_raw, list):
        responsibilities = []
    else:
        responsibilities = [str(x) for x in responsibilities_raw if x]
    return summary, responsibilities
