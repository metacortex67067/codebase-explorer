"""
Module summarization via LLM.

We don't send the full source -- only a compact description (docstring +
signatures). Reasons:
  * Token budget. A large module with full bodies costs many times more
    tokens; we don't need the bodies to write a high-level summary.
  * Signal-to-noise. The LLM should see "what this module exposes" without
    being distracted by implementation details.

Graceful degradation:
  * If the LLM returns malformed JSON we fall back to treating the whole
    text as the summary and an empty responsibilities list. The
    application never crashes because of bad model output -- this matters
    for the demo, where the student should not have to debug live.
"""
from __future__ import annotations

import json
from typing import Optional

from src.core.models import Module, ModuleSummary
from src.llm.client import LLMClient, default_llm_client
from src.llm.prompts import (
    SUMMARIZE_MODULE_SYSTEM,
    SUMMARIZE_MODULE_USER_TEMPLATE,
)


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

    raw = client.complete(
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


# ----- internals ------------------------------------------------------------

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
