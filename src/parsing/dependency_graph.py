"""
Dependency graph builder.

Maps each module's qualified name to the internal modules it imports (imports
that resolve to another module in the same repo; stdlib/pypi are dropped).
Returned as plain dicts/lists so it serialises cleanly to JSON.
"""
from __future__ import annotations

from src.core.models import Module


def build_dependency_graph(modules: list[Module]) -> dict[str, list[str]]:
    """Return adjacency list: module qname -> internal imports it has."""
    known: set[str] = {m.qualified_name for m in modules}
    graph: dict[str, list[str]] = {}
    for m in modules:
        internal_imports: set[str] = set()
        for imp in m.imports:
            resolved = _resolve_internal(imp, known)
            if resolved is not None and resolved != m.qualified_name:
                internal_imports.add(resolved)
        graph[m.qualified_name] = sorted(internal_imports)
    return graph


def _resolve_internal(imported: str, known: set[str]) -> str | None:
    """Return the longest known module name that matches `imported` or its prefix.

    Handles cases like `from pkg.sub.mod import Thing` where `pkg.sub.mod`
    is itself a module in the repo.
    """
    if imported in known:
        return imported
    # `from pkg import sub` may name a subpackage, walk up
    parts = imported.split(".")
    while parts:
        candidate = ".".join(parts)
        if candidate in known:
            return candidate
        parts.pop()
    return None
