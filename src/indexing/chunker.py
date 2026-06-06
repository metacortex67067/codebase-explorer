"""
Semantic chunking of parsed modules into ``CodeChunk`` objects.

Chunks are whole semantic units (not fixed-size windows) so each embeds
coherently and yields clean citation boundaries. Per module we emit:
``module_header`` (docstring + signatures, for whole-module questions),
one ``function`` per top-level function, one ``class`` per class, and one
``class_method`` per method. The class/method overlap is deliberate -- vector
search ranks the right granularity per question. Chunk ids are deterministic
from ``(repo_id, qualified_name, chunk_type)`` so re-indexing overwrites cleanly.
"""
from __future__ import annotations

from src.core.models import ClassInfo, CodeChunk, FunctionInfo, Module


def chunk_module(module: Module, source: str, repo_id: str) -> list[CodeChunk]:
    """Split one parsed module into a list of CodeChunks.

    `source` is the original file text. We slice it by AST line numbers to
    get verbatim chunks (preserving formatting, comments, decorators).
    """
    lines = source.splitlines()
    chunks: list[CodeChunk] = []

    # 1. Module header chunk -- always emit, even for empty modules, so that
    # queries about module purpose have something to match against.
    chunks.append(_module_header_chunk(module, repo_id))

    # 2. Top-level functions
    for fn in module.functions:
        chunks.append(_function_chunk(fn, module, lines, repo_id))

    # 3. Classes (whole) + each method (separate chunk)
    for cls in module.classes:
        chunks.append(_class_chunk(cls, module, lines, repo_id))
        for method in cls.methods:
            chunks.append(_method_chunk(method, module, lines, repo_id))

    return chunks


def chunk_modules(
    modules_with_source: list[tuple[Module, str]],
    repo_id: str,
) -> list[CodeChunk]:
    """Convenience wrapper: chunk many modules in one call."""
    out: list[CodeChunk] = []
    for module, source in modules_with_source:
        out.extend(chunk_module(module, source, repo_id))
    return out


def _slice_lines(lines: list[str], line_start: int, line_end: int) -> str:
    """Return the slice of `lines` (1-based, inclusive) joined with '\\n'.

    `ast` line numbers are 1-based and `end_lineno` is inclusive, so we map
    that to a 0-based half-open Python slice [start-1, end].
    """
    # Defensive clamp: malformed AST or missing end_lineno could produce
    # out-of-range indices. We clamp instead of raising so one weird module
    # cannot break indexing.
    start = max(0, line_start - 1)
    end = min(len(lines), line_end)
    return "\n".join(lines[start:end])


def _module_header_chunk(module: Module, repo_id: str) -> CodeChunk:
    """Synthesise a compact 'what this module contains' chunk.

    We include the docstring and the *signatures* (not full bodies) of
    top-level functions and classes. This gives the embedder a high-signal
    summary of the module without duplicating every chunk's content.
    """
    parts: list[str] = [f"Module: {module.qualified_name}"]
    if module.docstring:
        parts.append(f'"""{module.docstring}"""')

    if module.functions:
        parts.append("Functions:")
        for fn in module.functions:
            parts.append(f"  {_function_signature(fn)}")

    if module.classes:
        parts.append("Classes:")
        for cls in module.classes:
            base_str = f"({', '.join(cls.bases)})" if cls.bases else ""
            parts.append(f"  class {cls.name}{base_str}")
            for m in cls.methods:
                parts.append(f"    {_function_signature(m)}")

    content = "\n".join(parts)

    return CodeChunk(
        chunk_id=_chunk_id(repo_id, module.qualified_name, "module_header"),
        repo_id=repo_id,
        module_path=module.relative_path,
        qualified_name=module.qualified_name,
        chunk_type="module_header",
        content=content,
        # Line range covers the whole module; useful when citing "this whole file"
        line_start=1,
        line_end=max(1, module.line_count),
    )


def _function_chunk(
    fn: FunctionInfo, module: Module, lines: list[str], repo_id: str
) -> CodeChunk:
    return CodeChunk(
        chunk_id=_chunk_id(repo_id, fn.qualified_name, "function"),
        repo_id=repo_id,
        module_path=module.relative_path,
        qualified_name=fn.qualified_name,
        chunk_type="function",
        content=_slice_lines(lines, fn.line_start, fn.line_end),
        line_start=fn.line_start,
        line_end=fn.line_end,
    )


def _class_chunk(
    cls: ClassInfo, module: Module, lines: list[str], repo_id: str
) -> CodeChunk:
    return CodeChunk(
        chunk_id=_chunk_id(repo_id, cls.qualified_name, "class"),
        repo_id=repo_id,
        module_path=module.relative_path,
        qualified_name=cls.qualified_name,
        chunk_type="class",
        content=_slice_lines(lines, cls.line_start, cls.line_end),
        line_start=cls.line_start,
        line_end=cls.line_end,
    )


def _method_chunk(
    method: FunctionInfo, module: Module, lines: list[str], repo_id: str
) -> CodeChunk:
    return CodeChunk(
        chunk_id=_chunk_id(repo_id, method.qualified_name, "class_method"),
        repo_id=repo_id,
        module_path=module.relative_path,
        qualified_name=method.qualified_name,
        chunk_type="class_method",
        content=_slice_lines(lines, method.line_start, method.line_end),
        line_start=method.line_start,
        line_end=method.line_end,
    )


def _function_signature(fn: FunctionInfo) -> str:
    """Render a one-line signature like 'async def fetch(url)'."""
    prefix = "async def" if fn.is_async else "def"
    return f"{prefix} {fn.name}({', '.join(fn.parameters)})"


def _chunk_id(repo_id: str, qualified_name: str, chunk_type: str) -> str:
    """Deterministic id so re-indexing the same repo overwrites old chunks."""
    return f"{repo_id}:{qualified_name}:{chunk_type}"
