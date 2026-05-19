"""
Python AST parser.

This module turns raw source text into structured `Module` objects.
Approach: walk the AST produced by stdlib `ast.parse`. We deliberately
do NOT use third-party libraries -- the stdlib `ast` module is sufficient,
auditable, and free of dependency surprises.

Why AST instead of regex / line scanning:
  * Robust to formatting, multi-line definitions, type hints, decorators.
  * Gives us exact line numbers for every node (needed for chunking and
    for showing code citations in QA answers).
  * Lets us walk the tree once and collect imports, classes, functions
    in a single pass.

Defensive choices:
  * Syntax errors are reported, never raised: a single broken file should
    not prevent the rest of the repo from being indexed. We return None
    for that file and let the caller decide what to do.
  * Docstrings come from `ast.get_docstring`, which understands the
    convention of the first string literal in a body.
"""
from __future__ import annotations

import ast
from pathlib import PurePosixPath
from typing import Optional

from src.core.models import ClassInfo, FunctionInfo, Module, SourceFile


class ParseError(Exception):
    """Raised when source text cannot be parsed."""


def parse_source_file(source_file: SourceFile) -> Optional[Module]:
    """Parse one SourceFile into a Module, or return None on syntax error.

    We choose to return None (instead of raising) so callers can keep
    indexing the rest of the repo when a single file is malformed.
    """
    try:
        tree = ast.parse(source_file.content, filename=source_file.relative_path)
    except SyntaxError:
        return None

    qualified_name = _path_to_qualified_name(source_file.relative_path)

    return Module(
        relative_path=source_file.relative_path,
        qualified_name=qualified_name,
        docstring=ast.get_docstring(tree),
        imports=_extract_imports(tree),
        functions=_extract_top_level_functions(tree, qualified_name),
        classes=_extract_top_level_classes(tree, qualified_name),
        line_count=source_file.content.count("\n") + 1,
    )


def parse_repo(source_files: list[SourceFile]) -> tuple[list[Module], list[str]]:
    """Parse every source file. Returns (modules, list_of_failed_paths)."""
    modules: list[Module] = []
    failed: list[str] = []
    for sf in source_files:
        module = parse_source_file(sf)
        if module is None:
            failed.append(sf.relative_path)
        else:
            modules.append(module)
    return modules, failed


# ----- internals ------------------------------------------------------------

def _path_to_qualified_name(relative_path: str) -> str:
    """Convert 'pkg/sub/mod.py' to 'pkg.sub.mod'; drop trailing '__init__'."""
    parts = list(PurePosixPath(relative_path).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else relative_path


def _extract_imports(tree: ast.AST) -> list[str]:
    """Collect top-level imported module names, deduplicated and sorted.

    For `import a.b` we emit 'a.b'.
    For `from a.b import c` we emit 'a.b' (we care about modules, not symbols).
    Relative imports (`from . import x`) yield '.' / '..' etc. -- still useful
    for the dependency graph because it shows intra-package coupling.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            prefix = "." * (node.level or 0)
            module = node.module or ""
            full = (prefix + module) if module else (prefix or "")
            if full:
                names.add(full)
    return sorted(names)


def _decorator_to_str(node: ast.expr) -> str:
    """Render a decorator AST node back to a readable name."""
    try:
        return ast.unparse(node)
    except Exception:
        return "<decorator>"


def _params(args: ast.arguments) -> list[str]:
    """Render function parameters as strings (names only -- type hints
    would bloat the output without helping the LLM much)."""
    names: list[str] = []
    names.extend(a.arg for a in args.posonlyargs)
    names.extend(a.arg for a in args.args)
    if args.vararg:
        names.append("*" + args.vararg.arg)
    names.extend(a.arg for a in args.kwonlyargs)
    if args.kwarg:
        names.append("**" + args.kwarg.arg)
    return names


def _function_info(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    parent_qname: str,
) -> FunctionInfo:
    return FunctionInfo(
        name=node.name,
        qualified_name=f"{parent_qname}.{node.name}",
        line_start=node.lineno,
        line_end=getattr(node, "end_lineno", node.lineno),
        docstring=ast.get_docstring(node),
        parameters=_params(node.args),
        is_async=isinstance(node, ast.AsyncFunctionDef),
        decorators=[_decorator_to_str(d) for d in node.decorator_list],
    )


def _class_info(node: ast.ClassDef, parent_qname: str) -> ClassInfo:
    qname = f"{parent_qname}.{node.name}"
    methods: list[FunctionInfo] = []
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            methods.append(_function_info(item, qname))
    bases = [_decorator_to_str(b) for b in node.bases]
    return ClassInfo(
        name=node.name,
        qualified_name=qname,
        line_start=node.lineno,
        line_end=getattr(node, "end_lineno", node.lineno),
        docstring=ast.get_docstring(node),
        bases=bases,
        methods=methods,
    )


def _extract_top_level_functions(tree: ast.AST, module_qname: str) -> list[FunctionInfo]:
    """Only functions at module scope; methods are stored inside ClassInfo."""
    funcs: list[FunctionInfo] = []
    if isinstance(tree, ast.Module):
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                funcs.append(_function_info(node, module_qname))
    return funcs


def _extract_top_level_classes(tree: ast.AST, module_qname: str) -> list[ClassInfo]:
    classes: list[ClassInfo] = []
    if isinstance(tree, ast.Module):
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                classes.append(_class_info(node, module_qname))
    return classes
