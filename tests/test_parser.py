"""Tests for src.parsing.python_parser and dependency_graph."""
from __future__ import annotations

from pathlib import Path

from src.core.models import Language, SourceFile
from src.ingestion.loader import load_from_directory
from src.parsing.dependency_graph import build_dependency_graph
from src.parsing.python_parser import parse_repo, parse_source_file


def _make_source(rel_path: str, content: str) -> SourceFile:
    return SourceFile(
        relative_path=rel_path,
        language=Language.PYTHON,
        size_bytes=len(content.encode("utf-8")),
        content=content,
    )


class TestParseSourceFile:
    def test_parses_function(self) -> None:
        src = _make_source(
            "m.py",
            '''
def greet(name: str = "world") -> str:
    """Say hello."""
    return f"hello {name}"
'''.lstrip(),
        )
        module = parse_source_file(src)
        assert module is not None
        assert module.qualified_name == "m"
        assert len(module.functions) == 1
        fn = module.functions[0]
        assert fn.name == "greet"
        assert fn.qualified_name == "m.greet"
        assert fn.docstring == "Say hello."
        assert fn.parameters == ["name"]
        assert fn.is_async is False

    def test_parses_async_function(self) -> None:
        src = _make_source("m.py", "async def f(): pass\n")
        module = parse_source_file(src)
        assert module is not None
        assert module.functions[0].is_async is True

    def test_parses_class_with_method(self) -> None:
        src = _make_source(
            "m.py",
            '''
class Foo(Base):
    """A foo."""

    def bar(self, x):
        """Do bar."""
        return x
'''.lstrip(),
        )
        module = parse_source_file(src)
        assert module is not None
        assert len(module.classes) == 1
        cls = module.classes[0]
        assert cls.name == "Foo"
        assert cls.bases == ["Base"]
        assert cls.docstring == "A foo."
        assert len(cls.methods) == 1
        assert cls.methods[0].name == "bar"
        assert cls.methods[0].qualified_name == "m.Foo.bar"

    def test_extracts_imports(self) -> None:
        src = _make_source(
            "m.py",
            "import os\nfrom collections import OrderedDict\nfrom . import sibling\n",
        )
        module = parse_source_file(src)
        assert module is not None
        assert "os" in module.imports
        assert "collections" in module.imports
        assert "." in module.imports

    def test_extracts_module_docstring(self) -> None:
        src = _make_source("m.py", '"""Top-level."""\nx = 1\n')
        module = parse_source_file(src)
        assert module is not None
        assert module.docstring == "Top-level."

    def test_syntax_error_returns_none(self) -> None:
        src = _make_source("bad.py", "def broken(:\n")
        assert parse_source_file(src) is None

    def test_init_dropped_from_qualified_name(self) -> None:
        src = _make_source("pkg/__init__.py", "")
        module = parse_source_file(src)
        assert module is not None
        assert module.qualified_name == "pkg"

    def test_nested_path_becomes_dotted(self) -> None:
        src = _make_source("pkg/sub/mod.py", "")
        module = parse_source_file(src)
        assert module is not None
        assert module.qualified_name == "pkg.sub.mod"

    def test_function_line_numbers(self) -> None:
        src = _make_source("m.py", "\n\ndef f():\n    return 1\n")
        module = parse_source_file(src)
        assert module is not None
        fn = module.functions[0]
        assert fn.line_start == 3
        assert fn.line_end == 4


class TestParseRepo:
    def test_separates_failed_files(self, sample_repo: Path) -> None:
        files = load_from_directory(sample_repo)
        modules, failed = parse_repo(files)
        assert any("broken.py" in p for p in failed)
        assert any(m.qualified_name == "mypkg.utils" for m in modules)


class TestDependencyGraph:
    def test_resolves_internal_import(self, sample_repo: Path) -> None:
        files = load_from_directory(sample_repo)
        modules, _ = parse_repo(files)
        graph = build_dependency_graph(modules)
        # mypkg.service imports from mypkg.utils
        assert "mypkg.utils" in graph["mypkg.service"]

    def test_drops_external_imports(self) -> None:
        src = _make_source("m.py", "import os\nimport sys\n")
        modules, _ = parse_repo([src])
        graph = build_dependency_graph(modules)
        assert graph["m"] == []

    def test_no_self_edges(self, sample_repo: Path) -> None:
        files = load_from_directory(sample_repo)
        modules, _ = parse_repo(files)
        graph = build_dependency_graph(modules)
        for mod, deps in graph.items():
            assert mod not in deps
