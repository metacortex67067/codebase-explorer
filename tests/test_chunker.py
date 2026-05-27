"""Tests for src.indexing.chunker.

We test the chunker against synthetic modules built directly (rather than
against parser output) so the tests stay focused on chunking behaviour
and don't accidentally depend on parser quirks. One end-to-end test does
exercise the parser -> chunker pipeline.
"""
from __future__ import annotations

from src.core.models import Language, SourceFile
from src.indexing.chunker import chunk_module, chunk_modules
from src.parsing.python_parser import parse_source_file


REPO_ID = "test-repo"


def _make_source(rel_path: str, content: str) -> SourceFile:
    return SourceFile(
        relative_path=rel_path,
        language=Language.PYTHON,
        size_bytes=len(content.encode("utf-8")),
        content=content,
    )


class TestChunkModule:
    def test_empty_module_yields_only_header(self) -> None:
        src = _make_source("m.py", '"""Just a docstring."""\n')
        module = parse_source_file(src)
        assert module is not None
        chunks = chunk_module(module, src.content, REPO_ID)
        assert len(chunks) == 1
        assert chunks[0].chunk_type == "module_header"
        assert "Just a docstring." in chunks[0].content

    def test_top_level_function_becomes_chunk(self) -> None:
        content = (
            '"""M."""\n'
            "def add(a, b):\n"
            '    """Return a+b."""\n'
            "    return a + b\n"
        )
        src = _make_source("m.py", content)
        module = parse_source_file(src)
        assert module is not None
        chunks = chunk_module(module, content, REPO_ID)
        # header + one function
        assert len(chunks) == 2
        fn_chunk = next(c for c in chunks if c.chunk_type == "function")
        assert fn_chunk.qualified_name == "m.add"
        # Body must be present verbatim (this is what the LLM cites)
        assert "return a + b" in fn_chunk.content
        # Line range must match the function definition (not the whole file)
        assert fn_chunk.line_start == 2
        assert fn_chunk.line_end == 4

    def test_class_yields_class_and_method_chunks(self) -> None:
        content = (
            "class Counter:\n"
            '    """Counts things."""\n'
            "    def __init__(self):\n"
            "        self.n = 0\n"
            "    def inc(self):\n"
            "        self.n += 1\n"
        )
        src = _make_source("m.py", content)
        module = parse_source_file(src)
        assert module is not None
        chunks = chunk_module(module, content, REPO_ID)
        # Expected: 1 header + 1 class + 2 method chunks = 4
        types = [c.chunk_type for c in chunks]
        assert types.count("module_header") == 1
        assert types.count("class") == 1
        assert types.count("class_method") == 2

        class_chunk = next(c for c in chunks if c.chunk_type == "class")
        # The class chunk holds the whole class body, so a question about
        # how Counter generally works retrieves enough context.
        assert "def __init__" in class_chunk.content
        assert "def inc" in class_chunk.content

        method_chunks = [c for c in chunks if c.chunk_type == "class_method"]
        method_qnames = {c.qualified_name for c in method_chunks}
        assert method_qnames == {"m.Counter.__init__", "m.Counter.inc"}

    def test_module_header_lists_signatures_only(self) -> None:
        """Header is a digest, not full bodies -- otherwise it duplicates
        every function chunk and wastes embedding capacity."""
        content = (
            '"""Top doc."""\n'
            "def visible_fn(x):\n"
            "    secret_marker_in_body = 1\n"
            "    return secret_marker_in_body\n"
        )
        src = _make_source("m.py", content)
        module = parse_source_file(src)
        assert module is not None
        chunks = chunk_module(module, content, REPO_ID)
        header = next(c for c in chunks if c.chunk_type == "module_header")
        # Signature appears
        assert "def visible_fn(x)" in header.content
        # Body does NOT appear -- header is a summary, not a duplicate
        assert "secret_marker_in_body" not in header.content

    def test_chunk_ids_are_deterministic(self) -> None:
        """Re-chunking the same module produces identical chunk_ids so the
        vector store can overwrite cleanly on re-index."""
        content = "def f(): pass\n"
        src = _make_source("m.py", content)
        module = parse_source_file(src)
        assert module is not None
        ids_a = [c.chunk_id for c in chunk_module(module, content, REPO_ID)]
        ids_b = [c.chunk_id for c in chunk_module(module, content, REPO_ID)]
        assert ids_a == ids_b
        # And they contain the qualified_name so they're debuggable in the DB
        assert any("m.f" in cid for cid in ids_a)

    def test_chunk_ids_include_repo_id(self) -> None:
        """Two different repos with the same module name must not collide."""
        content = "def f(): pass\n"
        src = _make_source("m.py", content)
        module = parse_source_file(src)
        assert module is not None
        ids_a = {c.chunk_id for c in chunk_module(module, content, "repo-a")}
        ids_b = {c.chunk_id for c in chunk_module(module, content, "repo-b")}
        assert ids_a.isdisjoint(ids_b)

    def test_async_function_chunk(self) -> None:
        content = "async def fetch(url):\n    return url\n"
        src = _make_source("m.py", content)
        module = parse_source_file(src)
        assert module is not None
        chunks = chunk_module(module, content, REPO_ID)
        fn_chunk = next(c for c in chunks if c.chunk_type == "function")
        assert "async def fetch" in fn_chunk.content

    def test_line_slicing_clamps_oob_ranges(self) -> None:
        """If end_lineno overshoots (defensive case), we should clamp rather
        than raise -- a single odd module must not abort the whole index."""
        content = "def f():\n    return 1\n"
        src = _make_source("m.py", content)
        module = parse_source_file(src)
        assert module is not None
        # Manually corrupt: pretend the function extends to line 999
        module.functions[0].line_end = 999
        chunks = chunk_module(module, content, REPO_ID)
        fn_chunk = next(c for c in chunks if c.chunk_type == "function")
        # Should still produce a chunk with whatever content is available
        assert "return 1" in fn_chunk.content


class TestChunkModulesBatch:
    def test_combines_multiple_modules(self) -> None:
        src1 = _make_source("a.py", "def f(): pass\n")
        src2 = _make_source("b.py", "def g(): pass\n")
        m1 = parse_source_file(src1)
        m2 = parse_source_file(src2)
        assert m1 is not None and m2 is not None
        chunks = chunk_modules([(m1, src1.content), (m2, src2.content)], REPO_ID)
        # Each module yields header + function = 2 chunks, total 4
        assert len(chunks) == 4
        qnames = {c.qualified_name for c in chunks}
        assert {"a", "b", "a.f", "b.g"} <= qnames
