"""Tests for src.indexing.pipeline.

Both the embedder and the vector store are stubbed -- we just want to
verify the wiring (chunker output flows into embedder input, embedder
output flows into vector store upsert), not the components themselves.
"""
from __future__ import annotations

from src.core.models import Language, SourceFile
from src.indexing.pipeline import index_repo
from src.parsing.python_parser import parse_source_file


class FakeEmbedder:
    """Returns one zero vector per input. Records the inputs it saw."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[0.0, 0.0, 0.0] for _ in texts]


class FakeVectorStore:
    """Captures upsert arguments so we can assert on them."""

    def __init__(self) -> None:
        self.upserts: list[tuple] = []

    def upsert_chunks(self, chunks, embeddings) -> None:
        self.upserts.append((chunks, embeddings))


def _make_module(rel_path: str, content: str):
    src = SourceFile(
        relative_path=rel_path,
        language=Language.PYTHON,
        size_bytes=len(content.encode("utf-8")),
        content=content,
    )
    return parse_source_file(src), content


class TestIndexRepo:
    def test_indexes_chunks_and_returns_count(self) -> None:
        module, source = _make_module(
            "m.py",
            '"""M."""\n'
            "def f():\n    return 1\n"
            "class C:\n    def m(self): pass\n",
        )
        embedder = FakeEmbedder()
        vstore = FakeVectorStore()

        n = index_repo(
            [(module, source)],
            repo_id="r1",
            embedder=embedder,
            vector_store=vstore,
        )
        # header + function + class + method = 4 chunks
        assert n == 4
        assert len(vstore.upserts) == 1
        chunks, embeddings = vstore.upserts[0]
        assert len(chunks) == 4
        assert len(embeddings) == 4
        # Embedder must have been called with the chunk contents
        assert embedder.calls and len(embedder.calls[0]) == 4

    def test_empty_module_list_returns_zero(self) -> None:
        embedder = FakeEmbedder()
        vstore = FakeVectorStore()
        n = index_repo([], repo_id="r1", embedder=embedder, vector_store=vstore)
        assert n == 0
        # No upsert should happen for an empty index
        assert vstore.upserts == []
        # And embedder must not have been invoked (cheaper, and avoids edge cases)
        assert embedder.calls == []

    def test_chunks_share_repo_id(self) -> None:
        module, source = _make_module("m.py", "def f(): pass\n")
        vstore = FakeVectorStore()
        index_repo(
            [(module, source)],
            repo_id="my-repo",
            embedder=FakeEmbedder(),
            vector_store=vstore,
        )
        chunks, _ = vstore.upserts[0]
        assert all(c.repo_id == "my-repo" for c in chunks)
