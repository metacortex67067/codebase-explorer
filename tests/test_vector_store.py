"""Tests for src.indexing.vector_store.

We use fake deterministic embeddings (basis vectors) instead of running the
real sentence-transformer model. That keeps the tests:
  * fast -- no model load,
  * offline -- no model download,
  * deterministic -- nearest-neighbour outcomes are predictable.

Each test points the VectorStore at a tmp_path-scoped storage dir so the
tests don't share Chroma state.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.core.models import CodeChunk


# sentence-transformers / chromadb are heavy native dependencies; skip the
# whole module if Chroma isn't available so the rest of the suite still runs.
pytest.importorskip("chromadb")

from src.indexing.vector_store import VectorStore  # noqa: E402


def _chunk(
    chunk_id: str,
    repo_id: str = "repo-a",
    qualified_name: str = "m.f",
    content: str = "def f(): pass",
    chunk_type: str = "function",
) -> CodeChunk:
    return CodeChunk(
        chunk_id=chunk_id,
        repo_id=repo_id,
        module_path="m.py",
        qualified_name=qualified_name,
        chunk_type=chunk_type,
        content=content,
        line_start=1,
        line_end=1,
    )


def _basis(dim: int, idx: int) -> list[float]:
    """Return a one-hot vector of the given dimension."""
    v = [0.0] * dim
    v[idx] = 1.0
    return v


class TestVectorStore:
    def test_upsert_and_query_returns_chunk(self, tmp_path: Path) -> None:
        store = VectorStore(storage_dir=tmp_path)
        chunks = [_chunk("c1", qualified_name="m.add", content="add body")]
        store.upsert_chunks(chunks, embeddings=[_basis(4, 0)])

        results = store.query("repo-a", query_embedding=_basis(4, 0), top_k=5)
        assert len(results) == 1
        assert results[0].chunk_id == "c1"
        assert results[0].qualified_name == "m.add"
        assert results[0].content == "add body"
        assert results[0].line_start == 1

    def test_query_ranks_closer_vector_first(self, tmp_path: Path) -> None:
        store = VectorStore(storage_dir=tmp_path)
        chunks = [
            _chunk("near", qualified_name="m.near"),
            _chunk("far", qualified_name="m.far"),
        ]
        # `near` shares its direction with the query vector; `far` is orthogonal.
        store.upsert_chunks(chunks, embeddings=[_basis(4, 0), _basis(4, 3)])

        results = store.query("repo-a", query_embedding=_basis(4, 0), top_k=2)
        assert [c.qualified_name for c in results] == ["m.near", "m.far"]

    def test_query_isolates_by_repo_id(self, tmp_path: Path) -> None:
        """The where-filter must prevent repo cross-contamination."""
        store = VectorStore(storage_dir=tmp_path)
        store.upsert_chunks(
            [_chunk("a", repo_id="repo-a", qualified_name="m.a")],
            [_basis(4, 0)],
        )
        store.upsert_chunks(
            [_chunk("b", repo_id="repo-b", qualified_name="m.b")],
            [_basis(4, 0)],
        )

        a_results = store.query("repo-a", query_embedding=_basis(4, 0), top_k=5)
        assert [c.qualified_name for c in a_results] == ["m.a"]

        b_results = store.query("repo-b", query_embedding=_basis(4, 0), top_k=5)
        assert [c.qualified_name for c in b_results] == ["m.b"]

    def test_upsert_overwrites_on_duplicate_id(self, tmp_path: Path) -> None:
        """Re-indexing the same chunk_id should replace, not append."""
        store = VectorStore(storage_dir=tmp_path)
        store.upsert_chunks(
            [_chunk("same", content="old body")],
            [_basis(4, 0)],
        )
        store.upsert_chunks(
            [_chunk("same", content="new body")],
            [_basis(4, 0)],
        )
        results = store.query("repo-a", _basis(4, 0), top_k=5)
        assert len(results) == 1
        assert results[0].content == "new body"

    def test_delete_repo_removes_only_that_repo(self, tmp_path: Path) -> None:
        store = VectorStore(storage_dir=tmp_path)
        store.upsert_chunks([_chunk("a", repo_id="repo-a")], [_basis(4, 0)])
        store.upsert_chunks([_chunk("b", repo_id="repo-b")], [_basis(4, 0)])

        store.delete_repo("repo-a")

        assert store.query("repo-a", _basis(4, 0), top_k=5) == []
        assert len(store.query("repo-b", _basis(4, 0), top_k=5)) == 1

    def test_empty_upsert_is_noop(self, tmp_path: Path) -> None:
        store = VectorStore(storage_dir=tmp_path)
        store.upsert_chunks([], [])  # must not raise
        assert store.query("repo-a", _basis(4, 0), top_k=5) == []

    def test_length_mismatch_raises(self, tmp_path: Path) -> None:
        store = VectorStore(storage_dir=tmp_path)
        with pytest.raises(ValueError):
            store.upsert_chunks([_chunk("c1")], embeddings=[])
