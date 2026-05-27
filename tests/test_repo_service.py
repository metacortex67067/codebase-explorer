"""Tests for src.core.repo_store and src.core.repo_service.

The vector store, embedder, and LLM are all replaced with fakes so the
tests stay offline, fast, and deterministic. We use the `sample_repo`
fixture from conftest as the input repo.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pytest

from src.core.models import CodeChunk, ModuleSummary
from src.core.repo_service import RepoService, RepoServiceError
from src.core.repo_store import RepoStore


# ----- fakes ----------------------------------------------------------------

class FakeEmbedder:
    def __init__(self, dim: int = 3) -> None:
        self.dim = dim
        self.calls: int = 0

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        # Deterministic, content-dependent vector so different chunks get
        # different (but reproducible) embeddings.
        return [
            [float(len(t) % 17), float(sum(map(ord, t[:8])) % 23), 0.5]
            for t in texts
        ]


class FakeVectorStore:
    """In-memory stand-in that records upserts and returns canned query results."""

    def __init__(self, canned_query_result: Optional[list[CodeChunk]] = None) -> None:
        self.upserts: list[tuple[list[CodeChunk], list[list[float]]]] = []
        self.deleted: list[str] = []
        self._canned = canned_query_result or []

    def upsert_chunks(self, chunks, embeddings):
        self.upserts.append((list(chunks), list(embeddings)))

    def query(self, repo_id, query_embedding, top_k=5):
        return list(self._canned[:top_k])

    def delete_repo(self, repo_id):
        self.deleted.append(repo_id)


class FakeLLM:
    def __init__(self, response_factory=None) -> None:
        # Default: a valid summary JSON regardless of input.
        self._factory = response_factory or (
            lambda system, user: json.dumps({
                "summary": "auto summary",
                "key_responsibilities": [],
            })
        )
        self.call_count = 0
        self.last_user: Optional[str] = None

    def complete(self, system: str, user: str, max_tokens=None) -> str:
        self.call_count += 1
        self.last_user = user
        return self._factory(system, user)


# ----- RepoStore ------------------------------------------------------------

class TestRepoStore:
    def test_upsert_and_get(self, tmp_path: Path) -> None:
        store = RepoStore(storage_dir=tmp_path)

        summary = ModuleSummary(
            module_path="m.py", qualified_name="m",
            summary="s", key_responsibilities=["x"],
        )
        idx = store.upsert(
            repo_id="r1", name="demo",
            file_count=3, module_count=2, chunk_count=10,
            dependency_graph={"a": ["b"], "b": []},
            module_summaries=[summary],
            modules=[{"relative_path": "m.py", "qualified_name": "m"}],
        )

        assert idx.repo_id == "r1"
        assert idx.name == "demo"
        assert idx.chunk_count == 10
        # indexed_at is an ISO timestamp added automatically
        assert idx.indexed_at

        fetched = store.get("r1")
        assert fetched is not None
        assert fetched.module_count == 2

    def test_get_missing_returns_none(self, tmp_path: Path) -> None:
        store = RepoStore(storage_dir=tmp_path)
        assert store.get("nope") is None

    def test_list_all_orders_by_indexed_at_desc(self, tmp_path: Path) -> None:
        store = RepoStore(storage_dir=tmp_path)
        store.upsert("a", "A", 1, 1, 1, {}, [], [], indexed_at="2025-01-01T00:00:00")
        store.upsert("b", "B", 1, 1, 1, {}, [], [], indexed_at="2025-06-01T00:00:00")
        ids = [r.repo_id for r in store.list_all()]
        assert ids == ["b", "a"]

    def test_dependency_graph_roundtrip(self, tmp_path: Path) -> None:
        store = RepoStore(storage_dir=tmp_path)
        graph = {"pkg.a": ["pkg.b"], "pkg.b": []}
        store.upsert("r", "r", 0, 0, 0, graph, [], [])
        assert store.get_dependency_graph("r") == graph
        assert store.get_dependency_graph("missing") is None

    def test_module_summaries_roundtrip(self, tmp_path: Path) -> None:
        store = RepoStore(storage_dir=tmp_path)
        s = ModuleSummary(
            module_path="m.py", qualified_name="m",
            summary="hello", key_responsibilities=["r1"],
        )
        store.upsert("r", "r", 0, 0, 0, {}, [s], [])
        back = store.get_module_summaries("r")
        assert back is not None
        assert len(back) == 1
        assert back[0].summary == "hello"
        assert back[0].key_responsibilities == ["r1"]

    def test_modules_roundtrip(self, tmp_path: Path) -> None:
        store = RepoStore(storage_dir=tmp_path)
        modules = [{"relative_path": "a.py", "qualified_name": "a"}]
        store.upsert("r", "r", 0, 0, 0, {}, [], modules)
        assert store.get_modules("r") == modules

    def test_delete(self, tmp_path: Path) -> None:
        store = RepoStore(storage_dir=tmp_path)
        store.upsert("r", "r", 0, 0, 0, {}, [], [])
        assert store.delete("r") is True
        assert store.get("r") is None
        assert store.delete("r") is False

    def test_upsert_overwrites_existing(self, tmp_path: Path) -> None:
        store = RepoStore(storage_dir=tmp_path)
        store.upsert("r", "first", 1, 1, 1, {}, [], [])
        store.upsert("r", "second", 9, 9, 9, {}, [], [])
        idx = store.get("r")
        assert idx is not None
        assert idx.name == "second"
        assert idx.chunk_count == 9


# ----- RepoService ----------------------------------------------------------

def _make_service(tmp_path: Path, **overrides) -> tuple[RepoService, dict]:
    """Build a RepoService wired with fakes; return service + the fake objects."""
    repo_store = RepoStore(storage_dir=tmp_path)
    vector_store = FakeVectorStore()
    embedder = FakeEmbedder()
    llm = FakeLLM()
    parts = {
        "repo_store": repo_store,
        "vector_store": vector_store,
        "embedder": embedder,
        "llm_client": llm,
    }
    parts.update(overrides)
    svc = RepoService(**parts)
    return svc, parts


class TestRepoServiceIndex:
    def test_index_from_directory(self, tmp_path: Path, sample_repo: Path) -> None:
        svc, parts = _make_service(tmp_path)
        idx = svc.create_repo_from_directory(sample_repo, name="sample")

        # The sample repo has 3 valid .py files (init, utils, service) plus
        # the broken one (filtered out by parser) and tests_inside/test_dummy.
        # parse_repo returns successful modules only.
        assert idx.file_count >= 3
        assert idx.module_count >= 3
        assert idx.chunk_count > 0
        assert idx.name == "sample"

        # Vector store should have received an upsert
        assert len(parts["vector_store"].upserts) == 1
        chunks, embeddings = parts["vector_store"].upserts[0]
        assert len(chunks) == len(embeddings) == idx.chunk_count
        assert all(c.repo_id == idx.repo_id for c in chunks)

    def test_index_from_zip(self, tmp_path: Path, sample_zip: Path) -> None:
        svc, _ = _make_service(tmp_path)
        idx = svc.create_repo_from_zip(sample_zip, name="zipped")
        assert idx.file_count >= 3

    def test_index_from_zip_bytes(self, tmp_path: Path, sample_zip: Path) -> None:
        svc, _ = _make_service(tmp_path)
        zip_bytes = sample_zip.read_bytes()
        idx = svc.create_repo_from_zip_bytes(zip_bytes, name="bytes")
        assert idx.module_count >= 3

    def test_zip_bytes_rejects_non_zip(self, tmp_path: Path) -> None:
        svc, _ = _make_service(tmp_path)
        from src.ingestion.loader import IngestionError
        with pytest.raises(IngestionError):
            svc.create_repo_from_zip_bytes(b"not a zip", name="bad")

    def test_summaries_are_generated_per_module(
        self, tmp_path: Path, sample_repo: Path,
    ) -> None:
        svc, parts = _make_service(tmp_path)
        idx = svc.create_repo_from_directory(sample_repo, name="s")
        # Summariser should have been called once per parsed module
        assert parts["llm_client"].call_count == idx.module_count

    def test_llm_failure_does_not_break_indexing(
        self, tmp_path: Path, sample_repo: Path,
    ) -> None:
        """If the LLM blows up, we should still get a stored repo with
        placeholder summaries -- the rest of the pipeline must not fail."""
        from src.llm.client import LLMConfigError

        class ExplodingLLM:
            def complete(self, system, user, max_tokens=None):
                raise LLMConfigError("nope")

        svc, _ = _make_service(tmp_path, llm_client=ExplodingLLM())
        idx = svc.create_repo_from_directory(sample_repo, name="s")
        assert idx.module_count >= 3

        summaries = svc.get_modules(idx.repo_id)
        assert all("unavailable" in s.summary for s in summaries)


class TestRepoServiceReads:
    def test_get_repo_unknown_raises(self, tmp_path: Path) -> None:
        svc, _ = _make_service(tmp_path)
        with pytest.raises(RepoServiceError):
            svc.get_repo("nope")

    def test_list_repos(self, tmp_path: Path, sample_repo: Path) -> None:
        svc, _ = _make_service(tmp_path)
        svc.create_repo_from_directory(sample_repo, name="a")
        svc.create_repo_from_directory(sample_repo, name="b")
        repos = svc.list_repos()
        assert {r.name for r in repos} == {"a", "b"}

    def test_get_module_detail(self, tmp_path: Path, sample_repo: Path) -> None:
        svc, _ = _make_service(tmp_path)
        idx = svc.create_repo_from_directory(sample_repo, name="s")
        modules_meta = svc.get_modules(idx.repo_id)
        a_path = modules_meta[0].module_path

        detail = svc.get_module_detail(idx.repo_id, a_path)
        assert detail["module"]["relative_path"] == a_path
        assert detail["summary"] is not None

    def test_get_module_detail_unknown_module(
        self, tmp_path: Path, sample_repo: Path,
    ) -> None:
        svc, _ = _make_service(tmp_path)
        idx = svc.create_repo_from_directory(sample_repo, name="s")
        with pytest.raises(RepoServiceError):
            svc.get_module_detail(idx.repo_id, "nothing/here.py")

    def test_dependency_graph_returned(
        self, tmp_path: Path, sample_repo: Path,
    ) -> None:
        svc, _ = _make_service(tmp_path)
        idx = svc.create_repo_from_directory(sample_repo, name="s")
        graph = svc.get_dependency_graph(idx.repo_id)
        # service.py imports utils.py inside the sample repo
        assert "mypkg.service" in graph
        assert "mypkg.utils" in graph["mypkg.service"]


class TestRepoServiceAsk:
    def test_ask_unknown_repo_raises(self, tmp_path: Path) -> None:
        svc, _ = _make_service(tmp_path)
        with pytest.raises(RepoServiceError):
            svc.ask("nope", "what?")

    def test_ask_returns_qa_response(
        self, tmp_path: Path, sample_repo: Path,
    ) -> None:
        # Vector store will return a canned chunk; LLM will return canned JSON.
        canned_chunk = CodeChunk(
            chunk_id="x:y:z",
            repo_id="will-be-replaced",
            module_path="mypkg/utils.py",
            qualified_name="mypkg.utils.add",
            chunk_type="function",
            content="def add(a, b): return a + b",
            line_start=1, line_end=3,
        )

        def factory(system, user):
            # If it's the answer prompt, return the answer schema; otherwise
            # the summary schema (used during indexing).
            if "Retrieved code chunks" in user:
                return json.dumps({
                    "answer": "It adds two numbers.",
                    "used_chunk_indices": [1],
                })
            return json.dumps({"summary": "x", "key_responsibilities": []})

        svc, _ = _make_service(
            tmp_path,
            vector_store=FakeVectorStore(canned_query_result=[canned_chunk]),
            llm_client=FakeLLM(response_factory=factory),
        )
        idx = svc.create_repo_from_directory(sample_repo, name="s")

        resp = svc.ask(idx.repo_id, "what does add do?")
        assert resp.answer == "It adds two numbers."
        assert len(resp.sources) == 1
        assert resp.sources[0].qualified_name == "mypkg.utils.add"


class TestRepoServiceDelete:
    def test_delete_removes_repo(self, tmp_path: Path, sample_repo: Path) -> None:
        svc, parts = _make_service(tmp_path)
        idx = svc.create_repo_from_directory(sample_repo, name="s")
        assert svc.delete_repo(idx.repo_id) is True

        with pytest.raises(RepoServiceError):
            svc.get_repo(idx.repo_id)
        # Vector store delete should have been called too
        assert idx.repo_id in parts["vector_store"].deleted

    def test_delete_missing_returns_false(self, tmp_path: Path) -> None:
        svc, _ = _make_service(tmp_path)
        assert svc.delete_repo("nope") is False
