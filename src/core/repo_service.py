"""
High-level orchestrator: the single entry point used by the API and the
MCP server.

RepoService is a thin facade that wires together the lower layers in the
right order. The public methods correspond one-to-one with what the API
and MCP tools want to do:

  * `create_repo_from_zip` / `create_repo_from_git`  -- ingest -> parse ->
    chunk + embed -> store + summarise.
  * `get_repo` / `list_repos` / `delete_repo`        -- metadata lookups.
  * `get_modules` / `get_module_detail`              -- module listings.
  * `get_dependency_graph`                           -- the parsed graph.
  * `ask`                                            -- RAG Q&A.

Design notes
------------

* Summarising every module via the LLM is expensive (one API call per
  module). We do it eagerly on index time anyway because:
    (1) it warms the cache once, then `/modules` reads are instant;
    (2) any LLM-budget surprise happens at the user-visible "indexing"
        action, not later when they expect a snappy listing.
  If the LLM is unavailable (no API key) we degrade gracefully: each
  module gets a placeholder summary and indexing still succeeds.

* All long-running work runs synchronously. For a coursework demo on a
  small repo this is fine -- index time is dominated by the embedder
  warmup, ~seconds, not minutes. A future version could switch to a
  background task with a job-id endpoint.

* `repo_id` is a short UUID4 hex. We bind ChromaDB and the SQLite row by
  the same id so cleanup is symmetric: `delete_repo` wipes both.
"""
from __future__ import annotations

import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Optional

from src.core.config import settings
from src.core.models import (
    Module,
    ModuleSummary,
    QAResponse,
    RepoIndex,
    SourceFile,
)
from src.core.repo_store import RepoStore
from src.indexing.embedder import Embedder, default_embedder
from src.indexing.pipeline import index_repo as index_repo_pipeline
from src.indexing.vector_store import VectorStore
from src.ingestion.loader import IngestionError, load_from_directory, load_from_git, load_from_zip
from src.llm.client import LLMClient, LLMConfigError, LLMError, default_llm_client
from src.llm.summarizer import summarize_module
from src.parsing.dependency_graph import build_dependency_graph
from src.parsing.python_parser import parse_repo
from src.qa.rag import answer_question
from src.qa.retriever import Retriever


class RepoServiceError(Exception):
    """Raised for service-level failures (unknown repo id, etc.)."""


class RepoService:
    """Facade that orchestrates ingestion, indexing, summarisation and Q&A."""

    def __init__(
        self,
        repo_store: Optional[RepoStore] = None,
        vector_store: Optional[VectorStore] = None,
        embedder: Optional[Embedder] = None,
        llm_client: Optional[LLMClient] = None,
    ) -> None:
        self._repo_store = repo_store or RepoStore()
        self._vector_store = vector_store or VectorStore()
        self._embedder = embedder or default_embedder
        self._llm = llm_client or default_llm_client

    # ----- ingestion entry points -----------------------------------------

    def create_repo_from_zip(self, zip_path: Path, name: str) -> RepoIndex:
        """Ingest a zip-archived repo, index it, and return its RepoIndex."""
        files = load_from_zip(Path(zip_path))
        return self._ingest(files, name)

    def create_repo_from_zip_bytes(self, zip_bytes: bytes, name: str) -> RepoIndex:
        """Convenience wrapper for the API: dump bytes to a temp file first."""
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp.write(zip_bytes)
            tmp_path = Path(tmp.name)
        try:
            if not zipfile.is_zipfile(tmp_path):
                raise IngestionError("Uploaded payload is not a valid zip archive")
            return self.create_repo_from_zip(tmp_path, name=name)
        finally:
            tmp_path.unlink(missing_ok=True)

    def create_repo_from_git(self, git_url: str, name: Optional[str] = None) -> RepoIndex:
        """Clone a git repo, index it, and return its RepoIndex."""
        files = load_from_git(git_url)
        return self._ingest(files, name or _name_from_git_url(git_url))

    def create_repo_from_directory(self, directory: Path, name: str) -> RepoIndex:
        """Index a directory that's already on disk (useful for examples/)."""
        files = load_from_directory(Path(directory))
        return self._ingest(files, name)

    # ----- metadata reads -------------------------------------------------

    def get_repo(self, repo_id: str) -> RepoIndex:
        idx = self._repo_store.get(repo_id)
        if idx is None:
            raise RepoServiceError(f"Unknown repo_id: {repo_id}")
        return idx

    def list_repos(self) -> list[RepoIndex]:
        return self._repo_store.list_all()

    def get_modules(self, repo_id: str) -> list[ModuleSummary]:
        summaries = self._repo_store.get_module_summaries(repo_id)
        if summaries is None:
            raise RepoServiceError(f"Unknown repo_id: {repo_id}")
        return summaries

    def get_module_detail(self, repo_id: str, module_path: str) -> dict:
        """Return one module's full parsed info + its LLM summary.

        `module_path` is matched against `relative_path` (the canonical
        identifier produced by the loader).
        """
        modules = self._repo_store.get_modules(repo_id)
        summaries = self._repo_store.get_module_summaries(repo_id)
        if modules is None or summaries is None:
            raise RepoServiceError(f"Unknown repo_id: {repo_id}")

        module = next(
            (m for m in modules if m["relative_path"] == module_path),
            None,
        )
        if module is None:
            raise RepoServiceError(
                f"Module not found in repo {repo_id}: {module_path}"
            )
        summary = next(
            (s for s in summaries if s.module_path == module_path),
            None,
        )
        return {
            "module": module,
            "summary": summary.model_dump() if summary else None,
        }

    def get_dependency_graph(self, repo_id: str) -> dict[str, list[str]]:
        graph = self._repo_store.get_dependency_graph(repo_id)
        if graph is None:
            raise RepoServiceError(f"Unknown repo_id: {repo_id}")
        return graph

    # ----- Q&A ------------------------------------------------------------

    def ask(self, repo_id: str, question: str) -> QAResponse:
        # Existence check first so callers get a clean "unknown repo" error
        # instead of a confusing "no chunks found" answer.
        if self._repo_store.get(repo_id) is None:
            raise RepoServiceError(f"Unknown repo_id: {repo_id}")
        retriever = Retriever(
            embedder=self._embedder,
            vector_store=self._vector_store,
        )
        return answer_question(
            repo_id=repo_id,
            question=question,
            retriever=retriever,
            llm_client=self._llm,
        )

    # ----- delete ---------------------------------------------------------

    def delete_repo(self, repo_id: str) -> bool:
        """Remove a repo from both vector store and metadata store."""
        deleted = self._repo_store.delete(repo_id)
        # Always try to wipe vectors, even if the metadata row was missing --
        # otherwise an orphan vector set could survive forever.
        try:
            self._vector_store.delete_repo(repo_id)
        except Exception:
            # Vector store failures shouldn't mask the metadata delete result.
            pass
        return deleted

    # ----- internals ------------------------------------------------------

    def _ingest(self, files: list[SourceFile], name: str) -> RepoIndex:
        """Shared post-ingestion pipeline: parse, index, summarise, persist."""
        repo_id = uuid.uuid4().hex[:12]

        modules, _failed = parse_repo(files)

        modules_with_source = _pair_modules_with_source(modules, files)
        chunk_count = index_repo_pipeline(
            modules_with_source=modules_with_source,
            repo_id=repo_id,
            embedder=self._embedder,
            vector_store=self._vector_store,
        )

        graph = build_dependency_graph(modules)
        summaries = self._summarise_all(modules)

        return self._repo_store.upsert(
            repo_id=repo_id,
            name=name,
            file_count=len(files),
            module_count=len(modules),
            chunk_count=chunk_count,
            dependency_graph=graph,
            module_summaries=summaries,
            modules=[m.model_dump() for m in modules],
        )

    def _summarise_all(self, modules: list[Module]) -> list[ModuleSummary]:
        """Best-effort summarisation. Falls back to placeholders on failure.

        Why placeholders rather than raising: a missing API key shouldn't
        block the whole index pipeline -- the user can still browse the
        parsed structure and the dependency graph; they just won't get
        LLM-generated descriptions until they configure the key.
        """
        summaries: list[ModuleSummary] = []
        for module in modules:
            try:
                summaries.append(summarize_module(module, llm_client=self._llm))
            except (LLMConfigError, LLMError):
                summaries.append(
                    ModuleSummary(
                        module_path=module.relative_path,
                        qualified_name=module.qualified_name,
                        summary="(summary unavailable: LLM not configured)",
                        key_responsibilities=[],
                    )
                )
        return summaries


# ----- helpers --------------------------------------------------------------

def _pair_modules_with_source(
    modules: list[Module],
    files: list[SourceFile],
) -> list[tuple[Module, str]]:
    """Match each Module back to its raw source text.

    The chunker needs the original file content to slice chunks by line
    number. Modules and SourceFiles share a `relative_path`, so we build
    a small index and look up.
    """
    source_by_path = {f.relative_path: f.content for f in files}
    pairs: list[tuple[Module, str]] = []
    for module in modules:
        content = source_by_path.get(module.relative_path)
        if content is None:
            # Shouldn't happen in practice -- parse_repo only emits modules
            # whose source it just read -- but be defensive.
            continue
        pairs.append((module, content))
    return pairs


def _name_from_git_url(url: str) -> str:
    """Derive a reasonable display name from a git URL."""
    tail = url.rstrip("/").rsplit("/", 1)[-1]
    return tail.removesuffix(".git") or url
