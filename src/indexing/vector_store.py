"""
Vector store wrapper around ChromaDB.

Design choices worth defending:

1. ChromaDB rather than FAISS / Qdrant / pinecone.
   * Persistent local mode -- no external server, ideal for a coursework
     project demonstrated on the developer's laptop.
   * Stores vectors AND metadata together, so we can attach
     `module_path`, `qualified_name`, `line_start`, `line_end` to each
     chunk and recover them at query time without a separate database.

2. One shared collection, repos isolated by `where={"repo_id": ...}` filter.
   * ChromaDB collections are lightweight but managing per-repo collection
     creation / deletion adds another failure surface.
   * Metadata-filter isolation is well-supported by Chroma's query API and
     keeps the wrapper code small.

3. Deterministic chunk_ids (from chunker) + `upsert` semantics:
   * Re-indexing the same repo overwrites old vectors instead of
     accumulating duplicates. This matters when the user re-uploads an
     updated version of the same project.

4. Return `CodeChunk` from `query`, not raw Chroma dicts.
   * Keeps the storage backend hidden behind the model contract -- the QA
     layer should not know whether we use Chroma or something else.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.core.config import settings
from src.core.models import CodeChunk


COLLECTION_NAME = "code_chunks"


class VectorStore:
    """Thin wrapper over a persistent ChromaDB collection."""

    def __init__(self, storage_dir: Optional[Path] = None) -> None:
        self._storage_dir = Path(storage_dir or settings.storage_dir) / "chromadb"
        self._client = None
        self._collection = None

    def _ensure_collection(self):
        """Lazy-init Chroma so importing this module is free of side-effects.

        Why lazy: tests that mock this class shouldn't trigger Chroma's
        on-disk initialisation, and other modules (API schemas etc.) can
        import VectorStore without paying the init cost.
        """
        if self._collection is not None:
            return self._collection

        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self._storage_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        # We supply our own embeddings, so disable Chroma's default embedder.
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )
        return self._collection

    def upsert_chunks(
        self,
        chunks: list[CodeChunk],
        embeddings: list[list[float]],
    ) -> None:
        """Insert or replace vectors for the given chunks.

        We use `upsert` (not `add`) so re-indexing a repo cleanly overwrites
        previous vectors by chunk_id rather than raising a duplicate-id error.
        """
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                "must have the same length"
            )

        collection = self._ensure_collection()
        collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.content for c in chunks],
            metadatas=[_chunk_to_metadata(c) for c in chunks],
        )

    def query(
        self,
        repo_id: str,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[CodeChunk]:
        """Return the top-K most similar chunks belonging to `repo_id`.

        The `where` filter is what keeps repos isolated -- without it a query
        against repo A could leak chunks from repo B.
        """
        collection = self._ensure_collection()
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"repo_id": repo_id},
        )
        return _result_to_chunks(result)

    def delete_repo(self, repo_id: str) -> None:
        """Remove every chunk belonging to `repo_id`."""
        collection = self._ensure_collection()
        collection.delete(where={"repo_id": repo_id})


# ----- internals ------------------------------------------------------------

def _chunk_to_metadata(chunk: CodeChunk) -> dict:
    """Build the metadata dict that Chroma stores alongside the vector.

    Chroma metadata values must be primitives (str/int/float/bool). We only
    store fields we need to reconstruct the CodeChunk at query time.
    """
    return {
        "repo_id": chunk.repo_id,
        "module_path": chunk.module_path,
        "qualified_name": chunk.qualified_name,
        "chunk_type": chunk.chunk_type,
        "line_start": chunk.line_start,
        "line_end": chunk.line_end,
    }


def _result_to_chunks(result: dict) -> list[CodeChunk]:
    """Reconstruct CodeChunk objects from a Chroma `query` result.

    Chroma returns nested lists (one entry per query embedding). We always
    issue a single query, so we index into `[0]`.
    """
    if not result.get("ids") or not result["ids"][0]:
        return []

    ids = result["ids"][0]
    docs = result["documents"][0]
    metas = result["metadatas"][0]

    chunks: list[CodeChunk] = []
    for chunk_id, doc, meta in zip(ids, docs, metas):
        chunks.append(
            CodeChunk(
                chunk_id=chunk_id,
                repo_id=meta["repo_id"],
                module_path=meta["module_path"],
                qualified_name=meta["qualified_name"],
                chunk_type=meta["chunk_type"],
                content=doc,
                line_start=int(meta["line_start"]),
                line_end=int(meta["line_end"]),
            )
        )
    return chunks
