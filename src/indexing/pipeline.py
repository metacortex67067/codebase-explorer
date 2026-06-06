"""
Indexing pipeline: chunk -> embed -> store, in one call for RepoService.

``embedder`` and ``vector_store`` are injectable (tests pass fakes); ``None``
falls back to the default singletons.
"""
from __future__ import annotations

from typing import Optional

from src.core.models import Module
from src.indexing.chunker import chunk_modules
from src.indexing.embedder import Embedder, default_embedder
from src.indexing.vector_store import VectorStore


def index_repo(
    modules_with_source: list[tuple[Module, str]],
    repo_id: str,
    embedder: Optional[Embedder] = None,
    vector_store: Optional[VectorStore] = None,
) -> int:
    """Chunk, embed, and store. Returns the number of chunks indexed.

    `modules_with_source` is a list of (Module, original_file_text) pairs.
    We need the raw text because Module does not store source bodies --
    chunks slice the text by AST line numbers.
    """
    embedder = embedder or default_embedder
    vector_store = vector_store or VectorStore()

    chunks = chunk_modules(modules_with_source, repo_id)
    if not chunks:
        return 0

    embeddings = embedder.embed_batch([c.content for c in chunks])
    vector_store.upsert_chunks(chunks, embeddings)
    return len(chunks)
