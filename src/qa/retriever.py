"""
Retrieval step of the RAG pipeline.

The retriever takes a free-form question, turns it into a vector via the
embedder, and asks the vector store for the top-K most similar code chunks
belonging to a given repository.

Why a separate module (vs. inlining into rag.py)?
  * The retriever is the natural seam to substitute in tests: rag.py logic
    (prompt formatting, LLM call, citation assembly) can be tested with a
    handcrafted list of CodeChunk objects, no real embedder or vector store
    needed.
  * If we ever want to swap retrieval strategies (hybrid BM25 + dense,
    reranking, etc.), only this module changes.

Why a class rather than a free function?
  * Holds the (lazily-initialised) embedder and vector store, so callers
    don't re-create them per question. For the API path that is the
    difference between sub-millisecond and several-second responses
    (the sentence-transformer model is ~80 MB and slow to load).
"""
from __future__ import annotations

from typing import Optional

from src.core.config import settings
from src.core.models import CodeChunk
from src.indexing.embedder import Embedder, default_embedder
from src.indexing.vector_store import VectorStore


class Retriever:
    """Embeds a question and queries the vector store for similar chunks."""

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        vector_store: Optional[VectorStore] = None,
    ) -> None:
        # Default to the module-level singletons so production code doesn't
        # accidentally reload the embedding model per request.
        self._embedder = embedder or default_embedder
        self._vector_store = vector_store or VectorStore()

    def retrieve(
        self,
        repo_id: str,
        question: str,
        top_k: Optional[int] = None,
    ) -> list[CodeChunk]:
        """Return the top-K chunks most similar to `question` for `repo_id`.

        Empty input is treated as "nothing to retrieve" rather than an error
        -- the caller (rag.answer_question) will handle the empty-result
        case explicitly.
        """
        if not question or not question.strip():
            return []

        k = top_k or settings.rag_top_k
        embeddings = self._embedder.embed_batch([question])
        if not embeddings:
            return []

        return self._vector_store.query(
            repo_id=repo_id,
            query_embedding=embeddings[0],
            top_k=k,
        )
