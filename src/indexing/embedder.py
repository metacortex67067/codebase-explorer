"""
Sentence-transformer wrapper for code-chunk embeddings.

We hide `sentence_transformers` behind a thin class so the rest of the code
depends only on a tiny API surface (`embed_batch`). That matters because:
  * Loading the model is slow (~1-3s + ~100 MB) and we want to do it lazily,
    once per process, not on import.
  * Tests can swap this class with a fake that returns deterministic vectors,
    avoiding a 100 MB download in CI and making tests fast and offline.
  * If we ever want to switch model providers (OpenAI, Voyage, a different
    local model), only this module changes.

Model choice: `all-MiniLM-L6-v2`.
  * 384-dim, ~22 MB, runs on CPU in real time even on a laptop.
  * Trained for general semantic similarity. Code is not its training
    domain, but for retrieval over function/class chunks with docstrings it
    works well enough -- the bottleneck for a coursework project is rarely
    the embedding quality but the surrounding RAG plumbing.
"""
from __future__ import annotations

from typing import Optional

from src.core.config import settings


class Embedder:
    """Lazily-loaded sentence-transformer wrapper."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        self._model_name = model_name or settings.embedding_model
        self._model = None  # type: ignore[assignment]

    def _ensure_model(self) -> None:
        """Load the model on first use, then keep it cached."""
        if self._model is None:
            # Imported here so test environments that mock this class don't
            # need to import sentence_transformers at all.
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of strings. Returns a list of vectors (one per text).

        We accept and return plain Python lists -- not numpy arrays -- so the
        boundary between this module and the vector store stays JSON-friendly
        and easy to mock in tests.
        """
        if not texts:
            return []
        self._ensure_model()
        # `encode` returns a numpy array; tolist() gives us list[list[float]].
        vectors = self._model.encode(  # type: ignore[union-attr]
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()


# Module-level singleton -- callers reuse the same loaded model across
# requests. Tests construct their own Embedder() (or a mock) directly.
default_embedder = Embedder()
