"""
Sentence-transformer wrapper for code-chunk embeddings.

Hides ``sentence_transformers`` behind a small ``embed_batch`` API. The model
(``all-MiniLM-L6-v2``: 384-dim, CPU-friendly) loads lazily once per process,
and tests can swap in a fake that returns deterministic vectors offline.
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
