"""Tests for src.qa.retriever and src.qa.rag.

Both the embedder and the vector store are stubbed -- we want to verify
the prompt construction, JSON parsing, and citation assembly, not the
underlying retrieval quality (which depends on a real embedding model and
ChromaDB).
"""
from __future__ import annotations

import json
from typing import Optional

import pytest

from src.core.models import CodeChunk, QAResponse
from src.qa.rag import _format_chunks_block, _parse_answer_json, answer_question
from src.qa.retriever import Retriever


# ----- helpers --------------------------------------------------------------

class FakeEmbedder:
    """Returns a fixed deterministic vector for every text it sees."""

    def __init__(self, vector: Optional[list[float]] = None) -> None:
        self.vector = vector or [0.1, 0.2, 0.3]
        self.received_texts: list[str] = []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.received_texts.extend(texts)
        return [list(self.vector) for _ in texts]


class FakeVectorStore:
    """Returns a pre-configured list of chunks regardless of input."""

    def __init__(self, chunks: list[CodeChunk]) -> None:
        self._chunks = chunks
        self.last_query: dict | None = None

    def query(self, repo_id: str, query_embedding: list[float], top_k: int = 5):
        self.last_query = {
            "repo_id": repo_id,
            "embedding": query_embedding,
            "top_k": top_k,
        }
        return list(self._chunks[:top_k])


class FakeLLM:
    """Records the prompt and returns a canned response."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.last_system: str | None = None
        self.last_user: str | None = None
        self.call_count = 0

    def complete(self, system: str, user: str, max_tokens=None) -> str:
        self.call_count += 1
        self.last_system = system
        self.last_user = user
        return self.response


def _chunk(
    qualified_name: str = "pkg.mod.func",
    chunk_type: str = "function",
    module_path: str = "pkg/mod.py",
    line_start: int = 10,
    line_end: int = 20,
    content: str = "def func():\n    return 42\n",
) -> CodeChunk:
    return CodeChunk(
        chunk_id=f"r:{qualified_name}:{chunk_type}",
        repo_id="r",
        module_path=module_path,
        qualified_name=qualified_name,
        chunk_type=chunk_type,
        content=content,
        line_start=line_start,
        line_end=line_end,
    )


# ----- Retriever ------------------------------------------------------------

class TestRetriever:
    def test_embeds_question_and_calls_vector_store(self) -> None:
        embedder = FakeEmbedder([0.5, 0.5])
        chunks = [_chunk()]
        store = FakeVectorStore(chunks)
        r = Retriever(embedder=embedder, vector_store=store)

        result = r.retrieve("repo-1", "what does func do?", top_k=3)

        assert embedder.received_texts == ["what does func do?"]
        assert store.last_query == {
            "repo_id": "repo-1",
            "embedding": [0.5, 0.5],
            "top_k": 3,
        }
        assert result == chunks

    def test_uses_default_top_k_when_not_specified(self) -> None:
        embedder = FakeEmbedder()
        store = FakeVectorStore([_chunk()])
        r = Retriever(embedder=embedder, vector_store=store)

        r.retrieve("repo-1", "q")
        # Default comes from settings.rag_top_k (5)
        assert store.last_query["top_k"] == 5

    def test_empty_question_returns_empty(self) -> None:
        embedder = FakeEmbedder()
        store = FakeVectorStore([_chunk()])
        r = Retriever(embedder=embedder, vector_store=store)

        assert r.retrieve("repo-1", "") == []
        assert r.retrieve("repo-1", "   ") == []
        # Vector store should NOT be queried for an empty question
        assert store.last_query is None


# ----- chunk-block formatting ----------------------------------------------

class TestFormatChunksBlock:
    def test_numbers_chunks_starting_from_one(self) -> None:
        chunks = [
            _chunk(qualified_name="a.f", module_path="a.py", line_start=1, line_end=3),
            _chunk(qualified_name="b.g", module_path="b.py", line_start=5, line_end=7),
        ]
        block = _format_chunks_block(chunks)
        assert "[1] a.py:1-3" in block
        assert "[2] b.py:5-7" in block

    def test_includes_chunk_type_and_qualified_name(self) -> None:
        chunks = [_chunk(qualified_name="pkg.C", chunk_type="class")]
        block = _format_chunks_block(chunks)
        assert "class: pkg.C" in block

    def test_includes_chunk_content(self) -> None:
        chunks = [_chunk(content="MARKER_TEXT_XYZ")]
        block = _format_chunks_block(chunks)
        assert "MARKER_TEXT_XYZ" in block


# ----- JSON answer parsing --------------------------------------------------

class TestParseAnswerJson:
    def test_valid_json(self) -> None:
        raw = json.dumps({"answer": "Hello", "used_chunk_indices": [1, 3]})
        text, idx = _parse_answer_json(raw)
        assert text == "Hello"
        assert idx == [1, 3]

    def test_strips_markdown_fences(self) -> None:
        raw = '```json\n{"answer": "x", "used_chunk_indices": [2]}\n```'
        text, idx = _parse_answer_json(raw)
        assert text == "x"
        assert idx == [2]

    def test_malformed_json_falls_back(self) -> None:
        text, idx = _parse_answer_json("not json")
        assert "not json" in text
        assert idx == []

    def test_non_integer_indices_are_skipped(self) -> None:
        raw = json.dumps({"answer": "y", "used_chunk_indices": [1, "two", None, 3]})
        text, idx = _parse_answer_json(raw)
        assert idx == [1, 3]

    def test_missing_indices_field(self) -> None:
        raw = json.dumps({"answer": "y"})
        text, idx = _parse_answer_json(raw)
        assert text == "y"
        assert idx == []


# ----- end-to-end answer_question ------------------------------------------

class TestAnswerQuestion:
    def test_assembles_response_with_sources(self) -> None:
        chunks = [
            _chunk(qualified_name="pkg.a.foo", module_path="pkg/a.py",
                   line_start=1, line_end=5),
            _chunk(qualified_name="pkg.b.bar", module_path="pkg/b.py",
                   line_start=10, line_end=20),
        ]
        retriever = Retriever(
            embedder=FakeEmbedder(),
            vector_store=FakeVectorStore(chunks),
        )
        llm = FakeLLM(json.dumps({
            "answer": "foo does X.",
            "used_chunk_indices": [1],
        }))

        result = answer_question("r", "what does foo do?", retriever=retriever,
                                 llm_client=llm)

        assert isinstance(result, QAResponse)
        assert result.question == "what does foo do?"
        assert result.answer == "foo does X."
        assert len(result.sources) == 1
        assert result.sources[0].qualified_name == "pkg.a.foo"
        assert result.sources[0].module_path == "pkg/a.py"
        assert result.sources[0].line_start == 1
        assert result.sources[0].line_end == 5

    def test_prompt_contains_chunks_and_question(self) -> None:
        chunks = [_chunk(content="def visible_helper(): pass")]
        retriever = Retriever(FakeEmbedder(), FakeVectorStore(chunks))
        llm = FakeLLM(json.dumps({"answer": "ok", "used_chunk_indices": []}))

        answer_question("r", "Q?", retriever=retriever, llm_client=llm)

        assert llm.last_user is not None
        assert "Q?" in llm.last_user
        assert "visible_helper" in llm.last_user

    def test_no_chunks_means_no_llm_call(self) -> None:
        retriever = Retriever(FakeEmbedder(), FakeVectorStore([]))
        llm = FakeLLM("should not be called")

        result = answer_question("r", "q?", retriever=retriever, llm_client=llm)

        assert llm.call_count == 0
        assert "cannot answer" in result.answer.lower()
        assert result.sources == []

    def test_out_of_range_indices_are_dropped(self) -> None:
        chunks = [_chunk()]  # only one chunk
        retriever = Retriever(FakeEmbedder(), FakeVectorStore(chunks))
        llm = FakeLLM(json.dumps({"answer": "a", "used_chunk_indices": [1, 5, 99]}))

        result = answer_question("r", "q?", retriever=retriever, llm_client=llm)

        assert len(result.sources) == 1  # only index 1 is valid

    def test_duplicate_indices_are_deduplicated(self) -> None:
        chunks = [_chunk(), _chunk(qualified_name="pkg.b")]
        retriever = Retriever(FakeEmbedder(), FakeVectorStore(chunks))
        llm = FakeLLM(json.dumps({"answer": "a", "used_chunk_indices": [1, 1, 2, 2]}))

        result = answer_question("r", "q?", retriever=retriever, llm_client=llm)

        assert len(result.sources) == 2

    def test_malformed_llm_response_keeps_raw_text(self) -> None:
        chunks = [_chunk()]
        retriever = Retriever(FakeEmbedder(), FakeVectorStore(chunks))
        llm = FakeLLM("totally unstructured response")

        result = answer_question("r", "q?", retriever=retriever, llm_client=llm)

        assert "totally unstructured" in result.answer
        assert result.sources == []

    def test_snippet_is_truncated_for_huge_chunks(self) -> None:
        big_content = "x" * 5000
        chunks = [_chunk(content=big_content)]
        retriever = Retriever(FakeEmbedder(), FakeVectorStore(chunks))
        llm = FakeLLM(json.dumps({"answer": "a", "used_chunk_indices": [1]}))

        result = answer_question("r", "q?", retriever=retriever, llm_client=llm)

        assert len(result.sources[0].snippet) < len(big_content)
        assert "truncated" in result.sources[0].snippet
