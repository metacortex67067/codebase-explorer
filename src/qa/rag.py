"""
RAG (retrieval-augmented generation) Q&A loop.

Given a question about a previously indexed repository, this module:

  1. Retrieves the top-K most relevant code chunks (delegated to Retriever).
  2. Renders them into a numbered prompt block.
  3. Asks the LLM to answer using only those chunks, and to indicate which
     chunks were actually used (by their 1-based index in the prompt).
  4. Parses the JSON response and builds a QAResponse with QASource
     citations reconstructed from the retrieved chunks.

Design choices worth defending:

* "Cite by index" rather than "cite by qualified_name".
  Asking the LLM to copy back the qualified_name is fragile -- the model may
  paraphrase, abbreviate, or hallucinate a name. Numeric indices into a list
  it just saw are far easier for it to get right, and we recover the full
  metadata (module path, line range) ourselves from the retrieved list.

* Graceful degradation on bad JSON.
  Same philosophy as the summarizer: if the LLM returns malformed JSON we
  keep the raw text as the answer and return an empty source list rather
  than crashing. A coursework demo must not blow up because of a one-off
  model wobble.

* Snippet truncation in citations.
  When showing a citation back to the user, we cap the snippet length so
  the API response doesn't balloon (a single class chunk can be hundreds
  of lines). The full text is still in the vector store if needed.
"""
from __future__ import annotations

import json
from typing import Optional

from src.core.config import settings
from src.core.models import CodeChunk, QAResponse, QASource
from src.llm.client import LLMClient, default_llm_client
from src.llm.prompts import ANSWER_QUESTION_SYSTEM, ANSWER_QUESTION_USER_TEMPLATE
from src.qa.retriever import Retriever


MAX_SNIPPET_CHARS = 800


def answer_question(
    repo_id: str,
    question: str,
    retriever: Optional[Retriever] = None,
    llm_client: Optional[LLMClient] = None,
    top_k: Optional[int] = None,
) -> QAResponse:
    """Run the RAG loop and return a QAResponse with citations.

    Both `retriever` and `llm_client` are injectable so tests can supply
    deterministic fakes -- the production path uses the module singletons.
    """
    retriever = retriever or Retriever()
    llm = llm_client or default_llm_client
    k = top_k or settings.rag_top_k

    chunks = retriever.retrieve(repo_id=repo_id, question=question, top_k=k)

    if not chunks:
        # Nothing to ground the answer on. We deliberately don't call the LLM
        # in this branch: a Q&A system that confidently answers without any
        # retrieved context is worse than one that admits it has nothing.
        return QAResponse(
            question=question,
            answer="No indexed code chunks were found for this repository, "
                   "so I cannot answer the question.",
            sources=[],
        )

    user_message = ANSWER_QUESTION_USER_TEMPLATE.format(
        question=question,
        chunks_block=_format_chunks_block(chunks),
    )

    raw = llm.complete(
        system=ANSWER_QUESTION_SYSTEM,
        user=user_message,
    )

    answer_text, used_indices = _parse_answer_json(raw)
    sources = _indices_to_sources(used_indices, chunks)

    return QAResponse(
        question=question,
        answer=answer_text,
        sources=sources,
    )


# ----- internals ------------------------------------------------------------

def _format_chunks_block(chunks: list[CodeChunk]) -> str:
    """Render retrieved chunks as a numbered, labelled block for the prompt.

    Each chunk is prefixed by [index] module_path:line_start-line_end so the
    LLM can refer back to it by index and the human reading the prompt can
    quickly orient.
    """
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        header = (
            f"[{i}] {chunk.module_path}:{chunk.line_start}-{chunk.line_end} "
            f"({chunk.chunk_type}: {chunk.qualified_name})"
        )
        parts.append(header)
        parts.append(chunk.content)
        parts.append("")  # blank line between chunks
    return "\n".join(parts).rstrip()


def _parse_answer_json(raw: str) -> tuple[str, list[int]]:
    """Extract (answer_text, used_chunk_indices) from the LLM response.

    Tolerates ```json fences and falls back to (raw, []) on malformed output.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return raw or "(empty response)", []

    answer = str(obj.get("answer", "")).strip() or "(empty answer)"

    raw_indices = obj.get("used_chunk_indices") or []
    indices: list[int] = []
    if isinstance(raw_indices, list):
        for x in raw_indices:
            try:
                indices.append(int(x))
            except (TypeError, ValueError):
                # Skip unparseable index entries rather than crashing.
                continue
    return answer, indices


def _indices_to_sources(
    indices: list[int],
    chunks: list[CodeChunk],
) -> list[QASource]:
    """Convert 1-based chunk indices into QASource objects.

    Out-of-range indices are silently dropped -- the LLM occasionally
    hallucinates an extra index, and that should not surface as an error.
    Duplicates are de-duplicated while preserving first-seen order.
    """
    seen: set[int] = set()
    sources: list[QASource] = []
    for idx in indices:
        if idx in seen:
            continue
        seen.add(idx)
        if not (1 <= idx <= len(chunks)):
            continue
        chunk = chunks[idx - 1]
        sources.append(_chunk_to_source(chunk))
    return sources


def _chunk_to_source(chunk: CodeChunk) -> QASource:
    snippet = chunk.content
    if len(snippet) > MAX_SNIPPET_CHARS:
        snippet = snippet[:MAX_SNIPPET_CHARS] + "\n... (truncated)"
    return QASource(
        module_path=chunk.module_path,
        qualified_name=chunk.qualified_name,
        line_start=chunk.line_start,
        line_end=chunk.line_end,
        snippet=snippet,
    )
