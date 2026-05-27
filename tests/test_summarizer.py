"""Tests for src.llm.summarizer.

We never hit the real Anthropic API in tests -- the LLMClient is
replaced with a fake that returns canned responses. That keeps tests:
  * offline,
  * deterministic,
  * runnable without an API key.
"""
from __future__ import annotations

import json

from src.core.models import Language, SourceFile
from src.llm.summarizer import summarize_module
from src.parsing.python_parser import parse_source_file


class FakeLLM:
    """Records the (system, user) prompt it was given and returns canned text."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.last_system: str | None = None
        self.last_user: str | None = None

    def complete(self, system: str, user: str, max_tokens=None) -> str:
        self.last_system = system
        self.last_user = user
        return self.response


def _parse(content: str, rel_path: str = "m.py"):
    src = SourceFile(
        relative_path=rel_path,
        language=Language.PYTHON,
        size_bytes=len(content.encode("utf-8")),
        content=content,
    )
    module = parse_source_file(src)
    assert module is not None
    return module


class TestSummarizeModule:
    def test_returns_parsed_summary_from_valid_json(self) -> None:
        module = _parse(
            '"""Math helpers."""\n'
            "def add(a, b): return a + b\n"
            "def sub(a, b): return a - b\n"
        )
        llm = FakeLLM(json.dumps({
            "summary": "Tiny arithmetic helpers.",
            "key_responsibilities": ["addition", "subtraction"],
        }))
        result = summarize_module(module, llm_client=llm)

        assert result.summary == "Tiny arithmetic helpers."
        assert result.key_responsibilities == ["addition", "subtraction"]
        assert result.qualified_name == "m"
        assert result.module_path == "m.py"

    def test_prompt_contains_signatures_not_bodies(self) -> None:
        """We deliberately send signatures only -- bodies would waste tokens."""
        module = _parse(
            '"""M."""\n'
            "def visible_fn(x):\n"
            "    secret_marker = 1\n"
            "    return secret_marker\n"
        )
        llm = FakeLLM(json.dumps({"summary": "ok", "key_responsibilities": []}))
        summarize_module(module, llm_client=llm)

        assert llm.last_user is not None
        assert "def visible_fn(x)" in llm.last_user  # signature present
        assert "secret_marker" not in llm.last_user  # body absent

    def test_malformed_json_falls_back_gracefully(self) -> None:
        module = _parse('"""M."""\n')
        llm = FakeLLM("this is not json at all, sorry")
        result = summarize_module(module, llm_client=llm)
        # Fallback: raw text becomes the summary, no responsibilities
        assert "this is not json" in result.summary
        assert result.key_responsibilities == []

    def test_strips_markdown_json_fences(self) -> None:
        """Some models wrap output in ```json ... ``` despite instructions."""
        module = _parse('"""M."""\n')
        llm = FakeLLM('```json\n{"summary": "ok", "key_responsibilities": ["a"]}\n```')
        result = summarize_module(module, llm_client=llm)
        assert result.summary == "ok"
        assert result.key_responsibilities == ["a"]

    def test_includes_class_methods_in_prompt(self) -> None:
        module = _parse(
            "class Counter:\n"
            "    def __init__(self): self.n = 0\n"
            "    def inc(self): self.n += 1\n"
        )
        llm = FakeLLM(json.dumps({"summary": "ok", "key_responsibilities": []}))
        summarize_module(module, llm_client=llm)
        assert "class Counter" in llm.last_user
        assert "def __init__" in llm.last_user
        assert "def inc" in llm.last_user

    def test_empty_module_still_returns_summary(self) -> None:
        """An empty module should not crash summarization."""
        module = _parse("")
        llm = FakeLLM(json.dumps({"summary": "empty file", "key_responsibilities": []}))
        result = summarize_module(module, llm_client=llm)
        assert result.summary == "empty file"
