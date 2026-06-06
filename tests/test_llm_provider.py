"""
Tests for the multi-provider LLMClient and the bundled web UI.

The OpenAI-compatible path (used by Ollama and clouds like Groq/OpenRouter)
is exercised with a fake client injected in place of the real SDK object,
so the test stays offline and deterministic.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import app
from src.llm.client import LLMClient


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeOpenAI:
    """Mimics the tiny slice of the openai SDK that LLMClient touches."""

    def __init__(self):
        self.calls = []

        class _Completions:
            def create(inner, **kwargs):
                self.calls.append(kwargs)
                return _FakeCompletion("hello from local model")

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


def test_openai_compatible_complete_sends_system_and_user():
    client = LLMClient(provider="ollama", model="qwen2.5-coder:7b")
    fake = _FakeOpenAI()
    client._client = fake  # skip real SDK construction

    out = client.complete(system="SYS", user="USER")

    assert out == "hello from local model"
    sent = fake.calls[0]
    assert sent["model"] == "qwen2.5-coder:7b"
    roles = [m["role"] for m in sent["messages"]]
    assert roles == ["system", "user"]
    assert sent["messages"][0]["content"] == "SYS"
    assert sent["messages"][1]["content"] == "USER"


def test_openai_provider_uses_same_path_as_ollama():
    client = LLMClient(provider="openai", base_url="https://api.groq.com/openai/v1")
    client._client = _FakeOpenAI()
    assert client.complete(system="s", user="u") == "hello from local model"


def test_unknown_provider_raises_config_error():
    from src.llm.client import LLMConfigError

    client = LLMClient(provider="bogus")
    try:
        client.complete(system="s", user="u")
        assert False, "expected LLMConfigError"
    except LLMConfigError:
        pass


def test_web_ui_served_at_root():
    c = TestClient(app)
    r = c.get("/")
    assert r.status_code == 200
    assert "Codebase Explorer" in r.text
