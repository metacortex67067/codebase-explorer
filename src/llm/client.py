"""
Provider-agnostic chat LLM wrapper.

Three backends are selected by ``settings.llm_provider``: ``ollama`` and
``openai`` (any OpenAI-compatible endpoint such as Groq or OpenRouter) share
one code path; ``anthropic`` uses its own SDK. The backend is built lazily on
the first ``complete()`` call, so importing this module needs no key or network.
"""
from __future__ import annotations

from typing import Optional

from src.core.config import settings


class LLMError(Exception):
    """Raised when the LLM call fails for any reason."""


class LLMConfigError(LLMError):
    """Raised when the LLM client is not configured (e.g. missing API key)."""


# Providers that speak the OpenAI chat-completions protocol. Ollama exposes
# exactly this on /v1, which is why a local model needs no special code.
_OPENAI_COMPATIBLE = {"ollama", "openai"}


class LLMClient:
    """Minimal chat client used for module summaries and Q&A.

    Provider-agnostic: the constructor records config, and the concrete
    backend (OpenAI-compatible HTTP or Anthropic) is created lazily on the
    first `complete()` call so that merely importing this module -- as the
    tests do -- never needs a key or a network.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> None:
        self._provider = (provider or settings.llm_provider).lower()
        self._base_url = base_url or settings.llm_base_url
        self._model = model or settings.llm_model
        self._max_tokens = max_tokens or settings.llm_max_tokens
        self._temperature = (
            temperature if temperature is not None else settings.llm_temperature
        )
        if api_key is not None:
            self._api_key = api_key
        elif self._provider == "anthropic":
            self._api_key = settings.anthropic_api_key
        else:
            self._api_key = settings.llm_api_key
        self._client = None  # lazy init

    def _ensure_client(self):
        """Instantiate the backend client on first use."""
        if self._client is not None:
            return
        if self._provider in _OPENAI_COMPATIBLE:
            self._client = self._build_openai_client()
        elif self._provider == "anthropic":
            self._client = self._build_anthropic_client()
        else:
            raise LLMConfigError(
                f"Unknown LLM provider: {self._provider!r}. "
                "Set LLM_PROVIDER to one of: ollama, openai, anthropic."
            )

    def _build_openai_client(self):
        # Ollama doesn't check the key, but the SDK requires a non-empty one.
        if not self._api_key:
            self._api_key = "not-needed"
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise LLMConfigError(
                "openai package is not installed. Run: pip install openai"
            ) from exc
        return OpenAI(base_url=self._base_url, api_key=self._api_key)

    def _build_anthropic_client(self):
        if not self._api_key:
            raise LLMConfigError(
                "ANTHROPIC_API_KEY is not set. Provide it via environment "
                "or .env to enable LLM features, or switch LLM_PROVIDER to "
                "'ollama' for a free local model."
            )
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise LLMConfigError(
                "anthropic package is not installed. Run: pip install anthropic"
            ) from exc
        return anthropic.Anthropic(api_key=self._api_key)

    def complete(self, system: str, user: str) -> str:
        """Send a system+user prompt and return the text response.

        Raises LLMError on any backend failure so callers can degrade
        gracefully (placeholder summaries, raw-text answers, etc.).
        """
        self._ensure_client()
        if self._provider in _OPENAI_COMPATIBLE:
            return self._complete_openai(system, user)
        return self._complete_anthropic(system, user)

    def _complete_openai(self, system: str, user: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        except Exception as exc:
            raise LLMError(
                f"LLM call failed ({self._provider} @ {self._base_url}): {exc}. "
                "If using Ollama, make sure it is running (`ollama serve`) and "
                f"the model is pulled (`ollama pull {self._model}`)."
            ) from exc
        return (response.choices[0].message.content or "").strip()

    def _complete_anthropic(self, system: str, user: str) -> str:
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except Exception as exc:
            raise LLMError(f"Anthropic API call failed: {exc}") from exc

        parts = [block.text for block in response.content if hasattr(block, "text")]
        return "\n".join(parts).strip()


# Module-level singleton used by the production code paths.
default_llm_client = LLMClient()
