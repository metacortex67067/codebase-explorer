"""
Anthropic API client wrapper.

Why wrap rather than use `anthropic.Anthropic` directly:
  * Lazy creation -- the SDK does an env-var check on instantiation, so
    importing this module is side-effect-free until something actually
    needs to call the LLM (tests, for example, never need a real client).
  * One place to handle retries on rate limits / transient errors. The
    rest of the codebase calls `LLMClient.complete(system, user)` and
    doesn't care about exponential backoff.
  * One place to enforce missing-API-key errors with a clear message
    (without this, `anthropic` raises deep in the SDK).
"""
from __future__ import annotations

import time
from typing import Optional

from src.core.config import settings


class LLMConfigError(Exception):
    """Raised when the LLM client cannot be configured (missing key, etc.)."""


class LLMError(Exception):
    """Raised when the LLM call fails after all retries."""


class LLMClient:
    """Thin wrapper around `anthropic.Anthropic` with retries."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_retries: int = 3,
    ) -> None:
        self._api_key = api_key or settings.anthropic_api_key
        self._model = model or settings.llm_model
        self._max_retries = max_retries
        self._client = None  # built lazily

    def _ensure_client(self):
        """Construct the SDK client on first use."""
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise LLMConfigError(
                "ANTHROPIC_API_KEY is not set. Add it to .env or export it."
            )
        # Import here so test environments that mock LLMClient don't need to
        # have `anthropic` installed at all.
        from anthropic import Anthropic
        self._client = Anthropic(api_key=self._api_key)
        return self._client

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Send a single-turn request and return the assistant's text.

        Retries on transient errors with exponential backoff. We re-raise
        as our own LLMError so callers don't catch anthropic-specific
        exception types (loose coupling).
        """
        client = self._ensure_client()
        max_tokens = max_tokens or settings.llm_max_tokens

        last_exc: Optional[Exception] = None
        for attempt in range(self._max_retries):
            try:
                response = client.messages.create(
                    model=self._model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                # The Messages API returns a list of content blocks; we
                # concatenate the text blocks. For our prompts (always asking
                # for plain text or JSON) there is normally just one block.
                return _extract_text(response)
            except Exception as exc:  # SDK exceptions vary; catch broadly
                last_exc = exc
                if not _is_retryable(exc) or attempt == self._max_retries - 1:
                    break
                # Exponential backoff: 1s, 2s, 4s...
                time.sleep(2 ** attempt)

        raise LLMError(f"LLM call failed after {self._max_retries} attempts: {last_exc}") from last_exc


def _extract_text(response) -> str:
    """Pull plain text out of an anthropic Messages API response."""
    parts: list[str] = []
    for block in getattr(response, "content", []) or []:
        # anthropic SDK returns TextBlock objects with a `.text` attribute
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "".join(parts)


def _is_retryable(exc: Exception) -> bool:
    """Decide whether to retry given an SDK exception.

    We're conservative: only obvious transient errors (rate limit, overload,
    network timeout) get retried. Programming errors (invalid request,
    auth) fail fast.
    """
    name = type(exc).__name__
    return name in {
        "RateLimitError",
        "APITimeoutError",
        "APIConnectionError",
        "InternalServerError",
        "OverloadedError",
    }


default_llm_client = LLMClient()
