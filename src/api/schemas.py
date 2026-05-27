"""
Request and response schemas for the REST API.

Why a separate module from core/models.py?

The pydantic models in `src.core.models` are the *internal* contract
between layers. The API surface is allowed to diverge from them -- for
example, a POST body for indexing only needs a `git_url`, not the full
RepoIndex.

We keep wire-level schemas here so:
  * Changing the API shape doesn't ripple through internal modules.
  * Internal models can grow fields (debugging metadata, cache keys) that
    we don't necessarily want to expose over HTTP.

For responses that are already-clean pydantic models (RepoIndex,
ModuleSummary, QAResponse), the route can return them directly; FastAPI
serialises them as JSON via pydantic.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class IndexFromGitRequest(BaseModel):
    """Body of `POST /repos/from-git`."""
    git_url: str = Field(..., description="HTTPS or SSH git clone URL")
    name: Optional[str] = Field(
        None,
        description="Display name; derived from the URL if omitted",
    )


class AskRequest(BaseModel):
    """Body of `POST /repos/{repo_id}/ask`."""
    question: str = Field(..., min_length=1)


class ModuleDetailResponse(BaseModel):
    """Wire shape for `GET /repos/{repo_id}/modules/{module_path}`.

    We don't reuse `Module` directly because we want the LLM summary
    next to it -- a denormalised view that's most useful for the UI.
    """
    module: dict
    summary: Optional[dict]


class DependencyGraphResponse(BaseModel):
    """Wire shape for `GET /repos/{repo_id}/dependencies`."""
    graph: dict[str, list[str]]


class HealthResponse(BaseModel):
    status: str = "ok"


class ErrorResponse(BaseModel):
    """Standard error envelope -- used for FastAPI custom exception handlers."""
    detail: str
