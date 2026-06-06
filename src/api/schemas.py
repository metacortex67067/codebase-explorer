"""
Request/response schemas for the REST API.

Kept separate from ``core.models`` (the internal cross-layer contract) so the
HTTP surface can evolve independently. Responses that are already clean domain
models (RepoIndex, ModuleSummary, QAResponse) are returned directly.
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
