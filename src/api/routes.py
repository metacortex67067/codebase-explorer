"""
REST API routes.

Endpoint design follows resource-oriented HTTP conventions:

  POST   /repos/from-zip              -- multipart upload of a .zip archive
  POST   /repos/from-git              -- JSON body with `git_url`
  GET    /repos                       -- list all indexed repos
  GET    /repos/{repo_id}             -- one repo's metadata
  GET    /repos/{repo_id}/modules     -- modules + summaries
  GET    /repos/{repo_id}/modules/{module_path:path}
                                      -- one module's full detail
  GET    /repos/{repo_id}/dependencies -- adjacency-list graph
  POST   /repos/{repo_id}/ask         -- RAG Q&A
  DELETE /repos/{repo_id}             -- remove repo (vectors + metadata)
  GET    /healthz                     -- liveness probe

Notes on choices:

* Two separate "create" endpoints rather than one polymorphic one. A zip
  upload is a multipart request; a git URL is JSON. Forcing both into one
  endpoint complicates both the OpenAPI schema (Swagger UI becomes much
  less readable) and the client code -- it's cleaner to have two routes
  that do one thing each.

* `module_path:path` uses FastAPI's `:path` converter so that slashes
  inside the module path (e.g. `pkg/sub/mod.py`) survive routing.

* All RepoService errors are translated to HTTP 4xx/5xx via a single
  exception handler, so route bodies stay small and uniform.

* RepoService is supplied via FastAPI's dependency-injection system
  (`Depends(get_repo_service)`) -- this is what lets tests override
  the service with a fake without monkey-patching.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from src.api.schemas import (
    AskRequest,
    DependencyGraphResponse,
    HealthResponse,
    IndexFromGitRequest,
    ModuleDetailResponse,
)
from src.core.models import ModuleSummary, QAResponse, RepoIndex
from src.core.repo_service import RepoService, RepoServiceError
from src.ingestion.loader import IngestionError


# --- Dependency injection ---------------------------------------------------
#
# We hold a single RepoService at module level so it's reused across requests
# (this keeps the embedder model loaded). Tests override via
# app.dependency_overrides[get_repo_service] = lambda: <fake>.

_repo_service: RepoService | None = None


def get_repo_service() -> RepoService:
    """Lazily build a single RepoService and return it on every request.

    Lazy because importing this module shouldn't try to open SQLite /
    ChromaDB files (which would happen if we built RepoService at import
    time). FastAPI dependency overrides bypass this anyway, so tests
    never trigger the real construction.
    """
    global _repo_service
    if _repo_service is None:
        _repo_service = RepoService()
    return _repo_service


# --- Routes -----------------------------------------------------------------

router = APIRouter()


@router.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    """Liveness probe -- always cheap, never touches storage or LLM."""
    return HealthResponse(status="ok")


@router.post(
    "/repos/from-zip",
    response_model=RepoIndex,
    status_code=status.HTTP_201_CREATED,
)
async def create_repo_from_zip(
    file: UploadFile = File(...),
    service: RepoService = Depends(get_repo_service),
) -> RepoIndex:
    """Accept a multipart zip upload, index the repo, return its metadata."""
    name = (file.filename or "uploaded").rsplit(".", 1)[0]
    body = await file.read()
    return service.create_repo_from_zip_bytes(body, name=name)


@router.post(
    "/repos/from-git",
    response_model=RepoIndex,
    status_code=status.HTTP_201_CREATED,
)
def create_repo_from_git(
    payload: IndexFromGitRequest,
    service: RepoService = Depends(get_repo_service),
) -> RepoIndex:
    """Clone the given git URL, index the repo, return its metadata."""
    return service.create_repo_from_git(payload.git_url, name=payload.name)


@router.get("/repos", response_model=list[RepoIndex])
def list_repos(
    service: RepoService = Depends(get_repo_service),
) -> list[RepoIndex]:
    return service.list_repos()


@router.get("/repos/{repo_id}", response_model=RepoIndex)
def get_repo(
    repo_id: str,
    service: RepoService = Depends(get_repo_service),
) -> RepoIndex:
    return service.get_repo(repo_id)


@router.get("/repos/{repo_id}/modules", response_model=list[ModuleSummary])
def list_modules(
    repo_id: str,
    service: RepoService = Depends(get_repo_service),
) -> list[ModuleSummary]:
    return service.get_modules(repo_id)


@router.get(
    "/repos/{repo_id}/modules/{module_path:path}",
    response_model=ModuleDetailResponse,
)
def get_module_detail(
    repo_id: str,
    module_path: str,
    service: RepoService = Depends(get_repo_service),
) -> ModuleDetailResponse:
    detail = service.get_module_detail(repo_id, module_path)
    return ModuleDetailResponse(**detail)


@router.get(
    "/repos/{repo_id}/dependencies",
    response_model=DependencyGraphResponse,
)
def get_dependencies(
    repo_id: str,
    service: RepoService = Depends(get_repo_service),
) -> DependencyGraphResponse:
    return DependencyGraphResponse(graph=service.get_dependency_graph(repo_id))


@router.post("/repos/{repo_id}/ask", response_model=QAResponse)
def ask(
    repo_id: str,
    payload: AskRequest,
    service: RepoService = Depends(get_repo_service),
) -> QAResponse:
    return service.ask(repo_id, payload.question)


@router.delete("/repos/{repo_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_repo(
    repo_id: str,
    service: RepoService = Depends(get_repo_service),
):
    deleted = service.delete_repo(repo_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown repo_id: {repo_id}",
        )
    # 204 No Content -- explicit empty body
    return JSONResponse(status_code=status.HTTP_204_NO_CONTENT, content=None)


# --- Exception handlers (registered on the app, exported for main.py) ------

def register_exception_handlers(app) -> None:
    """Translate domain exceptions to clean HTTP responses.

    Keeping this here (rather than scattered try/except in each route)
    means new routes get the same error handling for free.
    """

    @app.exception_handler(RepoServiceError)
    async def _repo_service_error(_request, exc: RepoServiceError):
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": str(exc)},
        )

    @app.exception_handler(IngestionError)
    async def _ingestion_error(_request, exc: IngestionError):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": str(exc)},
        )
