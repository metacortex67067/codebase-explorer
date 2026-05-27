"""Tests for the FastAPI layer.

We use FastAPI's TestClient and override `get_repo_service` with a fake
RepoService so we test routing / serialisation / status codes without
spinning up the real ingestion + embedding stack.
"""
from __future__ import annotations

from typing import Optional

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.routes import get_repo_service
from src.core.models import (
    ModuleSummary,
    QAResponse,
    QASource,
    RepoIndex,
)
from src.core.repo_service import RepoServiceError
from src.ingestion.loader import IngestionError


# ----- Fake service ---------------------------------------------------------

class FakeRepoService:
    """In-memory stand-in for RepoService -- records calls, returns canned data."""

    def __init__(self) -> None:
        self.repos: dict[str, RepoIndex] = {}
        self.calls: list[tuple[str, tuple, dict]] = []
        self.raise_on_create: Optional[Exception] = None
        self.raise_on_ask: Optional[Exception] = None

    def _record(self, method, *args, **kwargs):
        self.calls.append((method, args, kwargs))

    # Indexing
    def create_repo_from_zip_bytes(self, zip_bytes: bytes, name: str) -> RepoIndex:
        self._record("create_repo_from_zip_bytes", zip_bytes, name=name)
        if self.raise_on_create:
            raise self.raise_on_create
        idx = RepoIndex(
            repo_id="zip-1", name=name,
            file_count=1, module_count=1, chunk_count=1,
            indexed_at="2025-01-01T00:00:00",
        )
        self.repos[idx.repo_id] = idx
        return idx

    def create_repo_from_git(self, git_url: str, name: Optional[str] = None) -> RepoIndex:
        self._record("create_repo_from_git", git_url, name=name)
        if self.raise_on_create:
            raise self.raise_on_create
        idx = RepoIndex(
            repo_id="git-1", name=name or "repo",
            file_count=2, module_count=2, chunk_count=4,
            indexed_at="2025-01-01T00:00:00",
        )
        self.repos[idx.repo_id] = idx
        return idx

    # Reads
    def get_repo(self, repo_id: str) -> RepoIndex:
        if repo_id not in self.repos:
            raise RepoServiceError(f"Unknown repo_id: {repo_id}")
        return self.repos[repo_id]

    def list_repos(self) -> list[RepoIndex]:
        return list(self.repos.values())

    def get_modules(self, repo_id: str) -> list[ModuleSummary]:
        if repo_id not in self.repos:
            raise RepoServiceError(f"Unknown repo_id: {repo_id}")
        return [
            ModuleSummary(
                module_path="a/b.py", qualified_name="a.b",
                summary="s", key_responsibilities=["r1"],
            )
        ]

    def get_module_detail(self, repo_id: str, module_path: str) -> dict:
        if repo_id not in self.repos:
            raise RepoServiceError(f"Unknown repo_id: {repo_id}")
        if module_path != "a/b.py":
            raise RepoServiceError(f"Module not found: {module_path}")
        return {
            "module": {"relative_path": "a/b.py", "qualified_name": "a.b"},
            "summary": {
                "module_path": "a/b.py",
                "qualified_name": "a.b",
                "summary": "s",
                "key_responsibilities": [],
            },
        }

    def get_dependency_graph(self, repo_id: str) -> dict[str, list[str]]:
        if repo_id not in self.repos:
            raise RepoServiceError(f"Unknown repo_id: {repo_id}")
        return {"a.b": ["a.c"], "a.c": []}

    def ask(self, repo_id: str, question: str) -> QAResponse:
        if repo_id not in self.repos:
            raise RepoServiceError(f"Unknown repo_id: {repo_id}")
        if self.raise_on_ask:
            raise self.raise_on_ask
        return QAResponse(
            question=question,
            answer=f"Answer to: {question}",
            sources=[QASource(
                module_path="a/b.py", qualified_name="a.b.f",
                line_start=1, line_end=2, snippet="def f(): pass",
            )],
        )

    def delete_repo(self, repo_id: str) -> bool:
        if repo_id in self.repos:
            del self.repos[repo_id]
            return True
        return False


# ----- Fixtures -------------------------------------------------------------

@pytest.fixture()
def fake_service() -> FakeRepoService:
    return FakeRepoService()


@pytest.fixture()
def client(fake_service: FakeRepoService) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_repo_service] = lambda: fake_service
    return TestClient(app)


# ----- Tests ----------------------------------------------------------------

class TestHealth:
    def test_healthz(self, client: TestClient) -> None:
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestCreateRepo:
    def test_from_zip(
        self, client: TestClient, fake_service: FakeRepoService, sample_zip,
    ) -> None:
        with open(sample_zip, "rb") as f:
            resp = client.post(
                "/repos/from-zip",
                files={"file": ("sample.zip", f, "application/zip")},
            )
        assert resp.status_code == 201
        data = resp.json()
        assert data["repo_id"] == "zip-1"
        assert data["name"] == "sample"
        assert len(fake_service.calls) == 1

    def test_from_zip_ingestion_error_yields_400(
        self, client: TestClient, fake_service: FakeRepoService, sample_zip,
    ) -> None:
        fake_service.raise_on_create = IngestionError("too big")
        with open(sample_zip, "rb") as f:
            resp = client.post(
                "/repos/from-zip",
                files={"file": ("x.zip", f, "application/zip")},
            )
        assert resp.status_code == 400
        assert "too big" in resp.json()["detail"]

    def test_from_git(
        self, client: TestClient, fake_service: FakeRepoService,
    ) -> None:
        resp = client.post(
            "/repos/from-git",
            json={"git_url": "https://github.com/foo/bar.git", "name": "bar"},
        )
        assert resp.status_code == 201
        assert resp.json()["repo_id"] == "git-1"
        assert resp.json()["name"] == "bar"

    def test_from_git_optional_name(
        self, client: TestClient, fake_service: FakeRepoService,
    ) -> None:
        resp = client.post(
            "/repos/from-git",
            json={"git_url": "https://github.com/foo/bar.git"},
        )
        assert resp.status_code == 201
        # service was called with name=None -> falls back to derived name
        call = fake_service.calls[-1]
        assert call[2]["name"] is None


class TestReads:
    def test_list_repos_empty(self, client: TestClient) -> None:
        assert client.get("/repos").json() == []

    def test_get_repo_unknown_404(self, client: TestClient) -> None:
        resp = client.get("/repos/nope")
        assert resp.status_code == 404
        assert "Unknown" in resp.json()["detail"]

    def test_list_and_get_after_create(
        self, client: TestClient,
    ) -> None:
        client.post(
            "/repos/from-git",
            json={"git_url": "https://github.com/foo/bar.git"},
        )
        listed = client.get("/repos").json()
        assert len(listed) == 1
        repo_id = listed[0]["repo_id"]

        resp = client.get(f"/repos/{repo_id}")
        assert resp.status_code == 200
        assert resp.json()["repo_id"] == repo_id

    def test_list_modules(self, client: TestClient) -> None:
        client.post(
            "/repos/from-git",
            json={"git_url": "https://x", "name": "n"},
        )
        resp = client.get("/repos/git-1/modules")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["qualified_name"] == "a.b"

    def test_module_detail_path_with_slashes(self, client: TestClient) -> None:
        """:path converter should let module_path contain slashes."""
        client.post("/repos/from-git", json={"git_url": "x"})
        resp = client.get("/repos/git-1/modules/a/b.py")
        assert resp.status_code == 200
        body = resp.json()
        assert body["module"]["relative_path"] == "a/b.py"
        assert body["summary"]["qualified_name"] == "a.b"

    def test_module_detail_unknown_module(self, client: TestClient) -> None:
        client.post("/repos/from-git", json={"git_url": "x"})
        resp = client.get("/repos/git-1/modules/missing.py")
        assert resp.status_code == 404

    def test_dependencies(self, client: TestClient) -> None:
        client.post("/repos/from-git", json={"git_url": "x"})
        resp = client.get("/repos/git-1/dependencies")
        assert resp.status_code == 200
        assert resp.json() == {"graph": {"a.b": ["a.c"], "a.c": []}}


class TestAsk:
    def test_ask_returns_answer_with_sources(self, client: TestClient) -> None:
        client.post("/repos/from-git", json={"git_url": "x"})
        resp = client.post(
            "/repos/git-1/ask", json={"question": "what is this?"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["question"] == "what is this?"
        assert "Answer to" in body["answer"]
        assert len(body["sources"]) == 1
        assert body["sources"][0]["qualified_name"] == "a.b.f"

    def test_ask_unknown_repo_404(self, client: TestClient) -> None:
        resp = client.post(
            "/repos/missing/ask", json={"question": "?"},
        )
        assert resp.status_code == 404

    def test_ask_empty_question_422(self, client: TestClient) -> None:
        """Pydantic validation should reject an empty question."""
        client.post("/repos/from-git", json={"git_url": "x"})
        resp = client.post("/repos/git-1/ask", json={"question": ""})
        assert resp.status_code == 422


class TestDelete:
    def test_delete_then_list_is_empty(self, client: TestClient) -> None:
        client.post("/repos/from-git", json={"git_url": "x"})
        resp = client.delete("/repos/git-1")
        assert resp.status_code == 204
        assert client.get("/repos").json() == []

    def test_delete_unknown_404(self, client: TestClient) -> None:
        resp = client.delete("/repos/nope")
        assert resp.status_code == 404


class TestOpenAPI:
    def test_docs_endpoint_serves(self, client: TestClient) -> None:
        assert client.get("/docs").status_code == 200

    def test_openapi_schema_lists_main_routes(self, client: TestClient) -> None:
        spec = client.get("/openapi.json").json()
        paths = spec["paths"]
        assert "/healthz" in paths
        assert "/repos" in paths
        assert "/repos/from-zip" in paths
        assert "/repos/from-git" in paths
        assert "/repos/{repo_id}/ask" in paths
