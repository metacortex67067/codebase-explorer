"""Tests for the MCP-tool layer.

We exercise the synchronous `dispatch_tool` function with a fake
RepoService, verifying:
  * each tool name routes to the right handler,
  * arguments are validated,
  * service errors are surfaced as `{"error": ...}` JSON.

We do NOT spin up the async MCP runtime here -- that's just an adapter
in server.py over an SDK we don't own. A future integration test could
drive it over a real stdio transport if needed.
"""
from __future__ import annotations

import json
from typing import Optional

import pytest

from src.core.models import (
    ModuleSummary,
    QAResponse,
    QASource,
    RepoIndex,
)
from src.core.repo_service import RepoServiceError
from src.mcp_server.tools import (
    TOOL_DISPATCH,
    TOOL_SCHEMAS,
    dispatch_tool,
)


# ----- Fake service (mirrors the API test fake, smaller scope) -------------

class FakeService:
    def __init__(self) -> None:
        self.repos: dict[str, RepoIndex] = {}

    def create_repo_from_git(self, git_url: str, name: Optional[str] = None) -> RepoIndex:
        idx = RepoIndex(
            repo_id="git1", name=name or "repo",
            file_count=1, module_count=1, chunk_count=2,
            indexed_at="2025-01-01T00:00:00",
        )
        self.repos[idx.repo_id] = idx
        return idx

    def create_repo_from_directory(self, directory, name) -> RepoIndex:
        idx = RepoIndex(
            repo_id="dir1", name=name,
            file_count=1, module_count=1, chunk_count=2,
            indexed_at="2025-01-01T00:00:00",
        )
        self.repos[idx.repo_id] = idx
        return idx

    def list_repos(self) -> list[RepoIndex]:
        return list(self.repos.values())

    def get_modules(self, repo_id) -> list[ModuleSummary]:
        if repo_id not in self.repos:
            raise RepoServiceError(f"Unknown repo_id: {repo_id}")
        return [ModuleSummary(
            module_path="a.py", qualified_name="a",
            summary="s", key_responsibilities=["r"],
        )]

    def get_module_detail(self, repo_id, module_path) -> dict:
        if repo_id not in self.repos:
            raise RepoServiceError(f"Unknown repo_id: {repo_id}")
        if module_path != "a.py":
            raise RepoServiceError(f"Module not found: {module_path}")
        return {
            "module": {"relative_path": "a.py", "qualified_name": "a"},
            "summary": None,
        }

    def get_dependency_graph(self, repo_id) -> dict:
        if repo_id not in self.repos:
            raise RepoServiceError(f"Unknown repo_id: {repo_id}")
        return {"a": [], "b": ["a"]}

    def ask(self, repo_id, question) -> QAResponse:
        if repo_id not in self.repos:
            raise RepoServiceError(f"Unknown repo_id: {repo_id}")
        return QAResponse(
            question=question, answer="42",
            sources=[QASource(
                module_path="a.py", qualified_name="a.f",
                line_start=1, line_end=2, snippet="def f(): pass",
            )],
        )

    def delete_repo(self, repo_id) -> bool:
        return self.repos.pop(repo_id, None) is not None


@pytest.fixture()
def service() -> FakeService:
    return FakeService()


# ----- Tool schemas --------------------------------------------------------

class TestToolSchemas:
    def test_every_dispatched_tool_has_a_schema(self) -> None:
        dispatch_names = set(TOOL_DISPATCH.keys())
        schema_names = {s["name"] for s in TOOL_SCHEMAS}
        assert dispatch_names == schema_names

    def test_required_args_listed_in_schema(self) -> None:
        by_name = {s["name"]: s for s in TOOL_SCHEMAS}
        assert by_name["list_modules"]["inputSchema"]["required"] == ["repo_id"]
        assert by_name["explain_module"]["inputSchema"]["required"] == [
            "repo_id", "module_path",
        ]


# ----- dispatch_tool happy paths -------------------------------------------

class TestDispatchHappyPath:
    def test_index_repo_from_git(self, service: FakeService) -> None:
        out = dispatch_tool(service, "index_repo", {"git_url": "https://x"})
        data = json.loads(out)
        assert data["repo_id"] == "git1"

    def test_index_repo_from_local_path(self, service: FakeService) -> None:
        out = dispatch_tool(service, "index_repo",
                            {"local_path": "/tmp/x", "name": "x"})
        data = json.loads(out)
        assert data["repo_id"] == "dir1"

    def test_list_repos(self, service: FakeService) -> None:
        dispatch_tool(service, "index_repo", {"git_url": "x"})
        out = dispatch_tool(service, "list_repos", {})
        data = json.loads(out)
        assert len(data["repos"]) == 1
        assert data["repos"][0]["repo_id"] == "git1"

    def test_list_modules(self, service: FakeService) -> None:
        dispatch_tool(service, "index_repo", {"git_url": "x"})
        out = dispatch_tool(service, "list_modules", {"repo_id": "git1"})
        data = json.loads(out)
        assert len(data["modules"]) == 1
        assert data["modules"][0]["qualified_name"] == "a"

    def test_explain_module(self, service: FakeService) -> None:
        dispatch_tool(service, "index_repo", {"git_url": "x"})
        out = dispatch_tool(service, "explain_module",
                            {"repo_id": "git1", "module_path": "a.py"})
        data = json.loads(out)
        assert data["module"]["relative_path"] == "a.py"

    def test_get_dependency_graph(self, service: FakeService) -> None:
        dispatch_tool(service, "index_repo", {"git_url": "x"})
        out = dispatch_tool(service, "get_dependency_graph", {"repo_id": "git1"})
        data = json.loads(out)
        assert data["graph"] == {"a": [], "b": ["a"]}

    def test_ask_question(self, service: FakeService) -> None:
        dispatch_tool(service, "index_repo", {"git_url": "x"})
        out = dispatch_tool(service, "ask_question",
                            {"repo_id": "git1", "question": "what?"})
        data = json.loads(out)
        assert data["answer"] == "42"
        assert data["sources"][0]["qualified_name"] == "a.f"

    def test_delete_repo(self, service: FakeService) -> None:
        dispatch_tool(service, "index_repo", {"git_url": "x"})
        out = dispatch_tool(service, "delete_repo", {"repo_id": "git1"})
        data = json.loads(out)
        assert data["deleted"] is True


# ----- dispatch_tool error paths -------------------------------------------

class TestDispatchErrors:
    def test_unknown_tool_returns_error(self, service: FakeService) -> None:
        data = json.loads(dispatch_tool(service, "nope", {}))
        assert "error" in data
        assert "Unknown tool" in data["error"]

    def test_index_repo_requires_one_source(self, service: FakeService) -> None:
        data = json.loads(dispatch_tool(service, "index_repo", {}))
        assert "error" in data
        # Both git_url and local_path missing -> error
        data2 = json.loads(dispatch_tool(
            service, "index_repo",
            {"git_url": "x", "local_path": "/y"},
        ))
        assert "error" in data2

    def test_missing_required_argument(self, service: FakeService) -> None:
        data = json.loads(dispatch_tool(service, "list_modules", {}))
        assert "Missing required argument: repo_id" in data["error"]

    def test_repo_service_error_surfaces_as_error(self, service: FakeService) -> None:
        data = json.loads(dispatch_tool(
            service, "list_modules", {"repo_id": "nope"},
        ))
        assert "Unknown repo_id" in data["error"]

    def test_delete_unknown_repo_is_error(self, service: FakeService) -> None:
        data = json.loads(dispatch_tool(
            service, "delete_repo", {"repo_id": "nope"},
        ))
        assert "Unknown repo_id" in data["error"]


# ----- server.py wiring (minimal, just verifies build_server doesn't blow up) --

class TestServerBuild:
    def test_build_server_returns_server_with_tools(self, service: FakeService) -> None:
        from src.mcp_server.server import build_server

        server = build_server(service=service)
        # Server name set correctly
        assert server.name == "codebase-explorer"
