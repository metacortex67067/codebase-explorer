"""
Pure-logic implementations of the MCP tools.

Why split this from `server.py`?

The MCP SDK wires tools through an async `Server` runtime that listens
on stdio. We keep the *what each tool does* logic here as ordinary
synchronous functions, so:

  * Tests can call them directly with a fake RepoService -- no need to
    spawn an MCP process, no asyncio plumbing in the test code.
  * `server.py` becomes a thin adapter (declare schemas, dispatch by
    name, format text output), which is hard to break and easy to read.

Each tool function takes a `RepoService` and a dict of `arguments` (so
the call signature matches what the MCP runtime hands us) and returns a
JSON-serialisable result. `server.py` is responsible for turning that
result into the MCP TextContent envelope.
"""
from __future__ import annotations

import json
from typing import Any

from src.core.repo_service import RepoService, RepoServiceError
from src.ingestion.loader import IngestionError


# --- Tool schemas (also used by server.py to advertise the tools) ---------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "index_repo",
        "description": (
            "Index a Python repository so it can be queried. Provide either "
            "`git_url` (a clone URL) or `local_path` (a path to a directory "
            "on disk). Returns the new repo_id and indexing stats."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "git_url": {"type": "string"},
                "local_path": {"type": "string"},
                "name": {"type": "string"},
            },
            "required": [],
        },
    },
    {
        "name": "list_repos",
        "description": "List all repositories that have been indexed.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "list_modules",
        "description": (
            "List modules in an indexed repo, each with an LLM-generated "
            "summary."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"repo_id": {"type": "string"}},
            "required": ["repo_id"],
        },
    },
    {
        "name": "explain_module",
        "description": (
            "Return the parsed structure (classes, functions, docstrings, "
            "imports) of one module, together with its LLM summary."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo_id": {"type": "string"},
                "module_path": {
                    "type": "string",
                    "description": "Repo-relative path, e.g. pkg/sub/mod.py",
                },
            },
            "required": ["repo_id", "module_path"],
        },
    },
    {
        "name": "get_dependency_graph",
        "description": (
            "Return the internal import graph of the indexed repo as an "
            "adjacency list `{module: [imported_module, ...]}`."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"repo_id": {"type": "string"}},
            "required": ["repo_id"],
        },
    },
    {
        "name": "ask_question",
        "description": (
            "Ask a natural-language question about the indexed repo. Returns "
            "an answer plus citations (module path, line range, snippet) of "
            "the code chunks the answer is based on."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo_id": {"type": "string"},
                "question": {"type": "string"},
            },
            "required": ["repo_id", "question"],
        },
    },
    {
        "name": "delete_repo",
        "description": "Remove an indexed repo (both vectors and metadata).",
        "inputSchema": {
            "type": "object",
            "properties": {"repo_id": {"type": "string"}},
            "required": ["repo_id"],
        },
    },
]


# --- Tool implementations ---------------------------------------------------

class ToolError(Exception):
    """Domain error from a tool call -- rendered as an error TextContent."""


def index_repo(service: RepoService, arguments: dict) -> dict:
    """Dispatch to git or directory ingestion based on which arg is given."""
    git_url = arguments.get("git_url")
    local_path = arguments.get("local_path")
    name = arguments.get("name")

    if bool(git_url) == bool(local_path):
        # Either both or neither -- both cases ambiguous, refuse cleanly.
        raise ToolError(
            "Provide exactly one of `git_url` or `local_path`."
        )

    try:
        if git_url:
            idx = service.create_repo_from_git(git_url, name=name)
        else:
            idx = service.create_repo_from_directory(local_path, name=name or local_path)
    except IngestionError as e:
        raise ToolError(f"Ingestion failed: {e}") from e
    return idx.model_dump()


def list_repos(service: RepoService, arguments: dict) -> dict:
    return {"repos": [r.model_dump() for r in service.list_repos()]}


def list_modules(service: RepoService, arguments: dict) -> dict:
    repo_id = _require(arguments, "repo_id")
    try:
        summaries = service.get_modules(repo_id)
    except RepoServiceError as e:
        raise ToolError(str(e)) from e
    return {"modules": [s.model_dump() for s in summaries]}


def explain_module(service: RepoService, arguments: dict) -> dict:
    repo_id = _require(arguments, "repo_id")
    module_path = _require(arguments, "module_path")
    try:
        return service.get_module_detail(repo_id, module_path)
    except RepoServiceError as e:
        raise ToolError(str(e)) from e


def get_dependency_graph(service: RepoService, arguments: dict) -> dict:
    repo_id = _require(arguments, "repo_id")
    try:
        graph = service.get_dependency_graph(repo_id)
    except RepoServiceError as e:
        raise ToolError(str(e)) from e
    return {"graph": graph}


def ask_question(service: RepoService, arguments: dict) -> dict:
    repo_id = _require(arguments, "repo_id")
    question = _require(arguments, "question")
    try:
        resp = service.ask(repo_id, question)
    except RepoServiceError as e:
        raise ToolError(str(e)) from e
    return resp.model_dump()


def delete_repo(service: RepoService, arguments: dict) -> dict:
    repo_id = _require(arguments, "repo_id")
    deleted = service.delete_repo(repo_id)
    if not deleted:
        raise ToolError(f"Unknown repo_id: {repo_id}")
    return {"deleted": True, "repo_id": repo_id}


# --- Dispatch table --------------------------------------------------------

TOOL_DISPATCH = {
    "index_repo": index_repo,
    "list_repos": list_repos,
    "list_modules": list_modules,
    "explain_module": explain_module,
    "get_dependency_graph": get_dependency_graph,
    "ask_question": ask_question,
    "delete_repo": delete_repo,
}


def dispatch_tool(service: RepoService, name: str, arguments: dict) -> str:
    """Run the named tool and return its result as a JSON string.

    Returning a string keeps the boundary with the MCP runtime trivial:
    server.py wraps the string in a TextContent and is done. Errors are
    rendered as JSON too, so the client always gets parseable output.
    """
    handler = TOOL_DISPATCH.get(name)
    if handler is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        result = handler(service, arguments or {})
    except ToolError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:  # noqa: BLE001 -- last-resort safety net
        return json.dumps({"error": f"{type(e).__name__}: {e}"})
    return json.dumps(result, ensure_ascii=False, indent=2)


# --- helpers ---------------------------------------------------------------

def _require(arguments: dict, key: str) -> Any:
    """Validate that `key` is present and non-empty in `arguments`."""
    value = arguments.get(key)
    if value is None or value == "":
        raise ToolError(f"Missing required argument: {key}")
    return value
