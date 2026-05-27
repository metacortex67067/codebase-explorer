"""
MCP (Model Context Protocol) server for Codebase Explorer.

Run with:
    python -m src.mcp_server.server

This starts a stdio-transport MCP server that Claude Desktop / Cursor /
any MCP-compatible client can connect to. Sample claude_desktop_config.json
entry:

    {
      "mcpServers": {
        "codebase-explorer": {
          "command": "/abs/path/to/.venv/bin/python",
          "args": ["-m", "src.mcp_server.server"],
          "cwd": "/abs/path/to/codebase_explorer"
        }
      }
    }

This module is intentionally a thin adapter:
  * declares the tools (schemas come from `tools.TOOL_SCHEMAS`),
  * dispatches calls by name to `tools.dispatch_tool`,
  * wraps the JSON result in a TextContent envelope.

All real logic lives in `tools.py`, which is exercised by unit tests
without needing the async MCP runtime.
"""
from __future__ import annotations

import asyncio

import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server

from src.core.repo_service import RepoService
from src.mcp_server.tools import TOOL_SCHEMAS, dispatch_tool


SERVER_NAME = "codebase-explorer"
SERVER_VERSION = "0.1.0"


def build_server(service: RepoService | None = None) -> Server:
    """Create an MCP Server with our tools registered.

    `service` is injectable so a future integration test (if we ever add
    one over the real transport) could pass in a stub. Production uses
    the default singleton.
    """
    service = service or RepoService()
    app: Server = Server(SERVER_NAME)

    @app.list_tools()
    async def _list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name=schema["name"],
                description=schema["description"],
                inputSchema=schema["inputSchema"],
            )
            for schema in TOOL_SCHEMAS
        ]

    @app.call_tool()
    async def _call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        # Dispatch is sync; the MCP runtime is async. For our workloads
        # (LLM calls, sqlite reads) the sync path is fine -- if it ever
        # becomes a problem, run in a threadpool here.
        payload = dispatch_tool(service, name, arguments)
        return [types.TextContent(type="text", text=payload)]

    return app


async def main() -> None:
    """Run the server over stdio. Entry point for `python -m`."""
    app = build_server()
    init_options = InitializationOptions(
        server_name=SERVER_NAME,
        server_version=SERVER_VERSION,
        capabilities=app.get_capabilities(
            notification_options=NotificationOptions(),
            experimental_capabilities={},
        ),
    )
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, init_options)


if __name__ == "__main__":
    asyncio.run(main())
