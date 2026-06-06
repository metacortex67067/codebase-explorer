"""
FastAPI application entry point.

    uvicorn src.api.main:app --reload   # Swagger UI at /docs

Assembles the app from routes and exception handlers and serves the
single-page web UI at "/".
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

from src.api.routes import register_exception_handlers, router

# Bundled single-page web UI (src/web/index.html), served at "/".
WEB_INDEX = Path(__file__).resolve().parent.parent / "web" / "index.html"


def create_app() -> FastAPI:
    """Build the FastAPI instance.

    A factory rather than a module-level `app = FastAPI()` so tests can
    construct fresh instances with their own dependency overrides.
    """
    app = FastAPI(
        title="Codebase Explorer",
        description=(
            "AI-powered Python codebase understanding: ingest a repo, "
            "browse its parsed structure, and ask natural-language questions."
        ),
        version="0.1.0",
    )
    app.include_router(router)
    register_exception_handlers(app)

    @app.get("/", include_in_schema=False)
    async def web_ui():
        """Serve the bundled single-page UI (or a hint if it's missing)."""
        if WEB_INDEX.is_file():
            return FileResponse(WEB_INDEX)
        return JSONResponse(
            {"detail": "Web UI not found. API docs are at /docs."},
            status_code=404,
        )

    return app


# Convenience module-level instance for `uvicorn src.api.main:app`.
app = create_app()
