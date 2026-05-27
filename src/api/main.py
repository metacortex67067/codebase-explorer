"""
FastAPI application entry point.

Run with:
    uvicorn src.api.main:app --reload

Swagger UI: http://localhost:8000/docs
ReDoc:      http://localhost:8000/redoc

We intentionally keep this module tiny -- it only assembles the app from
parts defined elsewhere (routes, exception handlers). Putting business
logic here would make it harder to unit-test routes in isolation.
"""
from __future__ import annotations

from fastapi import FastAPI

from src.api.routes import register_exception_handlers, router


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
    return app


# Convenience module-level instance for `uvicorn src.api.main:app`.
app = create_app()
