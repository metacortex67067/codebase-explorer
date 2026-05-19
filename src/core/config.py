"""
Centralised configuration.

All settings come from environment variables (or a .env file).
This keeps secrets out of code and makes the service 12-factor-friendly.
"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Storage
    storage_dir: Path = Path("./storage")
    max_repo_size_mb: int = 50
    max_files_per_repo: int = 2000

    # Parsing
    excluded_dirs: tuple[str, ...] = (
        "__pycache__", ".git", ".venv", "venv", "env",
        "node_modules", "dist", "build", ".pytest_cache",
        ".mypy_cache", ".tox", "site-packages",
    )

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_overlap_lines: int = 3

    # LLM
    anthropic_api_key: str = ""
    llm_model: str = "claude-haiku-4-5-20251001"
    llm_max_tokens: int = 1024
    rag_top_k: int = 5

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000


settings = Settings()
