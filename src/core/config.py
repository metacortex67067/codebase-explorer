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
    # Provider selects how we talk to the model:
    #   "ollama"    -> local, free, offline (OpenAI-compatible endpoint)
    #   "openai"    -> any OpenAI-compatible cloud (Groq, OpenRouter, ...)
    #   "anthropic" -> Anthropic Claude (paid)
    # Default is ollama so the project runs free and offline out of the box.
    llm_provider: str = "ollama"

    # Used by the "ollama"/"openai" providers (OpenAI-compatible HTTP API).
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"  # Ollama ignores it; cloud providers need a real key

    # Used by the "anthropic" provider.
    anthropic_api_key: str = ""

    # Model name as the chosen provider expects it.
    #   ollama:    "qwen2.5-coder:7b", "deepseek-coder-v2", ...
    #   groq:      "llama-3.3-70b-versatile", "qwen-2.5-coder-32b"
    #   anthropic: "claude-haiku-4-5-20251001"
    llm_model: str = "qwen2.5-coder:7b"
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.0
    rag_top_k: int = 5

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000


settings = Settings()
