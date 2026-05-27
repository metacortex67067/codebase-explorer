"""
Persistent metadata store for indexed repositories.

What lives in this store
------------------------

Vectors live in ChromaDB (under storage/chromadb/), but the *metadata*
about each indexed repo -- name, indexed_at timestamp, file/module/chunk
counts, dependency graph, per-module summaries -- needs its own home.

This module provides that home in the form of a tiny SQLite database with
a single `repos` table. We could have used a JSON file, but:

  * SQLite gives us atomicity for free (write or roll back, no half-written
    files when the process is killed).
  * SQLite is in stdlib -- no extra dependency for a coursework project.
  * Looking up one repo by id is O(log n) and obvious; with JSON we'd be
    re-reading and re-writing the whole document on every change.

Why store JSON blobs (dependency_graph, summaries) inside a single column
rather than normalising into separate tables? Because we always read and
write them together with the parent repo row, and there are no queries
that need to filter by module name across repos. A blob keeps the schema
trivial.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.core.config import settings
from src.core.models import ModuleSummary, RepoIndex


_SCHEMA = """
CREATE TABLE IF NOT EXISTS repos (
    repo_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    indexed_at TEXT NOT NULL,
    file_count INTEGER NOT NULL,
    module_count INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL,
    dependency_graph_json TEXT NOT NULL DEFAULT '{}',
    module_summaries_json TEXT NOT NULL DEFAULT '[]',
    modules_json TEXT NOT NULL DEFAULT '[]'
);
"""


class RepoStore:
    """SQLite-backed CRUD store for repo metadata."""

    def __init__(self, storage_dir: Optional[Path] = None) -> None:
        # We keep our own little file inside the same storage_dir that
        # ChromaDB uses, so "delete the storage directory" cleans everything.
        base = Path(storage_dir or settings.storage_dir)
        base.mkdir(parents=True, exist_ok=True)
        self._db_path = base / "repos.sqlite3"
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        # Each call opens its own connection -- cheap with SQLite, and avoids
        # any thread-affinity gotchas if RepoService is shared across
        # FastAPI request handlers.
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    # ----- CRUD -----------------------------------------------------------

    def upsert(
        self,
        repo_id: str,
        name: str,
        file_count: int,
        module_count: int,
        chunk_count: int,
        dependency_graph: dict[str, list[str]],
        module_summaries: list[ModuleSummary],
        modules: list[dict],
        indexed_at: Optional[str] = None,
    ) -> RepoIndex:
        """Insert or replace one repo row and return the RepoIndex snapshot.

        `modules` is the serialised list of full Module objects (as plain
        dicts) -- needed so the API can serve `/modules/{path}` details
        without re-parsing source files.
        """
        ts = indexed_at or datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO repos (
                    repo_id, name, indexed_at,
                    file_count, module_count, chunk_count,
                    dependency_graph_json, module_summaries_json, modules_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(repo_id) DO UPDATE SET
                    name = excluded.name,
                    indexed_at = excluded.indexed_at,
                    file_count = excluded.file_count,
                    module_count = excluded.module_count,
                    chunk_count = excluded.chunk_count,
                    dependency_graph_json = excluded.dependency_graph_json,
                    module_summaries_json = excluded.module_summaries_json,
                    modules_json = excluded.modules_json
                """,
                (
                    repo_id, name, ts,
                    file_count, module_count, chunk_count,
                    json.dumps(dependency_graph),
                    json.dumps([s.model_dump() for s in module_summaries]),
                    json.dumps(modules),
                ),
            )
        return RepoIndex(
            repo_id=repo_id,
            name=name,
            file_count=file_count,
            module_count=module_count,
            chunk_count=chunk_count,
            indexed_at=ts,
        )

    def get(self, repo_id: str) -> Optional[RepoIndex]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM repos WHERE repo_id = ?", (repo_id,)
            ).fetchone()
        return _row_to_index(row) if row else None

    def list_all(self) -> list[RepoIndex]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM repos ORDER BY indexed_at DESC"
            ).fetchall()
        return [_row_to_index(r) for r in rows]

    def get_dependency_graph(self, repo_id: str) -> Optional[dict[str, list[str]]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT dependency_graph_json FROM repos WHERE repo_id = ?",
                (repo_id,),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row["dependency_graph_json"])

    def get_module_summaries(self, repo_id: str) -> Optional[list[ModuleSummary]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT module_summaries_json FROM repos WHERE repo_id = ?",
                (repo_id,),
            ).fetchone()
        if row is None:
            return None
        raw = json.loads(row["module_summaries_json"])
        return [ModuleSummary(**item) for item in raw]

    def get_modules(self, repo_id: str) -> Optional[list[dict]]:
        """Return the raw serialised Module dicts for `repo_id`.

        We keep them as dicts (not Module objects) at this layer because
        the caller may want to selectively pick one by relative_path
        without paying full pydantic-validation cost for every entry.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT modules_json FROM repos WHERE repo_id = ?", (repo_id,)
            ).fetchone()
        if row is None:
            return None
        return json.loads(row["modules_json"])

    def delete(self, repo_id: str) -> bool:
        """Remove a repo row. Returns True if a row was actually deleted."""
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM repos WHERE repo_id = ?", (repo_id,)
            )
            return cur.rowcount > 0


# ----- internals ------------------------------------------------------------

def _row_to_index(row: sqlite3.Row) -> RepoIndex:
    return RepoIndex(
        repo_id=row["repo_id"],
        name=row["name"],
        file_count=row["file_count"],
        module_count=row["module_count"],
        chunk_count=row["chunk_count"],
        indexed_at=row["indexed_at"],
    )
