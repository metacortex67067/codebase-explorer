"""
Repository ingestion.

Two entry points:
  - load_from_zip(zip_path) -- the user uploaded a zip archive
  - load_from_git(git_url)  -- the user gave a git URL

Both return a list[SourceFile] after walking the unpacked tree and
filtering out non-source files and excluded directories.

Decisions worth defending:
  * We refuse repos that exceed configured size/file-count caps -- prevents
    runaway memory use and unbounded indexing time.
  * We read files as UTF-8 with errors='replace' rather than skipping them,
    so a single weirdly-encoded file does not break ingestion.
  * The temp directory is the caller's responsibility (use tempfile.TemporaryDirectory).
"""
from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Iterator

from git import Repo

from src.core.config import settings
from src.core.models import Language, SourceFile


PYTHON_EXTENSIONS = {".py"}


class IngestionError(Exception):
    """Raised when a repo cannot be ingested (too big, corrupt, etc.)."""


def _is_excluded(path: Path, repo_root: Path) -> bool:
    """Return True if any parent directory is in the excluded set."""
    rel_parts = path.relative_to(repo_root).parts
    return any(part in settings.excluded_dirs for part in rel_parts)


def _iter_source_files(repo_root: Path) -> Iterator[Path]:
    """Yield .py files inside repo_root, skipping excluded directories."""
    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in PYTHON_EXTENSIONS:
            continue
        if _is_excluded(path, repo_root):
            continue
        yield path


def _read_source(path: Path, repo_root: Path) -> SourceFile:
    """Read a single source file, tolerating encoding issues."""
    content = path.read_text(encoding="utf-8", errors="replace")
    return SourceFile(
        relative_path=path.relative_to(repo_root).as_posix(),
        language=Language.PYTHON,
        size_bytes=path.stat().st_size,
        content=content,
    )


def _collect_files(repo_root: Path) -> list[SourceFile]:
    """Walk repo_root, applying size/count caps, returning SourceFile objects."""
    files: list[SourceFile] = []
    total_bytes = 0
    cap_bytes = settings.max_repo_size_mb * 1024 * 1024

    for path in _iter_source_files(repo_root):
        if len(files) >= settings.max_files_per_repo:
            raise IngestionError(
                f"Repository contains more than {settings.max_files_per_repo} source files"
            )
        size = path.stat().st_size
        total_bytes += size
        if total_bytes > cap_bytes:
            raise IngestionError(
                f"Repository source size exceeds {settings.max_repo_size_mb} MB"
            )
        files.append(_read_source(path, repo_root))

    if not files:
        raise IngestionError("No Python source files found in repository")
    return files


def load_from_directory(directory: Path) -> list[SourceFile]:
    """Load source files from an already-unpacked directory.

    Useful for tests and for the other loaders to share logic.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise IngestionError(f"Not a directory: {directory}")
    return _collect_files(directory)


def load_from_zip(zip_path: Path, extract_to: Path | None = None) -> list[SourceFile]:
    """Unpack a zip archive and load its source files.

    If extract_to is None, a temporary directory is used and cleaned up.
    """
    zip_path = Path(zip_path)
    if not zip_path.is_file():
        raise IngestionError(f"Zip file not found: {zip_path}")
    if not zipfile.is_zipfile(zip_path):
        raise IngestionError(f"Not a valid zip archive: {zip_path}")

    tmp_dir: Path | None = None
    if extract_to is None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="codeexp_"))
        extract_to = tmp_dir

    try:
        with zipfile.ZipFile(zip_path) as zf:
            # Guard against zip-slip: refuse entries that escape the target dir
            for member in zf.namelist():
                target = (extract_to / member).resolve()
                if not str(target).startswith(str(extract_to.resolve())):
                    raise IngestionError(f"Unsafe path in zip: {member}")
            zf.extractall(extract_to)

        # Some archives wrap everything in a single top-level dir; descend if so
        root = _resolve_repo_root(extract_to)
        return _collect_files(root)
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def load_from_git(git_url: str, clone_to: Path | None = None) -> list[SourceFile]:
    """Shallow-clone a git repo and load its source files."""
    tmp_dir: Path | None = None
    if clone_to is None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="codeexp_git_"))
        clone_to = tmp_dir

    try:
        Repo.clone_from(git_url, clone_to, depth=1)
        return _collect_files(Path(clone_to))
    except Exception as exc:
        raise IngestionError(f"Failed to clone {git_url}: {exc}") from exc
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def _resolve_repo_root(extract_to: Path) -> Path:
    """If the archive has a single top-level dir, return it; else return extract_to."""
    entries = [p for p in extract_to.iterdir() if not p.name.startswith(".")]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return extract_to
