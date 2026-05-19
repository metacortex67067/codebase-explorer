"""Tests for src.ingestion.loader."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.ingestion.loader import (
    IngestionError,
    load_from_directory,
    load_from_zip,
)


class TestLoadFromDirectory:
    def test_returns_only_python_files(self, sample_repo: Path) -> None:
        files = load_from_directory(sample_repo)
        suffixes = {Path(f.relative_path).suffix for f in files}
        assert suffixes == {".py"}

    def test_excludes_pycache(self, sample_repo: Path) -> None:
        files = load_from_directory(sample_repo)
        assert not any("__pycache__" in f.relative_path for f in files)

    def test_finds_expected_modules(self, sample_repo: Path) -> None:
        files = load_from_directory(sample_repo)
        paths = {f.relative_path for f in files}
        assert "mypkg/utils.py" in paths
        assert "mypkg/service.py" in paths
        assert "mypkg/__init__.py" in paths

    def test_excludes_readme(self, sample_repo: Path) -> None:
        files = load_from_directory(sample_repo)
        assert not any(f.relative_path.endswith(".md") for f in files)

    def test_uses_posix_paths(self, sample_repo: Path) -> None:
        files = load_from_directory(sample_repo)
        assert all("\\" not in f.relative_path for f in files)

    def test_empty_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(IngestionError):
            load_from_directory(tmp_path)

    def test_nonexistent_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(IngestionError):
            load_from_directory(tmp_path / "missing")


class TestLoadFromZip:
    def test_unpacks_and_loads(self, sample_zip: Path) -> None:
        files = load_from_zip(sample_zip)
        paths = {f.relative_path for f in files}
        assert "mypkg/utils.py" in paths
        assert "mypkg/service.py" in paths

    def test_rejects_non_zip_file(self, tmp_path: Path) -> None:
        fake = tmp_path / "fake.zip"
        fake.write_text("not a zip")
        with pytest.raises(IngestionError):
            load_from_zip(fake)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(IngestionError):
            load_from_zip(tmp_path / "nope.zip")
