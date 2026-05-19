"""Shared pytest fixtures.

We build a tiny synthetic Python project on disk so tests across
ingestion/parsing/indexing share the same input and stay deterministic.
"""
from __future__ import annotations

from pathlib import Path

import pytest


SAMPLE_FILES: dict[str, str] = {
    "mypkg/__init__.py": '"""Top-level package docstring."""\n',
    "mypkg/utils.py": '''
"""Utility helpers."""
from typing import Iterable


def add(a: int, b: int) -> int:
    """Return a + b."""
    return a + b


async def fetch(url: str) -> str:
    """Pretend HTTP fetch."""
    return url


class Counter:
    """A trivial counter class."""

    def __init__(self, start: int = 0) -> None:
        self.value = start

    def inc(self, amount: int = 1) -> int:
        """Increment and return new value."""
        self.value += amount
        return self.value
''',
    "mypkg/service.py": '''
"""Business logic using utils."""
from mypkg.utils import Counter, add


def run() -> int:
    """Entry point."""
    c = Counter()
    c.inc()
    return add(c.value, 41)
''',
    "mypkg/broken.py": "def bad(:\n",  # deliberate syntax error
    "tests_inside/test_dummy.py": "def test_ok(): assert True\n",
    "__pycache__/cached.py": "# should be excluded\n",
    "README.md": "not a python file\n",
}


@pytest.fixture()
def sample_repo(tmp_path: Path) -> Path:
    """Create the sample tree under tmp_path and return its root."""
    for rel_path, content in SAMPLE_FILES.items():
        target = tmp_path / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return tmp_path


@pytest.fixture()
def sample_zip(tmp_path: Path, sample_repo: Path) -> Path:
    """Zip the sample repo and return the zip path."""
    import shutil
    zip_base = tmp_path.parent / "sample_repo_archive"
    zip_path = shutil.make_archive(str(zip_base), "zip", root_dir=sample_repo)
    return Path(zip_path)
