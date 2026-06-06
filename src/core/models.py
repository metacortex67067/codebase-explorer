"""
Domain models for the Codebase Explorer.

These pydantic models are the contract between layers:
- ingestion produces SourceFile objects
- parsing produces Module / FunctionInfo / ClassInfo
- indexing consumes CodeChunk
- qa returns QAResponse
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Language(str, Enum):
    """Supported source languages. Only Python is implemented in v1."""
    PYTHON = "python"


class SourceFile(BaseModel):
    """A single source file discovered during ingestion."""
    relative_path: str = Field(..., description="Path relative to repo root, POSIX-style")
    language: Language
    size_bytes: int
    content: str


class FunctionInfo(BaseModel):
    """Top-level or method function extracted from AST."""
    name: str
    qualified_name: str = Field(..., description="module.Class.method or module.function")
    line_start: int
    line_end: int
    docstring: Optional[str] = None
    parameters: list[str] = Field(default_factory=list)
    is_async: bool = False
    decorators: list[str] = Field(default_factory=list)


class ClassInfo(BaseModel):
    """A class extracted from AST."""
    name: str
    qualified_name: str
    line_start: int
    line_end: int
    docstring: Optional[str] = None
    bases: list[str] = Field(default_factory=list)
    methods: list[FunctionInfo] = Field(default_factory=list)


class Module(BaseModel):
    """A parsed source module (one .py file)."""
    relative_path: str
    qualified_name: str = Field(..., description="Dotted import path, e.g. pkg.subpkg.module")
    docstring: Optional[str] = None
    imports: list[str] = Field(default_factory=list, description="Imported module names")
    functions: list[FunctionInfo] = Field(default_factory=list)
    classes: list[ClassInfo] = Field(default_factory=list)
    line_count: int = 0


class CodeChunk(BaseModel):
    """A chunk of code prepared for embedding."""
    chunk_id: str
    repo_id: str
    module_path: str
    qualified_name: str = Field(..., description="What this chunk represents (module / class / function)")
    chunk_type: str = Field(..., description="One of: module_header, function, class, class_method")
    content: str
    line_start: int
    line_end: int


class ModuleSummary(BaseModel):
    """LLM-generated explanation of a module."""
    module_path: str
    qualified_name: str
    summary: str = Field(..., description="Plain-language explanation of what this module does")
    key_responsibilities: list[str] = Field(default_factory=list)


class RepoIndex(BaseModel):
    """Metadata about an indexed repository."""
    repo_id: str
    name: str
    file_count: int
    module_count: int
    chunk_count: int
    indexed_at: str = Field(..., description="ISO timestamp")


class QASource(BaseModel):
    """A code citation used in a QA answer."""
    module_path: str
    qualified_name: str
    line_start: int
    line_end: int
    snippet: str


class QAResponse(BaseModel):
    """Answer to a user question about the codebase."""
    question: str
    answer: str
    sources: list[QASource] = Field(default_factory=list)
