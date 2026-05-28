"""Domain models for the todo application.

Two simple dataclasses model the world: a single Todo and a Priority enum.
We deliberately avoid pydantic here so the example stays beginner-friendly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class Priority(str, Enum):
    """How urgent a todo is. Ordered low -> high."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class Todo:
    """A single todo item."""
    id: int
    title: str
    priority: Priority = Priority.MEDIUM
    done: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    def mark_done(self) -> None:
        """Mark this todo as completed and stamp the completion time."""
        self.done = True
        self.completed_at = datetime.utcnow()

    def reopen(self) -> None:
        """Re-open a completed todo, clearing the completion timestamp."""
        self.done = False
        self.completed_at = None
