"""Business logic for the todo app.

The service owns the rules ("how is a todo created", "what counts as
overdue") and delegates persistence to the storage layer. The CLI and any
future API only talk to TodoService -- not to InMemoryTodoStore directly.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from todo_app.models import Priority, Todo
from todo_app.storage import InMemoryTodoStore


class TodoService:
    """Application service: validates inputs and orchestrates the store."""

    def __init__(self, store: Optional[InMemoryTodoStore] = None) -> None:
        self._store = store or InMemoryTodoStore()

    def create(self, title: str, priority: Priority = Priority.MEDIUM) -> Todo:
        """Create a new todo. Empty titles are rejected."""
        title = title.strip()
        if not title:
            raise ValueError("title cannot be empty")
        return self._store.add(Todo(id=0, title=title, priority=priority))

    def complete(self, todo_id: int) -> Todo:
        """Mark a todo as done."""
        todo = self._store.get(todo_id)
        todo.mark_done()
        return todo

    def reopen(self, todo_id: int) -> Todo:
        """Re-open a completed todo."""
        todo = self._store.get(todo_id)
        todo.reopen()
        return todo

    def delete(self, todo_id: int) -> None:
        self._store.delete(todo_id)

    def list_open(self) -> list[Todo]:
        """Return open todos, highest priority first."""
        order = {Priority.HIGH: 0, Priority.MEDIUM: 1, Priority.LOW: 2}
        return sorted(
            (t for t in self._store.list_all() if not t.done),
            key=lambda t: order[t.priority],
        )

    def overdue(self, max_age: timedelta) -> list[Todo]:
        """Return open todos older than `max_age`."""
        threshold = datetime.utcnow() - max_age
        return [
            t for t in self._store.list_all()
            if not t.done and t.created_at < threshold
        ]
