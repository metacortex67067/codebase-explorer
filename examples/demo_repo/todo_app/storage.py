"""In-memory storage for todos.

A real app would use sqlite/postgres; for the demo a dict is enough to
illustrate a "repository pattern" layer that the service depends on.
"""
from __future__ import annotations

from typing import Iterable, Optional

from todo_app.models import Todo


class TodoNotFound(Exception):
    """Raised when a todo id does not exist in storage."""


class InMemoryTodoStore:
    """Dictionary-backed store. Not thread-safe (single-process demo)."""

    def __init__(self) -> None:
        self._items: dict[int, Todo] = {}
        self._next_id: int = 1

    def add(self, todo: Todo) -> Todo:
        """Insert a todo, assigning a fresh id if it does not have one."""
        if todo.id == 0:
            todo.id = self._next_id
            self._next_id += 1
        else:
            self._next_id = max(self._next_id, todo.id + 1)
        self._items[todo.id] = todo
        return todo

    def get(self, todo_id: int) -> Todo:
        """Return the todo with this id or raise TodoNotFound."""
        if todo_id not in self._items:
            raise TodoNotFound(f"No todo with id={todo_id}")
        return self._items[todo_id]

    def list_all(self) -> list[Todo]:
        """Return all todos sorted by id."""
        return sorted(self._items.values(), key=lambda t: t.id)

    def delete(self, todo_id: int) -> None:
        """Remove a todo. Idempotent: missing ids are silently ignored."""
        self._items.pop(todo_id, None)
