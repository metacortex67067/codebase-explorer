"""Tiny command-line frontend for the todo app.

Not a fully-featured CLI -- just enough to demonstrate how the layers
plug together: argv -> CLI command -> TodoService -> InMemoryTodoStore.
"""
from __future__ import annotations

import sys
from typing import Sequence

from todo_app.models import Priority
from todo_app.service import TodoService


def main(argv: Sequence[str]) -> int:
    """Entry point. Returns a process exit code."""
    if not argv:
        _print_usage()
        return 1

    service = TodoService()
    command, *rest = argv

    if command == "add":
        title = " ".join(rest) or "(untitled)"
        todo = service.create(title)
        print(f"created #{todo.id}: {todo.title}")
        return 0

    if command == "done" and rest:
        todo = service.complete(int(rest[0]))
        print(f"done #{todo.id}")
        return 0

    if command == "list":
        for t in service.list_open():
            print(f"  [{t.priority.value}] #{t.id} {t.title}")
        return 0

    _print_usage()
    return 1


def _print_usage() -> None:
    print("usage: todo (add <title> | done <id> | list)")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
