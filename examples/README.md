# Examples

## `demo_repo/`

Synthetic Python project used to demo Codebase Explorer end-to-end.

Layout:

```
todo_app/
  __init__.py    package marker + brief description
  models.py      Todo dataclass + Priority enum
  storage.py     InMemoryTodoStore (repository pattern)
  service.py     TodoService (business logic on top of storage)
  cli.py         minimal command-line frontend
```

Why these particular shapes? It exercises every chunker type:
- `__init__.py` -> a module_header chunk with only a docstring
- `models.py` -> classes (dataclasses, enum) with methods
- `storage.py` -> a class with several methods + a top-level exception
- `service.py` -> a class whose methods rely on imports across modules
- `cli.py` -> top-level functions, no classes

And it produces a non-trivial dependency graph:
`cli -> service -> storage, models` and `service -> models`.

## Ground-truth Q&A set

The 10 questions in [`qa_groundtruth.json`](qa_groundtruth.json) are used
by `scripts/benchmark.py` to measure retrieval accuracy. Each question
declares the module that the correct answer must cite.
