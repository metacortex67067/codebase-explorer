# Codebase Explorer

AI-сервис для понимания чужого Python-кода. Принимает репозиторий
(zip-архив или git URL), индексирует его, и через REST API или MCP-интерфейс
отвечает на вопросы об архитектуре, объясняет модули и строит граф зависимостей.

## Возможности

- Загрузка репозитория (zip-архив или git URL) с защитой от zip-slip и
  лимитами на размер / число файлов.
- AST-парсинг Python-кода: модули, классы, функции, импорты, docstrings.
- Граф зависимостей между модулями (только внутренние импорты).
- Семантическое чанкование: один чанк = одна функция / класс / метод
  плюс отдельный `module_header` чанк со сводкой модуля.
- Векторная индексация (sentence-transformers + ChromaDB) для поиска
  по коду.
- LLM-генерация объяснений модулей через сменный провайдер: локальный
  Ollama (бесплатно, офлайн) по умолчанию, либо облако (Groq / OpenRouter)
  или Anthropic Claude — переключается одной строкой в `.env`.
- Q&A по коду через RAG с цитированием источников по индексу чанка.
- Веб-интерфейс (одна страница на `/`): загрузка репо, выбор, вопросы.
- REST API на FastAPI с автоматическим Swagger UI.
- MCP-сервер с 7 инструментами для интеграции в Claude Desktop / IDE.

## Архитектура

5 слоёв, каждый — отдельный модуль с явным контрактом. Подробности —
в [docs/architecture.md](docs/architecture.md).

```
ingestion -> parsing -> indexing -> llm / qa
                                       |
                                api / mcp_server
```

## Запуск за 30 секунд

```bash
python -m venv .venv
source .venv/bin/activate                # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env

# Бесплатная LLM через Groq (рекомендуется):
#   1) получи бесплатный ключ на https://console.groq.com/keys
#   2) впиши его в .env в строку LLM_API_KEY (блок Option A)
# Либо полностью офлайн через Ollama — см. блок Option B в .env.example.

# Запуск сервиса
uvicorn src.api.main:app --reload
# Веб-интерфейс: http://localhost:8000/
# Swagger UI:    http://localhost:8000/docs
```

LLM-провайдер выбирается в `.env` (`LLM_PROVIDER`): `openai` для Groq /
OpenRouter (бесплатные тарифы), `ollama` для локального офлайн-запуска или
`anthropic` (платно). Эмбеддинги и векторный индекс уже локальные и
бесплатные.

### Загрузка демо-репозитория через API

```bash
# Из локальной директории через git (если репо запушен)
curl -X POST http://localhost:8000/repos/from-git \
  -H 'Content-Type: application/json' \
  -d '{"git_url": "https://github.com/USER/REPO.git"}'

# Либо zip-архив
zip -r demo.zip examples/demo_repo
curl -X POST http://localhost:8000/repos/from-zip \
  -F 'file=@demo.zip'

# Получить ответ с цитатами на код
curl -X POST http://localhost:8000/repos/{repo_id}/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "How is a new todo created?"}'
```

### MCP-сервер для Claude Desktop

```bash
python -m src.mcp_server.server
```

Пример секции в `claude_desktop_config.json` (путь — абсолютный):

```json
{
  "mcpServers": {
    "codebase-explorer": {
      "command": "/Users/you/codebase_explorer/.venv/bin/python",
      "args": ["-m", "src.mcp_server.server"],
      "cwd": "/Users/you/codebase_explorer",
      "env": {"ANTHROPIC_API_KEY": "sk-..."}
    }
  }
}
```

После перезапуска Claude Desktop вызывает любой из 7 инструментов:
`index_repo`, `list_repos`, `list_modules`, `explain_module`,
`get_dependency_graph`, `ask_question`, `delete_repo`.

## Тесты

```bash
pytest                              # все тесты
pytest --cov=src --cov-report=html  # с покрытием
```

LLM, embedder и vector store в тестах мокаются — тесты офлайн,
детерминированные, не требуют API-ключа. Один тест (`test_vector_store.py`)
помечен skip когда не установлен `chromadb`.

## Бенчмарк

`scripts/benchmark.py` индексирует `examples/demo_repo`, прогоняет
ground-truth набор из 10 вопросов и пишет результаты в `docs/metrics.md`:

```bash
export ANTHROPIC_API_KEY=...
python -m scripts.benchmark
```

Замеры: время индексации, размер ChromaDB-индекса, медиана и p95 latency
`/ask`, точность RAG (доля ответов, цитирующих корректный модуль).

## Структура проекта

```
src/
  core/         pydantic-модели, config, RepoStore (sqlite), RepoService
  ingestion/    приём репо (zip / git / directory), фильтрация файлов
  parsing/      AST-парсер, граф зависимостей
  indexing/     chunker, embedder (sentence-transformers), vector_store
                (ChromaDB), pipeline
  llm/          клиент Anthropic, промпты, суммаризация модулей
  qa/           retriever, RAG-loop с цитированием по индексу
  api/          FastAPI: routes, schemas, exception handlers
  mcp_server/   stdio MCP-сервер с 7 tools поверх RepoService
tests/          pytest, 117 тестов
docs/           architecture.md, metrics.md, report.md
examples/       demo_repo + ground-truth Q&A
scripts/        benchmark.py
```

## Ограничения текущей версии

- Только Python (расширяемо через `LanguageParser`).
- Лимит репозитория: 50 МБ исходников / 2000 файлов.
- Только локальные эмбеддинги `sentence-transformers/all-MiniLM-L6-v2`.
- Все операции синхронные — для коллабораций на больших корпусах
  понадобится background-task очередь.

## Лицензия

MIT
