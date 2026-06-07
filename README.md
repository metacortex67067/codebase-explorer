# Codebase Explorer

AI-сервис для понимания чужого Python-кода. Принимает репозиторий
(zip-архив, git URL или локальную директорию), индексирует его, и через
REST API или MCP-интерфейс отвечает на вопросы об архитектуре, объясняет
модули и строит граф зависимостей.

## Возможности

- Загрузка репозитория (zip-архив / git URL / локальная директория) с
  защитой от zip-slip и лимитами на размер и число файлов.
- AST-парсинг Python-кода: модули, классы, функции, импорты, docstrings.
- Граф зависимостей между модулями (только внутренние импорты).
- Семантическое чанкование: один чанк = одна функция / класс / метод
  плюс отдельный `module_header` чанк со сводкой модуля (4 типа чанков).
- Векторная индексация (sentence-transformers + ChromaDB) для поиска
  по коду.
- LLM-генерация объяснений модулей через сменный провайдер: локальный
  Ollama (бесплатно, офлайн) — по умолчанию; либо облако (Groq /
  OpenRouter, бесплатные тарифы) или Anthropic Claude — переключается
  одной строкой в `.env`.
- Q&A по коду через RAG с цитированием источников по индексу чанка.
- Веб-интерфейс (одна страница на `/`): загрузка репо, выбор, вопросы.
- REST API на FastAPI с автоматическим Swagger UI.
- MCP-сервер с 7 инструментами для интеграции в Claude Desktop / IDE.

## Архитектура

5 слоёв, каждый — отдельный модуль с явным контрактом. Контракт между
слоями — pydantic-модели. Подробности — в [docs/architecture.md](docs/architecture.md).

```
ingestion -> parsing -> indexing -> llm / qa
                                       |
                                api / mcp_server
```

- **ingestion** — приём репо (zip / git / директория), фильтрация файлов, лимиты.
- **parsing** — AST → модули, классы, функции, импорты; граф зависимостей.
- **indexing** — chunker, embedder (sentence-transformers), vector_store (ChromaDB), pipeline.
- **llm** — provider-agnostic клиент, промпты, суммаризация модулей.
- **qa** — retriever + RAG-loop с цитированием по индексу чанка.
- **RepoService** — единый фасад, оркеструет весь пайплайн для обоих интерфейсов.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate                # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

## Выбор LLM-провайдера

Провайдер задаётся переменной `LLM_PROVIDER` в `.env`:

- `ollama` — **встроенный дефолт в коде**: локально, бесплатно, офлайн,
  без ключей. Требует установленного [Ollama](https://ollama.com/download)
  и стянутой модели (`ollama pull qwen2.5-coder:7b`).
- `openai` — любой OpenAI-совместимый эндпоинт: **Groq** или **OpenRouter**
  (бесплатные тарифы). Нужен бесплатный ключ.
- `anthropic` — Anthropic Claude (платно).

> Внимание: поставляемый `.env.example` в блоке Option A уже выбирает
> **Groq** (`LLM_PROVIDER=openai`) — поэтому после `cp .env.example .env`
> нужно вписать бесплатный ключ Groq в `LLM_API_KEY`. Чтобы запустить
> полностью офлайн без ключей, раскомментируй блок **Option B (Ollama)**
> в `.env`. Если `.env` не создавать вовсе, действует встроенный дефолт —
> Ollama.

Эмбеддинги и векторный индекс уже локальные и бесплатные при любом выборе
LLM-провайдера.

## Запуск

REST API + веб-интерфейс:

```bash
uvicorn src.api.main:app --reload
# Веб-интерфейс: http://localhost:8000/
# Swagger UI:    http://localhost:8000/docs
```

MCP-сервер (для подключения в Claude Desktop):

```bash
python -m src.mcp_server.server
```

### Загрузка репозитория через API

```bash
# git URL
curl -X POST http://localhost:8000/repos/from-git \
  -H 'Content-Type: application/json' \
  -d '{"git_url": "https://github.com/USER/REPO.git"}'

# zip-архив
zip -r demo.zip examples/demo_repo
curl -X POST http://localhost:8000/repos/from-zip \
  -F 'file=@demo.zip'

# Вопрос с цитатами на код
curl -X POST http://localhost:8000/repos/{repo_id}/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "How is a new todo created?"}'
```

### MCP-сервер для Claude Desktop

Пример секции в `claude_desktop_config.json` (путь — абсолютный). В `env`
передаются те же переменные, что и в `.env` — здесь показан вариант с Groq;
для Ollama ключ не нужен, для Anthropic используется `ANTHROPIC_API_KEY`:

```json
{
  "mcpServers": {
    "codebase-explorer": {
      "command": "/Users/you/codebase_explorer/.venv/bin/python",
      "args": ["-m", "src.mcp_server.server"],
      "cwd": "/Users/you/codebase_explorer",
      "env": {
        "LLM_PROVIDER": "openai",
        "LLM_BASE_URL": "https://api.groq.com/openai/v1",
        "LLM_API_KEY": "gsk_...",
        "LLM_MODEL": "llama-3.3-70b-versatile"
      }
    }
  }
}
```

После перезапуска Claude Desktop доступны 7 инструментов:
`index_repo`, `list_repos`, `list_modules`, `explain_module`,
`get_dependency_graph`, `ask_question`, `delete_repo`.

## REST API: эндпоинты

| Метод | Путь | Назначение |
|---|---|---|
| `POST` | `/repos/from-git` | Проиндексировать репо по git URL |
| `POST` | `/repos/from-zip` | Проиндексировать репо из zip-архива |
| `GET` | `/repos` | Список проиндексированных репозиториев |
| `GET` | `/repos/{repo_id}` | Метаданные репозитория |
| `GET` | `/repos/{repo_id}/modules` | Список модулей с summary |
| `GET` | `/repos/{repo_id}/modules/{module_path}` | Детали модуля |
| `GET` | `/repos/{repo_id}/dependencies` | Граф зависимостей (JSON) |
| `POST` | `/repos/{repo_id}/ask` | Вопрос → ответ с цитатами |
| `DELETE` | `/repos/{repo_id}` | Удалить репозиторий из индекса |

Плюс служебный `GET /healthz` и веб-интерфейс на `GET /`.

## Тесты

```bash
pytest                              # все тесты
pytest --cov=src --cov-report=html  # с покрытием
```

**121 тест проходит, 1 помечен skip** (`test_vector_store.py` — требует
установленного `chromadb`). Покрытие по `src` — около 87 %. LLM, embedder
и vector store в тестах мокаются: тесты офлайн, детерминированные, не
требуют API-ключа и идут за несколько секунд.

## Бенчмарк

`scripts/benchmark.py` индексирует `examples/demo_repo`, прогоняет
ground-truth набор из 10 вопросов и перезаписывает `docs/metrics.md`
измеренными числами (по умолчанию `docs/metrics.md` содержит только
целевые диапазоны — их нужно заменить реальным прогоном на своей машине):

```bash
source .venv/bin/activate
# Нужна работающая LLM (Ollama запущен, либо ключ Groq/OpenRouter/Anthropic в .env)
python -m scripts.benchmark
```

Замеряет: число файлов / модулей / чанков, время индексации, размер
ChromaDB-индекса, медиану и p95 latency `/ask`, точность RAG (доля
ответов, цитирующих корректный модуль).

## Структура проекта

```
src/
  core/         pydantic-модели, config, RepoStore (sqlite), RepoService
  ingestion/    приём репо (zip / git / directory), фильтрация файлов
  parsing/      AST-парсер, граф зависимостей
  indexing/     chunker, embedder (sentence-transformers), vector_store
                (ChromaDB), pipeline
  llm/          provider-agnostic клиент (Ollama / Groq / OpenRouter /
                Anthropic), промпты, суммаризация модулей
  qa/           retriever, RAG-loop с цитированием по индексу
  api/          FastAPI: routes, schemas, exception handlers, веб-UI
  mcp_server/   stdio MCP-сервер с 7 tools поверх RepoService
  web/          одностраничный веб-интерфейс (index.html)
tests/          pytest, 121 тест (+1 skip)
docs/           architecture.md, metrics.md, report.md
examples/       demo_repo + ground-truth Q&A
scripts/        benchmark.py
```

## Ограничения текущей версии

- Только Python (расширяемо через интерфейс `LanguageParser`).
- Лимит репозитория: 50 МБ исходников / 2000 файлов.
- Только локальные эмбеддинги `sentence-transformers/all-MiniLM-L6-v2`.
- Все операции синхронные — для больших корпусов понадобится
  background-task очередь.
- API без авторизации — рассчитан на локальный запуск.

## Лицензия

MIT.
