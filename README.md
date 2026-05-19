# Codebase Explorer

AI-сервис для понимания чужого кода. Принимает Python-репозиторий, индексирует
его и через REST API / MCP-интерфейс отвечает на вопросы об архитектуре,
объясняет модули и строит граф зависимостей.

## Возможности

- Загрузка репозитория (zip-архив или git URL).
- AST-парсинг Python-кода: модули, классы, функции, импорты, docstrings.
- Построение графа зависимостей между модулями.
- Векторная индексация кода для семантического поиска.
- LLM-генерация объяснений модулей (Anthropic Claude).
- Q&A по коду через RAG с цитированием источников.
- REST API на FastAPI.
- MCP-сервер для интеграции с Claude Desktop / IDE.

## Архитектура

5 слоёв, каждый — отдельный модуль с явным контрактом:

```
ingestion -> parsing -> indexing -> llm
                                      |
                              api / mcp_server
```

Подробности — в `docs/architecture.md`.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env         # затем впиши свой ANTHROPIC_API_KEY
```

## Запуск

REST API:

```bash
uvicorn src.api.main:app --reload
# Swagger UI: http://localhost:8000/docs
```

MCP-сервер (для подключения в Claude Desktop):

```bash
python -m src.mcp_server.server
```

## Тесты

```bash
pytest                          # запуск
pytest --cov=src --cov-report=html   # с покрытием
```

## Структура проекта

```
src/
  core/         — pydantic-модели и конфигурация
  ingestion/    — приём репо (zip / git), фильтрация файлов
  parsing/      — AST-анализ, граф зависимостей
  indexing/     — chunker, эмбеддинги, векторное хранилище
  llm/          — клиент Anthropic, промпты, суммаризация
  qa/           — RAG-пайплайн для Q&A
  api/          — FastAPI-роуты
  mcp_server/   — MCP-tool-интерфейс
tests/          — pytest, юнит и интеграционные тесты
docs/           — архитектура, API-справочник
examples/       — тестовые репозитории для демо
```

## Ограничения текущей версии

- Только Python (расширяемо через интерфейс LanguageParser).
- Лимит репозитория: 50 МБ исходников / 2000 файлов.
- Только локальные эмбеддинги `sentence-transformers/all-MiniLM-L6-v2`.

## Лицензия

Учебный проект.
