# Backend

## Что уже есть

- FastAPI API
- JWT auth
- Postgres models для `users` и `jobs`
- S3 storage service
- worker для последовательной обработки jobs
- интеграция с current fine-tuned Donut model и MRZ-путём для паспортов

## Эндпоинты `/api/v1/jobs`

| Метод | Путь | Назначение |
|-------|------|------------|
| `GET` | `/jobs?status=&document_type=` | список заданий с фильтрами |
| `POST` | `/jobs` | загрузка документа (пустой файл → 400) |
| `GET` | `/jobs/stats` | сводная статистика (по статусам, типам, MRZ) |
| `GET` | `/jobs/export.csv?status=&document_type=` | выгрузка в CSV |
| `GET` | `/jobs/{id}` | детали задания |
| `POST` | `/jobs/{id}/retry` | повторный запуск |
| `DELETE` | `/jobs/{id}` | удаление задания и его объектов в хранилище |

Все эндпоинты изолируют данные по пользователю: чужое задание возвращает 404.

## Тесты

```powershell
pytest tests/ -v
```

- `tests/test_mrz.py` — пакет `app.mrz` (контрольные цифры, разбор, пост-коррекция).
- `tests/test_api.py` — HTTP API (auth + jobs) на in-memory SQLite; обращения к
  S3 и ML-движку подменяются. PostgreSQL/MinIO и torch для тестов не нужны
  (см. `tests/conftest.py`).

## Запуск всего стека (Docker)

Из корня репозитория (`E:\thesis`):

```powershell
docker compose up -d --build
```

Поднимаются: PostgreSQL, MinIO (+ автосоздание bucket), API (с применением
миграций), worker и фронтенд. UI доступен на http://localhost:5173, API — на
http://localhost:8000. Каталоги моделей (`./outputs`) и кэш HuggingFace
(`./.hf-cache`) монтируются в контейнеры; пути и `MODEL_PATH` настраиваются через
переменные окружения (см. `docker-compose.yml`).

## Локальный запуск (без Docker)

1. Поднять только инфраструктуру: `docker compose up -d postgres minio createbuckets`.
2. `Copy-Item .env.example .env`
3. `python -m pip install -r requirements.txt`
4. Применить миграции: `alembic upgrade head` (или оставить `AUTO_CREATE_TABLES=true`).
5. `uvicorn app.main:app --reload --port 8000`
6. В отдельном окне: `python worker.py`

## Миграции БД (Alembic)

Схема описана миграциями в `alembic/versions`. Применить:

```powershell
alembic upgrade head
```

Создать новую миграцию по изменениям моделей:

```powershell
alembic revision --autogenerate -m "описание"
```

В продакшене выставьте `AUTO_CREATE_TABLES=false`, чтобы схемой управлял только
Alembic (в Docker это уже сделано — API применяет `alembic upgrade head` при старте).

## Что дальше

- **Внешний OCR-baseline на held-out (приоритет).** Прогнать `scripts/run_ocr_baseline.py`
  (готовый Tesseract/EasyOCR + наш TD3-парсер `app.mrz`) на той же отложенной выборке,
  что и модель, и добавить столбец «OCR+parser» в сравнение. Это закрывает вопрос
  «зачем VLM+LoRA, а не простой OCR» числом, а не словами. Не требует обучения.
  Самопроверка обвязки: `python scripts/run_ocr_baseline.py --self-test`.
- стабилизировать схему `normalized_result`
- метрики и трассировка (OpenTelemetry/Prometheus)
- (сделано) очередь задач RQ/Redis как альтернатива поллингу — см. `worker_rq.py`, `app/services/queue.py`
