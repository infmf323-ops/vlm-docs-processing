# Invoice Extraction Web UI

## MVP scope

- Отдельный `backend` на FastAPI
- Отдельный `frontend` на React + Vite
- Авторизация по email/паролю
- Jobs pipeline:
  - загрузка изображения или PDF
  - создание `job`
  - просмотр списка `jobs`
  - страница деталей `job`
  - повторный запуск `job`
- Хранение:
  - метаданные и статусы в Postgres
  - файлы и артефакты в S3
  - отдельно `normalized_result` и `raw_result`
- Инференс:
  - локально на GPU
  - первая страница документа

## Что уже собрано

- backend-каркас:
  - конфиг
  - SQLAlchemy models
  - auth endpoints
  - jobs endpoints
  - S3 service
  - inference service
  - worker loop
- frontend-каркас:
  - login/register
  - upload page
  - jobs list
  - job detail
  - dark UI

## Следующие шаги

1. Подключить реальные миграции Alembic вместо `create_all`.
2. Добавить docker-compose для Postgres и S3-compatible storage.
3. Уточнить и стабилизировать схему `normalized_result`.
4. Проверить end-to-end локальный запуск и скорректировать зависимости.
