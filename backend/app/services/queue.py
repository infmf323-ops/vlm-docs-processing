"""Очередь задач на основе RQ + Redis (альтернатива воркеру-поллеру).

Включается флагом ``USE_TASK_QUEUE=true``. При создании задания оно ставится в
очередь (``enqueue_job``), а отдельный RQ-воркер (``worker_rq.py``) вызывает
``run_job`` для обработки конкретного задания. Это позволяет масштабировать
обработку горизонтально, запуская несколько воркеров.
"""
from __future__ import annotations

from functools import lru_cache

from app.core.config import get_settings

QUEUE_NAME = "documents"


@lru_cache
def get_queue():
    from redis import Redis
    from rq import Queue

    settings = get_settings()
    return Queue(QUEUE_NAME, connection=Redis.from_url(settings.redis_url))


def enqueue_job(job_id: int) -> None:
    """Ставит задание в очередь на обработку."""
    get_queue().enqueue("app.services.queue.run_job", job_id)


def run_job(job_id: int) -> None:
    """Точка входа RQ-воркера: обрабатывает одно задание по id."""
    from app.db.session import SessionLocal
    from app.services.jobs import JobService

    db = SessionLocal()
    try:
        JobService(db).process_job(job_id)
    finally:
        db.close()
