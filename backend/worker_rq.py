"""RQ-воркер: обрабатывает задания из очереди Redis.

Запуск (при USE_TASK_QUEUE=true):
    python worker_rq.py
или штатной командой RQ:
    rq worker -u redis://localhost:6379/0 documents
"""
import logging

from redis import Redis
from rq import Queue, Worker

from app.core.config import get_settings
from app.services.queue import QUEUE_NAME

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def main() -> None:
    settings = get_settings()
    connection = Redis.from_url(settings.redis_url)
    queue = Queue(QUEUE_NAME, connection=connection)
    Worker([queue], connection=connection).work(with_scheduler=True)


if __name__ == "__main__":
    main()
