import logging
import time

from app.core.config import get_settings
from app.db.session import SessionLocal
from app.services.jobs import JobService


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("worker")


def main():
    # В режиме очереди задач (RQ) поллер должен простаивать, иначе он и RQ-воркер
    # будут одновременно забирать одни и те же задания (гонка двойной обработки).
    if get_settings().use_task_queue:
        logger.warning(
            "USE_TASK_QUEUE=true — активен режим очереди (RQ). Воркер-поллер отключён; "
            "запускайте worker_rq.py."
        )
        while True:
            time.sleep(60)

    logger.info("Worker запущен, ожидание заданий...")
    while True:
        db = SessionLocal()
        try:
            service = JobService(db)
            processed = service.process_next_job()
            if processed is None:
                time.sleep(2)
            else:
                logger.info(
                    "Задание #%s обработано: статус=%s%s",
                    processed.id,
                    processed.status.value,
                    f", ошибка: {processed.error_message}"
                    if processed.error_message
                    else "",
                )
        except Exception:  # noqa: BLE001 — воркер не должен падать из-за одного задания
            logger.exception("Непредвиденная ошибка в цикле обработки")
            time.sleep(2)
        finally:
            db.close()


if __name__ == "__main__":
    main()
