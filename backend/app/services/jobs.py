from collections import Counter
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.job import Job, JobStatus
from app.services.document import (
    count_pages,
    load_all_page_images,
    load_first_page_image,
)
from app.schemas.documents import DocumentType
from app.schemas.jobs import JobRequestOptions, JobStats
from app.services.extraction import ExtractionService
from app.services.storage import StorageService


def job_document_type(job: Job) -> str | None:
    """Тип документа задания: запрошенный пользователем при загрузке (известен сразу,
    в том числе для queued/failed), с откатом на распознанный из normalized_result."""
    requested = getattr(job, "requested_document_type", None)
    if requested:
        return requested
    normalized = job.normalized_result or {}
    value = normalized.get("document_type")
    return value if isinstance(value, str) else None


def job_is_mrz(job: Job) -> bool:
    raw = job.raw_result or {}
    return raw.get("mode") == "passport_mrz"


def job_mrz_valid(job: Job) -> bool:
    normalized = job.normalized_result or {}
    validation = normalized.get("validation") or {}
    return bool(validation.get("is_valid"))


def merge_multipage_results(payloads: list[dict]) -> dict:
    """Объединяет результаты по страницам: за основу берётся первая страница,
    сущности и текст остальных страниц добавляются с пометкой номера страницы."""
    base = dict(payloads[0])
    entities = list(base.get("normalized_entities") or [])
    raw_texts = [base.get("raw_text") or ""]
    for page_no, payload in enumerate(payloads[1:], start=2):
        for entity in payload.get("normalized_entities") or []:
            tagged = dict(entity)
            origin = tagged.get("source_field") or tagged.get("key") or ""
            tagged["source_field"] = f"page{page_no}:{origin}"
            entities.append(tagged)
        if payload.get("raw_text"):
            raw_texts.append(payload["raw_text"])
    base["normalized_entities"] = entities
    base["raw_text"] = "\n\n".join(t for t in raw_texts if t)
    return base


class JobService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.storage = StorageService()
        self.storage.ensure_bucket()

    def create_job(
        self,
        *,
        user_id: int,
        filename: str,
        content_type: str,
        content: bytes,
        request_options: JobRequestOptions | None = None,
    ) -> Job:
        request_options = request_options or JobRequestOptions()
        object_prefix = f"jobs/{uuid4()}"
        source_key = f"{object_prefix}/source"
        preview_key = f"{object_prefix}/preview.png"
        request_key = f"{object_prefix}/request.json"

        image, preview_bytes = load_first_page_image(content, content_type)
        page_count = count_pages(content, content_type)
        self.storage.upload_bytes(source_key, content, content_type)
        self.storage.upload_bytes(preview_key, preview_bytes, "image/png")
        self.storage.upload_json(request_key, request_options.model_dump(mode="json"))

        job = Job(
            user_id=user_id,
            status=JobStatus.QUEUED,
            original_filename=filename,
            content_type=content_type,
            requested_document_type=request_options.document_type.value,
            source_s3_key=source_key,
            preview_s3_key=preview_key,
            page_count=page_count,
        )
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)
        if get_settings().use_task_queue:
            # Режим очереди задач: ставим задание в RQ (иначе его заберёт воркер-поллер).
            from app.services.queue import enqueue_job

            enqueue_job(job.id)
        return job

    def list_jobs(
        self,
        user_id: int,
        *,
        status: JobStatus | None = None,
        document_type: DocumentType | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Job]:
        stmt = select(Job).where(Job.user_id == user_id)
        if status is not None:
            stmt = stmt.where(Job.status == status)
        stmt = stmt.order_by(Job.created_at.desc())
        jobs = list(self.db.scalars(stmt).all())
        if document_type is not None:
            jobs = [j for j in jobs if job_document_type(j) == document_type.value]
        if offset:
            jobs = jobs[offset:]
        if limit is not None:
            jobs = jobs[:limit]
        return jobs

    def delete_job(self, job: Job) -> None:
        # Best-effort очистка объектов в хранилище — ошибки не должны мешать удалению записи.
        keys = [
            job.source_s3_key,
            job.preview_s3_key,
            job.normalized_result_s3_key,
            job.raw_result_s3_key,
            self._request_key_for_job(job),
        ]
        for key in keys:
            if not key:
                continue
            try:
                self.storage.delete_object(key)
            except Exception:
                pass
        self.db.delete(job)
        self.db.commit()

    def compute_stats(self, user_id: int) -> JobStats:
        jobs = list(
            self.db.scalars(select(Job).where(Job.user_id == user_id)).all()
        )
        by_status = Counter(job.status.value for job in jobs)
        by_document_type: Counter = Counter()
        mrz_total = 0
        mrz_valid = 0
        for job in jobs:
            doc_type = job_document_type(job)
            if doc_type:
                by_document_type[doc_type] += 1
            if job_is_mrz(job):
                mrz_total += 1
                if job_mrz_valid(job):
                    mrz_valid += 1
        finished = by_status.get(JobStatus.DONE.value, 0) + by_status.get(
            JobStatus.FAILED.value, 0
        )
        success_rate = (
            by_status.get(JobStatus.DONE.value, 0) / finished if finished else None
        )
        mrz_valid_rate = mrz_valid / mrz_total if mrz_total else None
        return JobStats(
            total=len(jobs),
            by_status=dict(by_status),
            by_document_type=dict(by_document_type),
            mrz_total=mrz_total,
            mrz_valid=mrz_valid,
            mrz_valid_rate=mrz_valid_rate,
            success_rate=success_rate,
        )

    def get_job(self, user_id: int, job_id: int) -> Job | None:
        stmt = select(Job).where(Job.user_id == user_id, Job.id == job_id)
        return self.db.scalar(stmt)

    def get_request_options(self, job: Job) -> JobRequestOptions:
        request_key = self._request_key_for_job(job)
        try:
            payload = self.storage.download_bytes(request_key)
        except Exception:
            return JobRequestOptions()
        return JobRequestOptions.model_validate_json(payload)

    def retry_job(self, job: Job) -> Job:
        job.status = JobStatus.QUEUED
        job.error_message = None
        job.finished_at = None
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)
        if get_settings().use_task_queue:
            # В режиме очереди поллера нет — повторно публикуем задание, иначе оно
            # навсегда останется в статусе queued.
            from app.services.queue import enqueue_job

            enqueue_job(job.id)
        return job

    def process_next_job(self) -> Job | None:
        """Берёт следующее задание из очереди (режим поллинга воркером)."""
        stmt = select(Job).where(Job.status == JobStatus.QUEUED).order_by(Job.created_at.asc())
        job = self.db.scalar(stmt)
        if job is None:
            return None
        return self._process(job)

    def process_job(self, job_id: int) -> Job | None:
        """Обрабатывает конкретное задание по id (режим очереди задач, RQ)."""
        job = self.db.get(Job, job_id)
        if job is None:
            return None
        return self._process(job)

    def _process(self, job: Job) -> Job:
        job.status = JobStatus.PROCESSING
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)

        try:
            content = self.storage.download_bytes(job.source_s3_key)
            request_options = self.get_request_options(job)
            images = load_all_page_images(content, job.content_type)
            engine = ExtractionService()

            if len(images) == 1:
                output = engine.predict(
                    image=images[0],
                    source_filename=job.original_filename,
                    request=request_options,
                )
                normalized_payload = output.normalized_result.model_dump(mode="json")
                raw_payload = output.raw_result
            else:
                # Многостраничный документ: обрабатываем каждую страницу и объединяем.
                page_outputs = [
                    engine.predict(
                        image=img,
                        source_filename=job.original_filename,
                        request=request_options,
                    )
                    for img in images
                ]
                normalized_payload = merge_multipage_results(
                    [o.normalized_result.model_dump(mode="json") for o in page_outputs]
                )
                raw_payload = {
                    "mode": "multipage",
                    "page_count": len(images),
                    "pages": [o.raw_result for o in page_outputs],
                }

            base_prefix = job.source_s3_key.rsplit("/", 1)[0]
            normalized_key = f"{base_prefix}/normalized.json"
            raw_key = f"{base_prefix}/raw.json"

            self.storage.upload_json(normalized_key, normalized_payload)
            self.storage.upload_json(raw_key, raw_payload)

            job.normalized_result = normalized_payload
            job.raw_result = raw_payload
            job.normalized_result_s3_key = normalized_key
            job.raw_result_s3_key = raw_key
            job.status = JobStatus.DONE
            job.finished_at = datetime.now(timezone.utc)
            job.error_message = None
        except Exception as exc:
            job.status = JobStatus.FAILED
            job.error_message = str(exc)
            job.finished_at = datetime.now(timezone.utc)

        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)
        return job

    def _request_key_for_job(self, job: Job) -> str:
        base_prefix = job.source_s3_key.rsplit("/", 1)[0]
        return f"{base_prefix}/request.json"
