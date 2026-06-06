import csv
import io

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Response, UploadFile, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.config import get_settings
from app.db.session import get_db
from app.models.job import JobStatus
from app.models.user import User
from app.schemas.documents import DocumentType, ProcessingEngine
from app.schemas.jobs import (
    JobCreateResponse,
    JobDetail,
    JobListItem,
    JobRequestOptions,
    JobRetryResponse,
    JobStats,
)
from app.services.jobs import JobService, job_document_type, job_mrz_valid, job_is_mrz
from app.services.storage import StorageService


router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("", response_model=list[JobListItem])
def list_jobs(
    status: JobStatus | None = Query(default=None),
    document_type: DocumentType | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return JobService(db).list_jobs(
        current_user.id,
        status=status,
        document_type=document_type,
        limit=limit,
        offset=offset,
    )


@router.get("/stats", response_model=JobStats)
def job_stats(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return JobService(db).compute_stats(current_user.id)


@router.get("/export.csv")
def export_jobs_csv(
    status: JobStatus | None = Query(default=None),
    document_type: DocumentType | None = Query(default=None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    jobs = JobService(db).list_jobs(
        current_user.id, status=status, document_type=document_type
    )
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "id",
            "original_filename",
            "status",
            "document_type",
            "mrz",
            "mrz_valid",
            "created_at",
            "finished_at",
            "error_message",
        ]
    )
    for job in jobs:
        writer.writerow(
            [
                job.id,
                job.original_filename,
                job.status.value,
                job_document_type(job) or "",
                "yes" if job_is_mrz(job) else "",
                "yes" if (job_is_mrz(job) and job_mrz_valid(job)) else "",
                job.created_at.isoformat() if job.created_at else "",
                job.finished_at.isoformat() if job.finished_at else "",
                (job.error_message or "").replace("\n", " "),
            ]
        )
    return Response(
        content=buffer.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=jobs.csv"},
    )


@router.post("", response_model=JobCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_job(
    file: UploadFile = File(...),
    document_type: DocumentType = Form(default=DocumentType.INVOICE),
    extraction_engine: ProcessingEngine = Form(default=ProcessingEngine.DONUT),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Загруженный файл пуст"
        )
    settings = get_settings()
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Файл превышает лимит {settings.max_upload_mb} МБ",
        )
    content_type = file.content_type or "application/octet-stream"
    if content_type not in settings.allowed_content_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Неподдерживаемый тип файла: {content_type}",
        )
    try:
        job = JobService(db).create_job(
            user_id=current_user.id,
            filename=file.filename or "document",
            content_type=content_type,
            content=content,
            request_options=JobRequestOptions(
                document_type=document_type,
                extraction_engine=extraction_engine,
            ),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Не удалось прочитать файл: {exc}",
        ) from exc
    return JobCreateResponse(id=job.id, status=job.status)


@router.get("/{job_id}", response_model=JobDetail)
def get_job(
    job_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    service = JobService(db)
    storage = StorageService()
    job = service.get_job(current_user.id, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    request_options = service.get_request_options(job)
    preview_url = storage.make_download_url(job.preview_s3_key) if job.preview_s3_key else None
    download_url = (
        storage.make_download_url(job.normalized_result_s3_key)
        if job.normalized_result_s3_key
        else None
    )

    return JobDetail(
        id=job.id,
        status=job.status,
        original_filename=job.original_filename,
        created_at=job.created_at,
        updated_at=job.updated_at,
        finished_at=job.finished_at,
        error_message=job.error_message,
        content_type=job.content_type,
        page_count=job.page_count,
        requested_document_type=job.requested_document_type or request_options.document_type,
        requested_engine=request_options.extraction_engine,
        preview_url=preview_url,
        normalized_result=job.normalized_result,
        raw_result=job.raw_result,
        download_url=download_url,
    )


@router.post("/{job_id}/retry", response_model=JobRetryResponse)
def retry_job(
    job_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    service = JobService(db)
    job = service.get_job(current_user.id, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    job = service.retry_job(job)
    return JobRetryResponse(id=job.id, status=job.status)


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_job(
    job_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    service = JobService(db)
    job = service.get_job(current_user.id, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    service.delete_job(job)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
