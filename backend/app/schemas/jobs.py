from datetime import datetime

from pydantic import BaseModel

from app.schemas.documents import DocumentType, ProcessingEngine
from app.models.job import JobStatus


class JobListItem(BaseModel):
    id: int
    status: JobStatus
    original_filename: str
    created_at: datetime
    updated_at: datetime | None = None
    finished_at: datetime | None = None
    error_message: str | None = None

    class Config:
        from_attributes = True


class JobDetail(JobListItem):
    content_type: str
    page_count: int
    requested_document_type: DocumentType | None = None
    requested_engine: ProcessingEngine | None = None
    preview_url: str | None = None
    normalized_result: dict | None = None
    raw_result: dict | None = None
    download_url: str | None = None


class JobRequestOptions(BaseModel):
    document_type: DocumentType = DocumentType.INVOICE
    extraction_engine: ProcessingEngine | None = None


class JobCreateResponse(BaseModel):
    id: int
    status: JobStatus


class JobRetryResponse(BaseModel):
    id: int
    status: JobStatus


class JobStats(BaseModel):
    total: int
    by_status: dict[str, int]
    by_document_type: dict[str, int]
    mrz_total: int
    mrz_valid: int
    mrz_valid_rate: float | None = None
    success_rate: float | None = None
