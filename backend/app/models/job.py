from datetime import datetime
from enum import Enum

from sqlalchemy import DateTime, Enum as SqlEnum, ForeignKey, Integer, JSON, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

# JSONB на PostgreSQL, обычный JSON на прочих СУБД (например, SQLite в тестах).
JsonType = JSONB().with_variant(JSON(), "sqlite")


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    status: Mapped[JobStatus] = mapped_column(
        SqlEnum(JobStatus, name="job_status"),
        default=JobStatus.QUEUED,
        index=True,
    )
    original_filename: Mapped[str] = mapped_column(String(512))
    content_type: Mapped[str] = mapped_column(String(128))
    # Тип документа, запрошенный пользователем при загрузке (известен сразу, до обработки).
    requested_document_type: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    source_s3_key: Mapped[str] = mapped_column(String(1024))
    preview_s3_key: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    normalized_result_s3_key: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    raw_result_s3_key: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    normalized_result: Mapped[dict | None] = mapped_column(JsonType, nullable=True)
    raw_result: Mapped[dict | None] = mapped_column(JsonType, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    page_count: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    user = relationship("User")
