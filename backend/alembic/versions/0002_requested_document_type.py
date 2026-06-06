"""Добавление jobs.requested_document_type

Revision ID: 0002
Revises: 0001
Create Date: 2026-06-06
"""
from alembic import op
import sqlalchemy as sa

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "jobs",
        sa.Column("requested_document_type", sa.String(length=64), nullable=True),
    )
    op.create_index(
        "ix_jobs_requested_document_type", "jobs", ["requested_document_type"]
    )


def downgrade() -> None:
    op.drop_index("ix_jobs_requested_document_type", table_name="jobs")
    op.drop_column("jobs", "requested_document_type")
