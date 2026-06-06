from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictSchemaModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DocumentType(str, Enum):
    INVOICE = "invoice"
    PASSPORT = "passport"
    ID_CARD = "id_card"
    DRIVER_LICENSE = "driver_license"
    FINANCIAL_STATEMENT = "financial_statement"
    OTHER = "other"


class ProcessingEngine(str, Enum):
    DONUT = "donut"
    PADDLEOCR_VL = "paddleocr_vl"
    QWEN2_5_VL = "qwen2_5_vl"
    CUSTOM = "custom"


class NormalizedEntity(StrictSchemaModel):
    key: str = Field(..., description="Canonical entity path, for example person.full_name")
    value: str | int | float | bool | None = Field(
        ..., description="Normalized scalar value ready for search and storage"
    )
    source_field: str | None = Field(
        default=None, description="Original field path from the document-specific schema"
    )
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class ValidationIssue(StrictSchemaModel):
    code: str
    message: str
    field_path: str | None = None
    severity: Literal["info", "warning", "error"] = "error"


class ValidationSummary(StrictSchemaModel):
    is_valid: bool = True
    issues: list[ValidationIssue] = Field(default_factory=list)


class ProcessingMeta(StrictSchemaModel):
    engine: ProcessingEngine
    model_name: str
    model_version: str | None = None
    inference_started_at: datetime | None = None
    inference_finished_at: datetime | None = None
    elapsed_ms: int | None = Field(default=None, ge=0)
    device: str | None = None
    page_count: int = Field(default=1, ge=1)


class DocumentPage(StrictSchemaModel):
    page_index: int = Field(..., ge=0)
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)
    preview_url: str | None = None


class PartyInfo(StrictSchemaModel):
    name: str | None = None
    address: str | None = None
    tax_id: str | None = None
    iban: str | None = None


class LineItem(StrictSchemaModel):
    description: str | None = None
    quantity: str | None = None
    unit_price: str | None = None
    total_price: str | None = None
    tax_rate: str | None = None


class InvoiceFields(StrictSchemaModel):
    invoice_no: str | None = None
    invoice_date: str | date | None = None
    due_date: str | date | None = None
    currency: str | None = None
    seller: PartyInfo = Field(default_factory=PartyInfo)
    buyer: PartyInfo = Field(default_factory=PartyInfo)
    line_items: list[LineItem] = Field(default_factory=list)
    total_net: str | None = None
    total_tax: str | None = None
    total_gross: str | None = None


class PassportFields(StrictSchemaModel):
    document_number: str | None = None
    surname: str | None = None
    given_names: str | None = None
    nationality: str | None = None
    date_of_birth: str | date | None = None
    sex: str | None = None
    place_of_birth: str | None = None
    date_of_issue: str | date | None = None
    date_of_expiry: str | date | None = None
    issuing_authority: str | None = None
    mrz: str | None = None


class IdCardFields(StrictSchemaModel):
    document_number: str | None = None
    surname: str | None = None
    given_names: str | None = None
    nationality: str | None = None
    date_of_birth: str | date | None = None
    sex: str | None = None
    address: str | None = None
    personal_number: str | None = None
    date_of_issue: str | date | None = None
    date_of_expiry: str | date | None = None
    issuing_authority: str | None = None


class DriverLicenseFields(StrictSchemaModel):
    license_number: str | None = None
    surname: str | None = None
    given_names: str | None = None
    date_of_birth: str | date | None = None
    place_of_birth: str | None = None
    address: str | None = None
    date_of_issue: str | date | None = None
    date_of_expiry: str | date | None = None
    issuing_authority: str | None = None
    categories: list[str] = Field(default_factory=list)


class FinancialStatementFields(StrictSchemaModel):
    statement_period_start: str | date | None = None
    statement_period_end: str | date | None = None
    account_holder: str | None = None
    account_number: str | None = None
    iban: str | None = None
    bank_name: str | None = None
    currency: str | None = None
    opening_balance: str | None = None
    closing_balance: str | None = None


class OtherDocumentFields(StrictSchemaModel):
    extracted_pairs: dict[str, Any] = Field(default_factory=dict)


DocumentFields = (
    InvoiceFields
    | PassportFields
    | IdCardFields
    | DriverLicenseFields
    | FinancialStatementFields
    | OtherDocumentFields
)


class MultiDocumentExtractionResult(StrictSchemaModel):
    schema_version: str = "1.0"
    document_id: str | None = None
    document_type: DocumentType
    source_filename: str | None = None
    pages: list[DocumentPage] = Field(default_factory=list)
    fields: DocumentFields
    normalized_entities: list[NormalizedEntity] = Field(default_factory=list)
    raw_text: str | None = None
    raw_result: dict[str, Any] | None = None
    validation: ValidationSummary = Field(default_factory=ValidationSummary)
    confidence: dict[str, float] = Field(default_factory=dict)
    processing_meta: ProcessingMeta
