# Multi-Document Extraction Schema

## Goal

This project is moving from a single `invoice` extraction flow to a multi-document pipeline that should handle:

- invoices
- passports
- identity cards
- driver licenses
- financial statements
- other document classes

The schema should stay stable for storage and API responses, while still allowing document-specific fields.

## Recommended design

The recommended design is a **hybrid schema**:

1. One shared top-level envelope for every processed document.
2. One strict field schema per `document_type`.
3. One optional normalized entity layer for cross-document analytics and search.

This follows the pattern used by modern document AI systems:

- classifier + extraction model pairs in Azure Document Intelligence
- document-type-specific extractors in Google Document AI
- separate identity and expense schemas in AWS Textract

## Top-level envelope

Every extraction result should use the same outer structure:

```json
{
  "schema_version": "1.0",
  "document_id": "uuid-or-job-id",
  "document_type": "passport",
  "source_filename": "passport.png",
  "pages": [],
  "fields": {},
  "normalized_entities": [],
  "raw_text": "...",
  "raw_result": {},
  "validation": {
    "is_valid": true,
    "issues": []
  },
  "confidence": {},
  "processing_meta": {}
}
```

## Why this design works

### 1. It supports many document types

The outer contract never changes, while `fields` can vary depending on `document_type`.

### 2. It is safe for databases and APIs

Backend code, storage, and UI can rely on one stable response envelope.

### 3. It is compatible with model routing

A classifier can pick the document type first, and the extraction engine can then fill the appropriate schema.

### 4. It is future-proof

New document types can be added without breaking old records.

## Proposed document types

The first set of document types should be:

- `invoice`
- `passport`
- `id_card`
- `driver_license`
- `financial_statement`
- `other`

## Type-specific field strategy

Instead of one giant flat JSON with hundreds of nullable fields, each document type has its own typed field model.

### Invoice

Core fields:

- `invoice_no`
- `invoice_date`
- `due_date`
- `currency`
- `seller`
- `buyer`
- `line_items`
- `total_net`
- `total_tax`
- `total_gross`

### Passport

Core fields:

- `document_number`
- `surname`
- `given_names`
- `nationality`
- `date_of_birth`
- `sex`
- `place_of_birth`
- `date_of_issue`
- `date_of_expiry`
- `issuing_authority`
- `mrz`

### ID card

Core fields:

- `document_number`
- `surname`
- `given_names`
- `nationality`
- `date_of_birth`
- `sex`
- `address`
- `personal_number`
- `date_of_issue`
- `date_of_expiry`
- `issuing_authority`

### Driver license

Core fields:

- `license_number`
- `surname`
- `given_names`
- `date_of_birth`
- `place_of_birth`
- `address`
- `date_of_issue`
- `date_of_expiry`
- `issuing_authority`
- `categories`

## Normalized entity layer

For search, BI, and cross-document comparison, a normalized entity layer should be stored alongside `fields`.

Examples:

- `person.full_name`
- `person.date_of_birth`
- `document.number`
- `document.issue_date`
- `document.expiry_date`
- `organization.issuer`

This allows:

- one unified search index
- easier analytics across document types
- simpler mapping to external systems

## Validation layer

Validation should be independent from the extraction model.

Recommended checks:

- required fields for each document type
- date normalization
- number and identifier format checks
- country-specific or issuer-specific rules when available
- low-confidence or partial extraction detection

Validation results should go into:

- `validation.is_valid`
- `validation.issues`

## Confidence layer

Confidence should be stored as a dictionary keyed by field path.

Examples:

```json
{
  "fields.document_number": 0.98,
  "fields.date_of_birth": 0.94,
  "fields.issuing_authority": 0.67
}
```

This is useful for:

- manual review routing
- UI highlighting
- quality monitoring

## Implementation note

Typed schema models for this design are defined in:

- [documents.py](/E:/thesis/backend/app/schemas/documents.py)

These models are not yet wired into the active API and worker pipeline. They are the target schema contract for the upcoming migration from `invoice-only Donut` to a multi-document architecture.
