# Multi-Document Dataset Format

## Purpose

This format is used to prepare a small but consistent training set for:

- `passport`
- `id_card`
- `driver_license`
- `invoice`

The same schema can be used for:

- `heuristic pipeline` evaluation
- `LoRA fine-tuning`
- final benchmark comparison

## File structure

Recommended directory layout:

```text
E:\thesis\data\multidoc\
  train.jsonl
  val.jsonl
  images\
    passport_001.png
    id_card_001.png
    driver_license_001.png
    invoice_001.png
```

## JSONL sample format

One line = one document.

```json
{
  "id": "passport_001",
  "image_path": "E:/thesis/data/multidoc/images/passport_001.png",
  "document_type": "passport",
  "fields": {
    "document_number": "910239248",
    "surname": "OBAMA",
    "given_names": "MICHELLE",
    "nationality": "UNITED STATES OF AMERICA",
    "date_of_birth": "17 Jan 1964",
    "sex": "F",
    "place_of_birth": "ILLINOIS, U.S.A.",
    "date_of_issue": "06 Dec 2013",
    "date_of_expiry": "05 Dec 2018",
    "issuing_authority": "United States Department of State"
  }
}
```

## Supported document types

### passport

```json
{
  "document_number": "string | null",
  "surname": "string | null",
  "given_names": "string | null",
  "nationality": "string | null",
  "date_of_birth": "string | null",
  "sex": "string | null",
  "place_of_birth": "string | null",
  "date_of_issue": "string | null",
  "date_of_expiry": "string | null",
  "issuing_authority": "string | null"
}
```

### id_card

```json
{
  "document_number": "string | null",
  "surname": "string | null",
  "given_names": "string | null",
  "nationality": "string | null",
  "date_of_birth": "string | null",
  "sex": "string | null",
  "address": "string | null",
  "personal_number": "string | null",
  "date_of_issue": "string | null",
  "date_of_expiry": "string | null",
  "issuing_authority": "string | null"
}
```

### driver_license

```json
{
  "license_number": "string | null",
  "surname": "string | null",
  "given_names": "string | null",
  "date_of_birth": "string | null",
  "place_of_birth": "string | null",
  "address": "string | null",
  "date_of_issue": "string | null",
  "date_of_expiry": "string | null",
  "issuing_authority": "string | null",
  "categories": ["B", "C"]
}
```

### invoice

```json
{
  "invoice_no": "string | null",
  "invoice_date": "string | null",
  "currency": "string | null",
  "seller": {
    "name": "string | null",
    "address": "string | null",
    "tax_id": "string | null",
    "iban": "string | null"
  },
  "buyer": {
    "name": "string | null",
    "address": "string | null",
    "tax_id": "string | null"
  },
  "total_net": "string | null",
  "total_tax": "string | null",
  "total_gross": "string | null"
}
```

## Training target format

For `PaddleOCR-VL + LoRA`, the training target should be a strict JSON string:

```json
{
  "document_type": "passport",
  "fields": {
    "document_number": "910239248",
    "surname": "OBAMA",
    "given_names": "MICHELLE"
  }
}
```

This makes the model learn structured extraction rather than plain OCR text generation.

## Recommended first dataset size

Pilot version:

- `10-20` samples per identity-document class
- `20-30` invoice samples

Stronger diploma version:

- `30-50` passports
- `30-50` ID cards
- `30-50` driver licenses
- `50-100` invoices

## Annotation rules

- Keep field names stable across the dataset.
- Use `null` when a field is not visible or absent.
- Keep date formatting consistent inside one document type.
- Do not mix OCR noise into labels; labels should contain the correct normalized values.
