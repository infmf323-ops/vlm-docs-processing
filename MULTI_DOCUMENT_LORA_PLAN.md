# Multi-Document LoRA Plan

## Goal

Adapt `PaddleOCR-VL` to a multi-document extraction setup that supports:

- `passport`
- `id_card`
- `driver_license`
- `invoice`

The diploma-friendly comparison should include three stages:

1. `Out of the box`
2. `Heuristic pipeline`
3. `Fine-tuned multi-document model`

## Why this path

- `Out of the box` demonstrates the baseline capability of a modern foundation model.
- `Heuristic pipeline` demonstrates the engineering contribution: OCR + field mapping + validation.
- `LoRA fine-tuning` demonstrates the ML contribution: adapting the model itself to the target domain.

This gives a stronger дипломный narrative than inference-only evaluation.

## Recommended training format

Each training sample should include:

- source image
- `document_type`
- target structured JSON

Example:

```json
{
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

## Minimal dataset plan

Recommended first milestone:

- `30-50` passports
- `30-50` ID cards
- `30-50` driver licenses
- `50-100` invoices

If full annotation is expensive, a smaller pilot set is still useful:

- `10-20` samples per identity-document class
- `20-30` invoice samples

## Fine-tuning method

Use parameter-efficient fine-tuning:

- `LoRA`
- mixed precision on GPU
- no full fine-tuning of all weights at the first stage

Why:

- practical on `RTX 3080`
- lower VRAM pressure
- easier iteration
- strong enough for a diploma experiment

## Benchmark protocol

For each document type compare:

### Stage 1. Out of the box

- raw `PaddleOCR-VL`
- no custom parsing
- direct extraction / OCR output only

### Stage 2. Heuristic pipeline

- `PaddleOCR-VL`
- custom field mapping
- validation
- normalization

### Stage 3. Fine-tuned model

- `PaddleOCR-VL + LoRA`
- same evaluation dataset
- same schema

## Metrics

Recommended metrics for extraction:

- `Field Accuracy`
- `F1-score`
- `TEDS`
- optional: `Reading Order Accuracy`

For document-by-document inspection also keep:

- raw OCR text
- normalized output
- validation issues

## Immediate next steps

1. Stabilize the heuristic pipeline for `passport`, `id_card`, and `driver_license`.
2. Prepare a small multi-document labeled set in the unified JSON schema.
3. Create a direct evaluation script for `out of the box` vs `heuristic`.
4. Add a LoRA training script for `PaddleOCR-VL`.
5. Run the final three-way comparison for the diploma:
   - `out of the box`
   - `heuristic pipeline`
   - `fine-tuned multi-document model`
