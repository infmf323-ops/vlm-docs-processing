# Multi-Document Annotation Guide

## What each file is for

- `E:/thesis/data/multidoc/train.jsonl`
  - verified records that are safe to use for training
- `E:/thesis/data/multidoc/val.jsonl`
  - verified records that are safe to use for validation
- `E:/thesis/data/multidoc/templates.jsonl`
  - schema examples and placeholders; **not** for training
- `E:/thesis/data/multidoc/pilot_train.jsonl`
  - filtered subset built automatically from verified records
- `E:/thesis/data/multidoc/pilot_val.jsonl`
  - filtered validation subset built automatically from verified records
- `E:/thesis/data/multidoc/annotation_queue.jsonl`
  - documents waiting for review or manual annotation

## Current workflow

1. Put a real document image into the dataset or point to an existing file.
2. Add or review the record in `annotation_queue.jsonl`.
3. When the fields are correct, move that record into `train.jsonl` or `val.jsonl`.
4. Rebuild `pilot_train.jsonl` and `pilot_val.jsonl`.
5. Run validation before training.

## Record statuses in `annotation_queue.jsonl`

- `needs_annotation`
  - image is known, fields are still empty
- `draft_from_previous_experiment`
  - fields were bootstrapped from historical ground truth and need confirmation
- `needs_annotation_external`
  - image was imported from an external dataset source and still needs full manual field annotation
- `verified_reference`
  - fields are already checked and can be copied into `train.jsonl` or `val.jsonl`

## External bootstrap sources

External identity-document samples are staged under:

- `E:/thesis/data/multidoc/external/ud_biometrics`

These rows are useful for:

- expanding the annotation queue
- testing OCR / heuristic behavior on new document appearances
- creating future verified `passport` and `driver_license` records

They are **not** training data until their fields are manually verified and promoted into `train.jsonl` or `val.jsonl`.

## Helpful commands

Validate the current dataset:

```powershell
E:\thesis\.conda311\python.exe E:\thesis\backend\scripts\validate_multidoc_dataset.py
```

Validate dataset and templates:

```powershell
E:\thesis\.conda311\python.exe E:\thesis\backend\scripts\validate_multidoc_dataset.py --include-templates
```

Rebuild clean train/val/templates files:

```powershell
E:\thesis\.conda311\python.exe E:\thesis\backend\scripts\bootstrap_multidoc_dataset.py
```

Rebuild the annotation queue:

```powershell
E:\thesis\.conda311\python.exe E:\thesis\backend\scripts\build_annotation_queue.py
```

Rebuild pilot splits:

```powershell
E:\thesis\.conda311\python.exe E:\thesis\backend\scripts\build_multidoc_pilot_split.py
```

## Practical note

For the diploma, the strongest story is:

- `out of the box` foundation model
- `heuristic pipeline` on top of the model
- `LoRA` fine-tuned multi-document model

That means the next high-value work is not more architecture changes, but adding a few verified records for `invoice`, `passport`, `id_card`, and `driver_license`.
