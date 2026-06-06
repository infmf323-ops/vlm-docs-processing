# Passport Curriculum V3 Plan

Date: 2026-05-20

## Goal

Expand the passport-specific training branch from the leak-reduced `v2` subset (`8` train rows)
to a genuinely larger and more diverse `v3` subset without reintroducing overlap with the
current `passport_eval_diverse_v1.jsonl`.

## Resulting size

- `passport_curriculum_train_v3.jsonl` -> `17` train rows
- `passport_curriculum_val_v3.jsonl` -> `1` validation row

## Base rows kept from `v2`

- `passport-dataset_001`
- `passport-dataset_003`
- `passport-dataset_004`
- `ud-synthetic__synthetic-french-passports_000`
- `ud-synthetic__synthetic-russian-passports_000`
- `ud-synthetic__synthetic-russian-passports_002`
- `ud-synthetic__synthetic-russian-passports_003`
- `ud-synthetic__synthetic-french-passports_004`

## New rows promoted into `v3`

- `ud-synthetic__synthetic-russian-passports_004`
- `ud-synthetic__synthetic-japanese-passports_000`
- `ud-synthetic__synthetic-japanese-passports_001`
- `ud-synthetic__synthetic-japanese-passports_002`
- `ud-synthetic__synthetic-japanese-passports_003`
- `ud-synthetic__synthetic-japanese-passports_004`
- `ud-synthetic__synthetic-printed-usa-passports_001`
- `ud-synthetic__synthetic-printed-german-passports_001`
- `ud-synthetic__synthetic-printed-german-passports_002`

## Why these additions were chosen

1. `ud-synthetic__synthetic-russian-passports_004`
Reason: clean full Russian layout with strong visible bilingual fields and full MRZ.

2. `ud-synthetic__synthetic-japanese-passports_000` to `004`
Reason: this gives the branch a real non-Latin-layout family with consistent formatting and
clear field grounding, while still staying readable enough for manual verification.

3. `ud-synthetic__synthetic-printed-usa-passports_001`
Reason: adds a US passport layout distinct from Michelle Obama's sample and keeps the
document-family mix less tied to one rendering style.

4. `ud-synthetic__synthetic-printed-german-passports_001` and `002`
Reason: adds another European layout family that is visually clean even though the heuristic
bootstrap was weak on it.

## Rows intentionally still held out

- `passport-dataset_000`
- `passport-dataset_002`
- `ud-synthetic__synthetic-russian-passports_001`
- `ud-synthetic__synthetic-french-passports_002`

Reason: these are either part of the current passport eval set or image-overlap with it.

## Honest risk note

`v3` is much stronger than `v2` in raw passport count and layout diversity, but some of the
new synthetic families still include visible `Unidata` watermarks and a few printed layouts
that the heuristic OCR path handled badly. That is acceptable for curriculum training, but the
next benchmark should be interpreted as a data-diversity experiment, not as a final production
passport corpus.

## Recommended Kaggle experiment

1. Train `passport_curriculum_flat_v3` for `6` epochs.
2. Evaluate with the existing `eval_passport_flat_on_kaggle.py`.
3. Compare against the current best Kaggle passport point:
   - `field_accuracy = 0.1353`
   - `f1 = 0.1990`
