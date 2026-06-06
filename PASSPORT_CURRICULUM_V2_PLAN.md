# Passport Curriculum V2 Plan

Date: 2026-05-20

## Why `v2` exists

The original `passport_curriculum_v1` is useful for experimentation, but it has a benchmarking problem:

- `passport-dataset_000` appears in both train and eval
- `passport-dataset_002` appears in both train and eval
- `ud-synthetic__synthetic-russian-passports_001` shares the same image with `ud-synthetic__synthetic-russian-passports_001_eval`

So `v2` is built as a cleaner next step:

- remove leaked train rows
- keep the strongest non-leaked passport base rows
- add new manually curated diverse passports

## Leak-free base rows kept from `v1`

- `passport-dataset_001`
- `passport-dataset_003`
- `passport-dataset_004`
- `ud-synthetic__synthetic-french-passports_000`
- `ud-synthetic__synthetic-russian-passports_000`

## New rows promoted into train

- `ud-synthetic__synthetic-russian-passports_002`
- `ud-synthetic__synthetic-russian-passports_003`
- `ud-synthetic__synthetic-french-passports_004`

## Promotion order used

1. `ud-synthetic__synthetic-russian-passports_002`
Reason: clean full-layout Russian passport with strong visible structured fields and MRZ.

2. `ud-synthetic__synthetic-russian-passports_003`
Reason: another high-quality Russian layout with different surname/given-name combination and dates.

3. `ud-synthetic__synthetic-french-passports_004`
Reason: adds a second French-style passport to the training branch, improving non-UAE diversity.

## Rows intentionally not promoted yet

- `ud-synthetic__synthetic-russian-passports_004`
Reason: strong candidate, but reserved as the next clean passport holdout/eval option.

- `ud-synthetic__synthetic-french-passports_002`
Reason: already used as holdout/eval material.

- `ud-synthetic__synthetic-japanese-passports_002`
Reason: useful future holdout candidate, but current heuristic draft is too weak for direct promotion.

- `ud-synthetic__synthetic-printed-usa-passports_001`
Reason: too noisy and suspicious for direct train promotion.

## Recommended next Kaggle experiment

1. Train on `passport_curriculum_train_v2.jsonl`
2. Keep `passport_michelle_obama` as validation
3. Do not compare `v2` against the old leaked eval without explicitly calling out the leakage
4. Build a dedicated leak-free `passport_eval_v2` after this train split is locked
