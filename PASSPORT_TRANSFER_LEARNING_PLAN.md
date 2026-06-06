# Passport Transfer Learning Plan

Date fixed: 2026-05-21

## Working strategy

For the next passport branch we use a three-stage scheme:

1. `multi-country synthetic passport pretrain`
2. `small curated Russian finetune`
3. `parser / postprocessing` at inference time

## Important note about UniData

The original hypothesis was:

- pretrain on `UniData synthetic-passports`
- then finetune on a Russian subset

After checking the public Hugging Face mirror we found:

- [UniDataPro/synthetic-passports](https://huggingface.co/datasets/UniDataPro/synthetic-passports) in public form does contain images
- and a CSV describing files/backgrounds
- but it does **not** contain the field-level metadata needed for our supervised extraction pipeline

So the conclusion is:

- the broader UniData idea is still good in spirit
- the currently public HF mirror is **not** a ready train source for our task

## What we use instead

We use open `ud-synthetic` passport previews where metadata is stored next to the images:

- `ud-synthetic/synthetic-russian-passports`
- `ud-synthetic/synthetic-japanese-passports`
- `ud-synthetic/synthetic-french-passports`
- `ud-synthetic/synthetic-printed-usa-passports`
- `ud-synthetic/synthetic-printed-german-passports`
- `ud-synthetic/synthetic-turkish-passports`
- `ud-synthetic/synthetic-greek-passports`
- `ud-synthetic/synthetic-indian-passports`
- `ud-synthetic/synthetic-chinese-passports`

## What is already prepared

### Unified source

Importer:

- [import_hf_passport_datasets.py](/E:/thesis/backend/scripts/import_hf_passport_datasets.py)

Outputs:

- [passport_hf_synthetic_source_v1.jsonl](/E:/thesis/data/multidoc/passport_hf_synthetic_source_v1.jsonl)

Current size:

- `45` passport rows

### Transfer splits

Builder:

- [build_passport_transfer_splits.py](/E:/thesis/backend/scripts/build_passport_transfer_splits.py)

Validator:

- [validate_passport_transfer_v1.py](/E:/thesis/backend/scripts/validate_passport_transfer_v1.py)

Outputs:

- [passport_pretrain_hf_train_v1.jsonl](/E:/thesis/data/multidoc/passport_pretrain_hf_train_v1.jsonl)
- [passport_pretrain_hf_val_v1.jsonl](/E:/thesis/data/multidoc/passport_pretrain_hf_val_v1.jsonl)
- [passport_russian_finetune_train_v1.jsonl](/E:/thesis/data/multidoc/passport_russian_finetune_train_v1.jsonl)
- [passport_russian_finetune_val_v1.jsonl](/E:/thesis/data/multidoc/passport_russian_finetune_val_v1.jsonl)
- [passport_transfer_v1_summary.json](/E:/thesis/data/multidoc/passport_transfer_v1_summary.json)

Current split sizes:

- `pretrain_train = 36`
- `pretrain_val = 9`
- `russian_finetune_train = 3`
- `russian_finetune_val = 1`

### Cleaner transfer variant `v2`

There is also a cleaner variant where Russian rows are removed from the broad pretrain stage and kept only for Russian adaptation.

Outputs:

- [passport_pretrain_hf_train_v2.jsonl](/E:/thesis/data/multidoc/passport_pretrain_hf_train_v2.jsonl)
- [passport_pretrain_hf_val_v2.jsonl](/E:/thesis/data/multidoc/passport_pretrain_hf_val_v2.jsonl)
- [passport_russian_finetune_train_v2.jsonl](/E:/thesis/data/multidoc/passport_russian_finetune_train_v2.jsonl)
- [passport_russian_finetune_val_v2.jsonl](/E:/thesis/data/multidoc/passport_russian_finetune_val_v2.jsonl)

Current split sizes:

- `v2 pretrain_train = 32`
- `v2 pretrain_val = 8`
- `v2 russian_finetune_train = 3`
- `v2 russian_finetune_val = 1`

This `v2` setup is the preferred next baseline because it keeps the transfer story cleaner:

1. learn passport structure on non-Russian synthetic passports
2. adapt on the small curated Russian subset

### Stricter transfer variant `v3`

There is also a stricter variant where the broad pretrain stage keeps only non-Russian and non-printed passports.

Outputs:

- [passport_pretrain_hf_train_v3.jsonl](/E:/thesis/data/multidoc/passport_pretrain_hf_train_v3.jsonl)
- [passport_pretrain_hf_val_v3.jsonl](/E:/thesis/data/multidoc/passport_pretrain_hf_val_v3.jsonl)
- [passport_russian_finetune_train_v3.jsonl](/E:/thesis/data/multidoc/passport_russian_finetune_train_v3.jsonl)
- [passport_russian_finetune_val_v3.jsonl](/E:/thesis/data/multidoc/passport_russian_finetune_val_v3.jsonl)

Current split sizes:

- `v3 pretrain_train = 24`
- `v3 pretrain_val = 6`
- `v3 russian_finetune_train = 3`
- `v3 russian_finetune_val = 1`

This `v3` setup is the cleanest transfer baseline we currently have:

1. broad pretrain on non-Russian, non-printed passport layouts
2. Russian adaptation on the curated Russian subset
3. parser / postprocessing on top

### Source profiling

Synthetic source profile:

- [passport_hf_synthetic_source_v1_profile.json](/E:/thesis/data/multidoc/passport_hf_synthetic_source_v1_profile.json)

Profiler script:

- [profile_passport_transfer_source.py](/E:/thesis/backend/scripts/profile_passport_transfer_source.py)

## Additional evaluation sets

Builder:

- [build_passport_transfer_eval_sets.py](/E:/thesis/backend/scripts/build_passport_transfer_eval_sets.py)

Outputs:

- [passport_transfer_holdout_eval_v1.jsonl](/E:/thesis/data/multidoc/passport_transfer_holdout_eval_v1.jsonl)
- [passport_transfer_printed_shift_eval_v1.jsonl](/E:/thesis/data/multidoc/passport_transfer_printed_shift_eval_v1.jsonl)
- [passport_transfer_crosscountry_eval_v1.jsonl](/E:/thesis/data/multidoc/passport_transfer_crosscountry_eval_v1.jsonl)

Current sizes:

- `transfer_holdout = 7`
- `printed_shift = 2`
- `crosscountry = 9`

Meaning:

- `transfer_holdout` checks the main `v3` story: non-printed cross-country holdout plus Russian holdout.
- `printed_shift` isolates the shift to printed passports that are excluded from `v3` pretrain.
- `crosscountry` is the broadest synthetic holdout among the currently prepared transfer eval sets.

## Eval suite

Runner:

- [run_passport_eval_suite.py](/E:/thesis/backend/scripts/run_passport_eval_suite.py)

Use it when we want to evaluate one adapter on:

- the legacy 5-sample passport benchmark
- the transfer holdout set
- the printed-shift set
- the broader cross-country set

## How to run locally

One-shot local runner:

- [run_passport_transfer_experiment.py](/E:/thesis/backend/scripts/run_passport_transfer_experiment.py)

Recommended full run:

```powershell
E:\thesis\.conda311\python.exe E:\thesis\backend\scripts\run_passport_transfer_experiment.py --prepare-data
```

This will:

1. rebuild the multi-country source
2. rebuild the transfer splits
3. validate image paths and split overlap
4. run multi-country pretrain
5. run Russian finetune
6. run the passport evaluator on the finetuned adapter

## How to run on Kaggle

### Stage 1: synthetic passport pretrain

```python
!python kaggle/run_train_on_kaggle.py --mode passport_pretrain_hf_v1 --epochs 6
```

### Stage 2: Russian passport finetune

```python
!python kaggle/run_train_on_kaggle.py \
  --mode passport_russian_finetune_v1 \
  --epochs 4 \
  --resume-from-adapter /kaggle/working/thesis_bundle/outputs/paddleocr_vl_passport_pretrain_hf_v1_kaggle
```

### Stage 3: evaluation

```python
!PASSPORT_ADAPTER_DIR=/kaggle/working/thesis_bundle/outputs/paddleocr_vl_passport_russian_finetune_v1_kaggle \
 CUDA_VISIBLE_DEVICES=0 python kaggle/eval_passport_flat_on_kaggle.py
```

## Honest limitations

- Open `ud-synthetic` previews are tiny: `5` images per country.
- This is still much better than a tiny Russian-only curriculum with no pretrain.
- But it is still **not** a large production-like passport dataset.

So this branch should be treated as:

- not a final production solution
- but a much more sensible transfer-learning baseline for our passport extraction task
