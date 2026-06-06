# Kaggle Launch Guide

This folder contains the smallest practical setup for running the current thesis experiments on Kaggle instead of the local Windows machine.

## What you will upload

Build the bundle locally first:

```powershell
E:\thesis\.conda311\python.exe E:\thesis\backend\scripts\build_kaggle_bundle.py
```

This creates:

- [kaggle_bundle.zip](/E:/thesis/artifacts/kaggle_bundle.zip)

Upload that zip as a Kaggle Notebook dataset or add it directly to the notebook workspace.

## Recommended Kaggle notebook settings

- Accelerator: `GPU`
- Internet: `On`
- Python notebook

Internet is useful because `PaddleOCR-VL` weights will be downloaded from Hugging Face the first time.

## First notebook cell

```python
!unzip -q /kaggle/input/YOUR_DATASET_NAME/kaggle_bundle.zip -d /kaggle/working/thesis_bundle
%cd /kaggle/working/thesis_bundle
!python -m pip install -q -r kaggle/requirements-kaggle.txt
```

If you uploaded the unzipped folder instead of the zip, just set:

```python
%cd /kaggle/input/YOUR_DATASET_NAME
```

## Training commands

### 1. Passport-first curriculum

```python
!python kaggle/run_train_on_kaggle.py --mode passport_curriculum --epochs 3
```

### 1b. Passport-first curriculum with flat text targets

```python
!python kaggle/run_train_on_kaggle.py --mode passport_curriculum_flat --epochs 3
```

### 1c. Passport-first curriculum `v2` (leak-reduced, expanded diversity)

```python
!python kaggle/run_train_on_kaggle.py --mode passport_curriculum_flat_v2 --epochs 6
```

### 1d. Passport-first curriculum `v3` (`17` train passports, wider layout diversity)

```python
!python kaggle/run_train_on_kaggle.py --mode passport_curriculum_flat_v3 --epochs 6
```

### 1e. Multi-country synthetic passport pretrain

```python
!python kaggle/run_train_on_kaggle.py --mode passport_pretrain_hf_v1 --epochs 6
```

### 1f. Russian passport finetune after synthetic pretrain

```python
!python kaggle/run_train_on_kaggle.py \
  --mode passport_russian_finetune_v1 \
  --epochs 4 \
  --resume-from-adapter /kaggle/working/thesis_bundle/outputs/paddleocr_vl_passport_pretrain_hf_v1_kaggle
```

### 1g. Non-Russian passport pretrain `v2`

This is the cleaner transfer setup: Russian passports are removed from the broad pretrain stage and reserved for the Russian finetune stage.

```python
!python kaggle/run_train_on_kaggle.py --mode passport_pretrain_hf_v2 --epochs 6
```

### 1h. Russian passport finetune after non-Russian pretrain `v2`

```python
!python kaggle/run_train_on_kaggle.py \
  --mode passport_russian_finetune_v2 \
  --epochs 4 \
  --resume-from-adapter /kaggle/working/thesis_bundle/outputs/paddleocr_vl_passport_pretrain_hf_v2_kaggle
```

### 1i. Stricter non-Russian, non-printed passport pretrain `v3`

This is the cleanest current transfer baseline: pretrain only on non-Russian and non-printed passport layouts, then adapt on Russian passports.

```python
!python kaggle/run_train_on_kaggle.py --mode passport_pretrain_hf_v3 --epochs 6
```

### 1j. Russian passport finetune after stricter pretrain `v3`

```python
!python kaggle/run_train_on_kaggle.py \
  --mode passport_russian_finetune_v3 \
  --epochs 4 \
  --resume-from-adapter /kaggle/working/thesis_bundle/outputs/paddleocr_vl_passport_pretrain_hf_v3_kaggle
```

### 1k. Full passport transfer pipeline in one command

```python
!python kaggle/run_passport_transfer_on_kaggle.py
```

### 1l. MRZ-first passport pretrain `v3`

This branch trains the model on a simpler target: extract only the passport MRZ, not the full flat field list.

```python
!python kaggle/run_train_on_kaggle.py --mode passport_pretrain_hf_mrz_v3 --epochs 6
```

### 1m. MRZ-first Russian finetune `v3`

```python
!python kaggle/run_train_on_kaggle.py \
  --mode passport_russian_finetune_mrz_v3 \
  --epochs 4 \
  --resume-from-adapter /kaggle/working/thesis_bundle/outputs/paddleocr_vl_passport_pretrain_hf_mrz_v3_kaggle
```

### 2. Driver-license-first curriculum

```python
!python kaggle/run_train_on_kaggle.py --mode driver_license_curriculum --epochs 4
```

### 3. Mixed identity baseline

```python
!python kaggle/run_train_on_kaggle.py --mode mixed_identity --epochs 2
```

## Evaluation commands

### Passport curriculum evaluation

```python
!python kaggle/run_eval_on_kaggle.py \
  --mode passport_curriculum \
  --adapter-dir /kaggle/working/thesis_bundle/outputs/paddleocr_vl_passport_curriculum_kaggle \
  --output /kaggle/working/passport_curriculum_eval.json
```

### Passport flat-format evaluation

```python
!CUDA_VISIBLE_DEVICES=0 python kaggle/eval_passport_flat_on_kaggle.py
```

Evaluate a specific passport adapter explicitly:

```python
!PASSPORT_ADAPTER_DIR=/kaggle/working/thesis_bundle/outputs/paddleocr_vl_passport_russian_finetune_v1_kaggle \
 CUDA_VISIBLE_DEVICES=0 python kaggle/eval_passport_flat_on_kaggle.py
```

Analyze the saved passport eval JSON:

```python
!python backend/scripts/analyze_passport_eval.py --input /kaggle/working/thesis_bundle/passport_transfer_eval_kaggle.json
```

Run one passport adapter across multiple eval sets:

```python
!python backend/scripts/run_passport_eval_suite.py \
  --adapter-dir /kaggle/working/thesis_bundle/outputs/paddleocr_vl_passport_russian_finetune_v3_kaggle \
  --output-dir /kaggle/working/passport_eval_suite_v3 \
  --mode flat
```

### MRZ-first passport evaluation

```python
!PASSPORT_ADAPTER_DIR=/kaggle/working/thesis_bundle/outputs/paddleocr_vl_passport_russian_finetune_mrz_v3_kaggle \
 CUDA_VISIBLE_DEVICES=0 python kaggle/eval_passport_mrz_on_kaggle.py
```

This evaluator now includes built-in passport-specific recovery for partial generations:

- compact MRZ-like first-line parsing,
- recovery of `surname` and `given_names` from `P<...<<...` outputs,
- nationality recovery for common issuer codes,
- lightweight date and passport-number fallback when the model emits only fragments.

## Recommended staged passport training on Kaggle

If the basic `passport_curriculum_flat` run is stable on your Kaggle GPU, prefer staged continuation over simply increasing epochs forever on the same setup.

### Stage 1: stable base adapter

```python
!python kaggle/run_train_on_kaggle.py --mode passport_curriculum_flat --epochs 6
```

### Stage 2: continue from the saved adapter with a larger visual budget

```python
!python kaggle/run_train_on_kaggle.py \
  --mode passport_curriculum_flat \
  --epochs 4 \
  --max-image-side 896 \
  --max-length 2560 \
  --max-new-tokens 224 \
  --learning-rate 1e-4 \
  --resume-from-adapter /kaggle/working/thesis_bundle/outputs/paddleocr_vl_passport_curriculum_flat_kaggle
```

### Stage 3: optional final polishing pass

```python
!python kaggle/run_train_on_kaggle.py \
  --mode passport_curriculum_flat \
  --epochs 2 \
  --max-image-side 1024 \
  --max-length 3072 \
  --max-new-tokens 256 \
  --learning-rate 5e-5 \
  --resume-from-adapter /kaggle/working/thesis_bundle/outputs/paddleocr_vl_passport_curriculum_flat_kaggle
```

Why this usually works better:

- early epochs teach the adapter the output structure;
- later stages spend extra GPU budget on finer text detail instead of relearning the basics from scratch;
- lowering the learning rate during continuation reduces destructive drift on tiny passport subsets.

### Driver-license curriculum evaluation

```python
!python kaggle/run_eval_on_kaggle.py \
  --mode driver_license_curriculum \
  --adapter-dir /kaggle/working/thesis_bundle/outputs/paddleocr_vl_driver_license_curriculum_kaggle \
  --output /kaggle/working/driver_license_curriculum_eval.json
```

### Mixed identity evaluation

```python
!python kaggle/run_eval_on_kaggle.py \
  --mode mixed_identity \
  --adapter-dir /kaggle/working/thesis_bundle/outputs/paddleocr_vl_mixed_identity_kaggle \
  --output /kaggle/working/mixed_identity_eval.json
```

## Which run to start with

Start with:

1. `passport_curriculum_flat`
2. `passport_curriculum_flat_v2`
3. `passport_curriculum_flat_v3`
4. `passport_pretrain_hf_v1`
5. `passport_russian_finetune_v1`
6. `passport_pretrain_hf_v2`
7. `passport_russian_finetune_v2`
8. `passport_pretrain_hf_v3`
9. `passport_russian_finetune_v3`
10. `passport_pretrain_hf_mrz_v3`
11. `passport_russian_finetune_mrz_v3`
12. `driver_license_curriculum`
13. `mixed_identity`

That order matches what already worked best locally:

- passport curriculum is the strongest positive learned result so far;
- passport flat curriculum is the next preferred Kaggle experiment because the JSON target format proved too brittle during generation;
- passport flat curriculum `v2` is the recommended next step once the new bundle is uploaded, because it removes known train/eval overlap and adds manually curated Russian/French diversity;
- passport flat curriculum `v3` is the next stronger passport-data experiment, because it increases the train subset to `17` passports and adds Japanese, German, printed USA, and another Russian layout family without overlapping the current passport eval split;
- `passport_pretrain_hf_v1` is the first transfer-learning variant that uses a broader multi-country synthetic passport source before any Russian-specific adaptation;
- `passport_russian_finetune_v1` is designed to continue from that synthetic pretrain adapter and then narrow the model back toward Russian passport behavior;
- `passport_pretrain_hf_v2` is the cleaner transfer-learning baseline, because it excludes Russian rows from the broad pretrain stage and preserves them for downstream adaptation;
- `passport_russian_finetune_v2` is the preferred Russian adaptation stage to pair with that non-Russian pretrain;
- `passport_pretrain_hf_v3` is the strictest current transfer baseline, because it also removes printed passport variants from the broad pretrain stage;
- `passport_russian_finetune_v3` is the preferred Russian adaptation stage to pair with that stricter non-printed pretrain;
- `passport_pretrain_hf_mrz_v3` and `passport_russian_finetune_mrz_v3` are the main MRZ-first exploratory branch, useful when full-field generation remains too brittle;
- driver-license curriculum is now much better after OCR crop fixes;
- mixed identity is still useful, but it is the least stable branch.

## Suggested stronger Kaggle settings

Kaggle is stronger than the local 16 GB RAM machine, so these are worth trying after the first successful run:

- `--epochs 4` for passport curriculum
- `--epochs 5` for driver-license curriculum
- increase `MAX_IMAGE_SIDE` to `896`
- increase `MAX_NEW_TOKENS` to `320`

The wrappers can be overridden like this:

```python
!MAX_IMAGE_SIDE=896 MAX_NEW_TOKENS=320 python kaggle/run_train_on_kaggle.py --mode passport_curriculum --epochs 4
```

## Most important output files

After training, look at:

- `outputs/.../training_summary.json`

After evaluation, look at:

- `/kaggle/working/passport_curriculum_eval.json`
- `/kaggle/working/driver_license_curriculum_eval.json`
- `/kaggle/working/mixed_identity_eval.json`

## Notes

- The bundle already includes the curated JSONL splits and the images referenced by them.
- The first model load can take a while because Kaggle has to download the base model.
- If a run still hits memory pressure, lower:
  - `MAX_IMAGE_SIDE`
  - `MAX_LENGTH`
  - `MAX_NEW_TOKENS`
