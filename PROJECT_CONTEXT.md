# PROJECT_CONTEXT.md

## 1. Project Overview

This repository contains a research and engineering project focused on **automatic extraction of structured data from identity documents**, with the current strongest emphasis on **passports**.

The project began as a broader multi-document identity extraction effort that included:

- passports
- driver licenses
- mixed identity-document evaluation

Over time, experiments showed that the passport branch is currently the most promising and coherent research direction. As a result, the project has evolved toward a more focused strategy:

1. **multi-country synthetic passport pretrain**
2. **small curated Russian finetune**
3. **parser / postprocessing at inference time**

In parallel, there is also an alternative branch:

1. **MRZ-first extraction**
2. parse MRZ into fields using deterministic logic

This file is intended to help another AI agent or developer understand:

- what the project is trying to achieve
- what has already been implemented
- which experiments have succeeded or failed
- what the main bottlenecks are
- what should likely be done next


## 2. Main Goal

The practical goal is **not** generic OCR. The real target is:

> Given an image of a document, produce a **normalized structured representation** of the important fields.

For passports, this means extracting fields such as:

- `document_number`
- `surname`
- `given_names`
- `nationality`
- `date_of_birth`
- `sex`
- `date_of_issue`
- `date_of_expiry`
- `issuing_authority`
- possibly `place_of_birth`
- and related fields depending on the source schema

The project therefore sits at the intersection of:

- OCR
- document understanding
- multimodal instruction-following
- structured generation
- rule-based postprocessing


## 3. Important Historical Context

The project has already gone through several phases:

### Phase A: Broad identity-document extraction

Early work explored:

- mixed identity-document datasets
- passports + synthetic driver licenses
- heuristic OCR pipelines
- learned structured extraction with fine-tuning

This phase was useful because it established:

- zero-shot structured extraction is poor
- heuristics can be surprisingly strong
- learned extraction can improve, but only with careful data and decoding

### Phase B: Passport-specific curriculum

The project then narrowed toward passport-only training and evaluation.

This led to one of the first clearly positive findings:

- a **passport-specific curriculum** worked better than another tiny mixed identity retrain

### Phase C: Transfer-learning strategy

The current mainline idea is:

- pretrain on synthetic multi-country passports
- then adapt to Russian data
- then use parser/postprocessing

This is the most important active research strategy in the repository.

### Phase D: MRZ-first alternative

Because full-field generation is still fragile, an alternative branch was created:

- extract only MRZ first
- derive fields from MRZ using deterministic rules

This is a more constrained but potentially more stable subproblem.


## 4. Why the Current Strategy Was Chosen

The current strategy was not chosen arbitrarily. It emerged from repeated negative and mixed results.

### What did not work well

- Pure zero-shot multimodal extraction from images
- Blindly increasing epochs on tiny passport subsets
- Assuming lower validation loss would automatically improve real extraction quality
- Treating all identity documents as one homogeneous training problem
- Relying on raw model outputs without parser/postprocessing

### What did show promise

- Family-specific curriculum learning
- Passport-first specialization
- Parser-assisted recovery
- Structured prompt constraints
- Cleaner train/val split design
- Transfer learning from synthetic multi-country passports
- Explicit evaluation across multiple holdout sets

The result is the current research direction:

> broad passport layout learning first, Russian adaptation second, structure recovery third.


## 5. Core Research Question

The central research question is roughly:

> How can we build a practical and reasonably robust system for structured extraction of passport data from images when available supervised data are small, noisy, synthetic, multilingual, and layout-diverse?

This question naturally leads to several subquestions:

- Does multi-country pretraining help Russian passport extraction?
- Does excluding Russian rows from broad pretraining improve the transfer story?
- Are printed passport variants helpful or harmful?
- Is full-field extraction feasible with current data?
- Is MRZ-first a stronger and more stable path?
- How much improvement comes from the model itself versus parser/postprocessing?


## 6. High-Level Architecture of the Current System

The project does not implement only one monolithic model. It is better understood as a **research pipeline** with several layers.

### 6.1 Data layer

This layer handles:

- importing synthetic passport datasets
- normalizing metadata
- generating train/val/eval JSONL files
- validating image paths
- checking train/val overlap
- profiling dataset composition

### 6.2 Training layer

This layer handles:

- multimodal model fine-tuning
- full-field extraction target format
- MRZ-only target format
- curriculum variants
- transfer-learning variants
- local and Kaggle runners

### 6.3 Inference layer

This layer handles:

- raw model generation
- schema-constrained prompting
- parser/postprocessing
- MRZ parsing
- heuristic fallback or hybrid recovery

### 6.4 Evaluation layer

This layer handles:

- exact field metrics
- normalized metrics
- per-field breakdown
- multiple holdout sets
- run-to-run comparison

### 6.5 Experiment orchestration layer

This layer handles:

- rebuilding datasets
- validating data
- launching transfer experiments
- launching MRZ experiments
- collecting evaluation outputs


## 7. Technical Stack

The project uses a Python-based ML and research stack.

Important technologies include:

- Python
- PyTorch
- Hugging Face Transformers
- PaddleOCR / PaddleOCR-VL
- LoRA fine-tuning
- Pillow
- OpenCV
- JSONL-based dataset representation
- Kaggle for heavier training runs

Also used:

- ad hoc parsers and deterministic normalizers
- evaluation scripts
- runner scripts for local and Kaggle workflows


## 8. Main Model Families Used

### 8.1 PaddleOCR-VL branch

This is the main passport-oriented multimodal extraction branch at the current stage.

It has been used for:

- passport curriculum runs
- mixed identity runs
- transfer-learning branches
- MRZ-first preparation

### 8.2 Earlier Donut-related work

The repository also contains older document extraction work on other document types, including an invoice-oriented Donut branch. This is historically useful, but it is not the main active branch for passport extraction.


## 9. Data Sources Used for the Passport Branch

The original idea was to rely on `UniData synthetic-passports`, but this had to be revised.

### 9.1 UniData finding

After inspection of the public HF mirror:

- images were present
- some descriptive CSV metadata were present
- but the public mirror did **not** provide the field-level metadata required for supervised field extraction

Conclusion:

- the **idea** of UniData-style pretraining remains good
- but the currently public version is **not** sufficient as a direct training source for this pipeline

### 9.2 Actual synthetic passport sources used

The project uses open `ud-synthetic` preview-style passport datasets where metadata is available alongside images.

Sources:

- `ud-synthetic/synthetic-russian-passports`
- `ud-synthetic/synthetic-japanese-passports`
- `ud-synthetic/synthetic-french-passports`
- `ud-synthetic/synthetic-printed-usa-passports`
- `ud-synthetic/synthetic-printed-german-passports`
- `ud-synthetic/synthetic-turkish-passports`
- `ud-synthetic/synthetic-greek-passports`
- `ud-synthetic/synthetic-indian-passports`
- `ud-synthetic/synthetic-chinese-passports`

These sources are small and imperfect, but they are the best currently integrated open passport sources in this repo.


## 10. Important Data Artifacts

### 10.1 Unified synthetic source

Prepared output:

- `E:\thesis\data\multidoc\passport_hf_synthetic_source_v1.jsonl`

Current size:

- `45` passport rows

### 10.2 Profile of unified source

- `E:\thesis\data\multidoc\passport_hf_synthetic_source_v1_profile.json`

This captures:

- source composition
- printed vs non-printed splits
- country/source counts


## 11. Transfer-Learning Split Families

Several split families exist. These are central to the current research strategy.

### 11.1 Version v1

Idea:

- broad pretrain on all synthetic passport sources
- includes Russian rows
- includes printed rows

Current sizes:

- `pretrain_train = 36`
- `pretrain_val = 9`
- `russian_finetune_train = 3`
- `russian_finetune_val = 1`

Interpretation:

- broadest option
- least clean transfer story
- useful as baseline/ablation

### 11.2 Version v2

Idea:

- broad pretrain only on non-Russian synthetic passports
- printed rows still allowed
- Russian rows reserved for adaptation stage

Current sizes:

- `pretrain_train = 32`
- `pretrain_val = 8`
- `russian_finetune_train = 3`
- `russian_finetune_val = 1`

Interpretation:

- cleaner transfer-learning story than v1
- still broader than v3

### 11.3 Version v3

Idea:

- broad pretrain only on non-Russian and non-printed synthetic passports
- Russian adaptation isolated

Current sizes:

- `pretrain_train = 24`
- `pretrain_val = 6`
- `russian_finetune_train = 3`
- `russian_finetune_val = 1`

Interpretation:

- strictest and cleanest transfer baseline
- currently the preferred conceptual mainline

### 11.4 Why three versions exist

They allow the following comparisons:

- `v1` vs `v2`
  - tests whether Russian overlap in broad pretrain is doing hidden work

- `v2` vs `v3`
  - tests whether printed passport layouts help or hurt transfer

- `v1` vs `v3`
  - contrasts broad mixed pretraining with the cleanest transfer setup


## 12. Evaluation Sets

The project now uses more than one benchmark. This is a major improvement over earlier stages.

### 12.1 Legacy passport benchmark

A small diverse passport benchmark used historically in many iterations.

Important because:

- it is the source of the best earlier Kaggle passport metrics
- many comparisons still reference it

### 12.2 Transfer holdout

- `passport_transfer_holdout_eval_v1.jsonl`

Size:

- `7`

Purpose:

- tests the main v3 transfer story

### 12.3 Printed-shift set

- `passport_transfer_printed_shift_eval_v1.jsonl`

Size:

- `2`

Purpose:

- isolates generalization to printed passports

### 12.4 Cross-country set

- `passport_transfer_crosscountry_eval_v1.jsonl`

Size:

- `9`

Purpose:

- broader synthetic holdout across countries/layouts


## 13. Important Evaluation Philosophy

The project explicitly learned that **raw validation loss is not enough**.

As a result, evaluation now includes:

- exact metrics
- normalized metrics
- per-field breakdown
- normalized per-field breakdown
- multiple eval sets
- run comparison utilities

This was a deliberate response to the fact that:

- some runs achieved better loss but worse passport extraction
- parser improvements sometimes helped more than retraining


## 14. Current Metrics That Must Be Remembered

These are the most important numbers for context.

### 14.1 Best practical Kaggle passport full-field result so far

This is the strongest earlier passport result from the flat-format plus parser branch:

- `field_accuracy = 0.1353`
- `precision = 0.4833`
- `recall = 0.1353`
- `f1 = 0.1990`

Interpretation:

- weak in absolute terms
- but historically important because it was the best practical Kaggle passport point for this branch

### 14.2 A worse expanded passport run

A later expanded 17-sample curriculum run on Kaggle produced:

- `field_accuracy = 0.0782`
- `precision = 0.35`
- `recall = 0.0782`
- `f1 = 0.1190`

Interpretation:

- better train loss and lower validation loss did **not** produce better practical extraction
- very important negative result

### 14.3 Staged continuation run

A continuation-style resumed training attempt produced:

- `field_accuracy = 0.1171`
- `precision = 0.2833`
- `recall = 0.1171`
- `f1 = 0.1657`

Interpretation:

- continuation worked technically
- but still did not beat the best earlier passport result

### 14.4 Mixed identity stronger learned branch

One of the strongest mixed identity points:

- `field_accuracy = 0.4357`
- `precision = 0.5486`
- `recall = 0.4357`
- `f1 = 0.4843`

Heuristic branch on the same benchmark:

- `field_accuracy = 0.4071`
- `precision = 0.5286`
- `recall = 0.4071`
- `f1 = 0.4570`

Interpretation:

- the learned branch can become competitive
- but mixed identity is no longer the preferred mainline

### 14.5 Passport-only curriculum benchmark

Heuristic:

- `field_accuracy = 0.3544`
- `precision = 0.4095`
- `recall = 0.3544`
- `f1 = 0.3723`

Fine-tuned passport curriculum:

- `field_accuracy = 0.4210`
- `precision = 0.4095`
- `recall = 0.4210`
- `f1 = 0.4079`

Interpretation:

- one of the clearest positive passport-specific learning signals in the repository


## 15. Key Experiment Lessons

### 15.1 Zero-shot structured extraction is weak

The model may output:

- OCR-like fragments
- pseudo-JSON
- partial structured strings

but not a reliable full structured result.

### 15.2 Heuristics matter

The heuristic OCR-based branch is not just a temporary toy baseline. It has repeatedly been a strong practical comparator and sometimes the best production-like branch.

### 15.3 Parser/postprocessing are critical

A large portion of quality comes not from raw generation alone, but from:

- cleaning partial outputs
- parsing MRZ-like fragments
- normalizing fields
- merging heuristic signals when needed

### 15.4 Better loss does not guarantee better extraction

This is one of the most important lessons in the project.

### 15.5 Passport-specific curriculum is stronger than unfocused mixed retraining

Narrower specialization helped more than repeatedly retraining a broader but tiny identity mixture.

### 15.6 The transfer-learning idea is plausible but not yet numerically “won”

The transfer-learning infrastructure is strong and well prepared, but its final decisive benchmark win over prior practical baselines has not yet been demonstrated.


## 16. Parser and Postprocessing Layer

This layer is one of the most important practical components in the repository.

### 16.1 What it does

It performs:

- name normalization
- nationality mapping
- date normalization
- conservative MRZ-like parsing
- field recovery from partial structured output

### 16.2 Typical successful parser recoveries

Examples:

- `P<USAOBAMA<<MICHELLE...` -> `surname=OBAMA`, `given_names=MICHELLE`
- `P<FARSI<<AHMAD<AL...` -> `surname=AL FARSI`, `given_names=AHMAD AL`
- `P<RUSSOKOLOVA<<TATYANA...` -> `surname=SOKOLOVA`, `given_names=TATYANA`

### 16.3 Why this matters

The model frequently emits:

- partial MRZ
- fragments of names
- partial second-line signals

rather than a complete clean field object.

Without parser support, much of the useful signal would be lost.


## 17. MRZ-First Branch

The repository contains an alternative strategy where the model is trained to output **only MRZ** rather than the full field set.

### 17.1 Why MRZ-first exists

Full-field generation is fragile.

MRZ extraction is:

- narrower
- more structured
- easier to validate
- easier to parse deterministically

### 17.2 Fields recoverable from MRZ

Usually:

- `surname`
- `given_names`
- `document_number`
- `nationality`
- `date_of_birth`
- `sex`
- `date_of_expiry`

### 17.3 Fields poorly covered by MRZ

Usually:

- `place_of_birth`
- `issuing_authority`
- often `date_of_issue`

### 17.4 Current state

The MRZ-first branch has:

- target-format support in training
- MRZ evaluator
- local runner
- Kaggle modes

It is ready for more serious experimental comparison.


## 18. Main Scripts and Components

The following files are especially important for a new agent.

### 18.1 Planning and reports

- `E:\thesis\PASSPORT_TRANSFER_LEARNING_PLAN.md`
- `E:\thesis\PASSPORT_TRANSFER_VERSION_COMPARISON.md`
- `E:\thesis\MULTIDOC_BENCHMARK_REPORT.md`
- `E:\thesis\TRAINING_RESULTS_2026-04-17.md`

### 18.2 Data preparation

- `E:\thesis\backend\scripts\import_hf_passport_datasets.py`
- `E:\thesis\backend\scripts\build_passport_transfer_splits.py`
- `E:\thesis\backend\scripts\validate_passport_transfer_v1.py`
- `E:\thesis\backend\scripts\profile_passport_transfer_source.py`
- `E:\thesis\backend\scripts\build_passport_transfer_eval_sets.py`

### 18.3 Experiment runners

- `E:\thesis\backend\scripts\run_passport_transfer_experiment.py`
- `E:\thesis\backend\scripts\run_passport_mrz_experiment.py`
- `E:\thesis\run_passport_transfer_pipeline.ps1`

### 18.4 Evaluation

- `E:\thesis\backend\scripts\run_passport_eval_suite.py`
- `E:\thesis\backend\scripts\compare_passport_eval_runs.py`
- `E:\thesis\backend\scripts\analyze_passport_eval.py`

### 18.5 Training / Kaggle

- `E:\thesis\backend\scripts\train_paddleocr_vl_lora.py`
- `E:\thesis\kaggle\run_train_on_kaggle.py`
- `E:\thesis\kaggle\run_passport_transfer_on_kaggle.py`
- `E:\thesis\kaggle\eval_passport_flat_on_kaggle.py`
- `E:\thesis\kaggle\eval_passport_mrz_on_kaggle.py`
- `E:\thesis\kaggle\README_KAGGLE.md`


## 19. Important Design Decisions and Why They Were Made

### Decision 1: Keep parser/postprocessing as a first-class layer

Why:

- raw model output is too unstable
- parser can recover real value from partial structured signals
- metrics improved in multiple branches due to parser work alone

### Decision 2: Move away from broad blind mixed identity retraining

Why:

- tiny mixed retrains often underperformed
- passport-specific training showed clearer gains
- the task family is too heterogeneous

### Decision 3: Build transfer split families v1/v2/v3

Why:

- needed a controlled way to test Russian overlap and printed-layout effects
- avoids hand-wavy claims about transfer learning

### Decision 4: Add multiple evaluation sets

Why:

- single tiny benchmark was too fragile
- needed better visibility into cross-country and printed-layout transfer

### Decision 5: Add MRZ-first branch

Why:

- full-field generation is currently fragile
- MRZ is a more realistic constrained objective

### Decision 6: Track normalized metrics

Why:

- exact match alone can be misleading
- normalized metrics better capture semantic progress


## 20. Known Problems

This section is especially important. The project is promising, but it has very real limitations.

### 20.1 Very small supervised data

Even after all improvements:

- unified source is only `45` passport rows
- Russian finetune is only `3 train / 1 val`

This is tiny for such a hard extraction problem.

### 20.2 Synthetic data limitations

Issues include:

- watermark-heavy samples
- printed styles
- synthetic appearance
- possible mismatch between visible content and metadata

### 20.3 Validation loss is not trustworthy enough as a primary success criterion

This must always be remembered.

### 20.4 Full-field generation remains unstable

Failure modes include:

- partial MRZ output
- incomplete structures
- schema drift
- hallucinated fragments
- partial field recovery without full consistency

### 20.5 Russian-specific adaptation is underpowered

The Russian subset is too small to be considered strong domain supervision.

### 20.6 Evaluation sets are still small

Even the improved holdouts remain relatively small, so metric volatility remains a concern.

### 20.7 Broad transfer branch is prepared better than it is validated

There is more infrastructure than definitive proof of success for the current transfer strategy.


## 21. What Has Been Successfully Prepared But Not Fully Proven Yet

The following parts are mature from an engineering perspective, but still need decisive experimental validation:

- transfer-learning branch `v2`
- transfer-learning branch `v3`
- MRZ-first branch
- printed-shift evaluation
- cross-country evaluation suite

This distinction matters:

- the project is not blocked by missing infrastructure
- it is blocked more by data quality/scale and by the need for careful validation


## 22. Recommended Interpretation for a New Agent

A new agent should interpret the project state as follows:

- The repository is already structured and mature enough for systematic work.
- Do not restart from “toy OCR baseline” thinking.
- Do not assume better train loss means better document extraction.
- Treat passports as the current mainline.
- Treat parser/postprocessing as essential, not optional.
- Treat MRZ-first as a serious alternative, not just an auxiliary experiment.
- Treat `v3` transfer as the cleanest conceptual baseline.
- Treat older best Kaggle passport result (`f1 = 0.1990`) as weak but still historically important.


## 23. What Should Likely Be Done Next

The most sensible next steps are:

### Priority 1: Run and compare transfer branches properly

Especially:

- `v2` full-field
- `v3` full-field
- `v3` MRZ-first

Evaluate across:

- legacy passport benchmark
- transfer holdout
- printed shift
- cross-country holdout

### Priority 2: Compare full-field vs MRZ-first honestly

This comparison may determine whether the project should continue pushing end-to-end full structure generation or move toward a more hybrid passport pipeline.

### Priority 3: Strengthen Russian finetune data

The Russian adaptation subset is probably too small to fully test the transfer hypothesis.

### Priority 4: Keep improving parser quality

Parser improvements have repeatedly produced large practical value relative to their cost.

### Priority 5: Resist blind retraining

Do not repeat “more epochs of the same thing” without a concrete hypothesis.


## 24. Practical One-Paragraph Summary

This project is a multimodal document extraction system with a current research focus on passports. The working hypothesis is that the best path is a staged pipeline: multi-country synthetic passport pretraining, followed by a small curated Russian finetune, plus parser/postprocessing at inference time. A parallel MRZ-first branch exists because full-field generation is still fragile. The repository already contains strong experimental infrastructure: multiple transfer split families (`v1/v2/v3`), multiple evaluation sets, data validators, local and Kaggle runners, parser-enhanced evaluators, comparison tools, and MRZ parsing logic. Historically, the best practical Kaggle passport full-field result remained weak (`field_accuracy = 0.1353`, `f1 = 0.1990`), and larger synthetic curriculum runs sometimes improved loss while worsening actual extraction. This means the core bottlenecks are not training mechanics alone, but data scale, supervision quality, structured decoding fragility, and domain adaptation. The current best next step is not another blind retrain, but a disciplined comparison of transfer branches and MRZ-first against the older passport baseline.


## 25. If You Are Another AI Agent, Read These First

Recommended reading order:

1. `E:\thesis\PROJECT_CONTEXT.md`
2. `E:\thesis\PASSPORT_TRANSFER_LEARNING_PLAN.md`
3. `E:\thesis\PASSPORT_TRANSFER_VERSION_COMPARISON.md`
4. `E:\thesis\MULTIDOC_BENCHMARK_REPORT.md`
5. `E:\thesis\kaggle\README_KAGGLE.md`

Then inspect:

- `E:\thesis\backend\scripts\build_passport_transfer_splits.py`
- `E:\thesis\backend\scripts\run_passport_transfer_experiment.py`
- `E:\thesis\backend\scripts\run_passport_mrz_experiment.py`
- `E:\thesis\kaggle\eval_passport_flat_on_kaggle.py`
- `E:\thesis\kaggle\eval_passport_mrz_on_kaggle.py`

If you need a concise operational assumption:

> The repo is already well-instrumented; the main challenge now is not scaffolding, but proving that the transfer-learning and/or MRZ-first passport branch can beat the weak but historically best passport baseline in a stable and interpretable way.
