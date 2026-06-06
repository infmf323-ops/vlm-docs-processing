# Identity Dataset Progress

Date: 2026-05-17

## Current dataset choices

- `DocXPand-25k` remains the main candidate for a richer supervised extraction source.
- `MIDV-500` remains the preferred real-world benchmark for document tampering and identity-document scenarios.
- `ud-biometrics` is still the practical bootstrap source for real passport and driver-license images in the local pipeline.

## What has already been done

- Added local ingestion scripts for external identity samples.
- Built a heuristic prefill flow on top of `PaddleOCR-VL`.
- Improved the passport parser:
  - nationality detection
  - sex extraction
  - issue-date extraction
  - MRZ-based recovery
  - fallback parsing from name crop
- Improved the driver-license parser:
  - `LN`, `FN`, `DOB`, `EXP`, `ISS`
  - better address parsing
  - better `license_number` selection
- Extended the weakly supervised train subset with external identity records.
- Separated the base OCR path from the LoRA-adapted structured path more cleanly in the runtime and benchmark flow.

## Current train subset

The current `train.jsonl` contains 17 records:

- `4` invoice
- `8` passport
- `5` driver_license

Passport records in train:

- `passport_michelle_obama`
- `passport-dataset_000`
- `passport-dataset_001`
- `passport-dataset_002`
- `passport-dataset_003`
- `passport-dataset_004`
- `ud-synthetic__synthetic-russian-passports_000`
- `ud-synthetic__synthetic-french-passports_000`

Driver-license records in train:

- `synthetic-usa-driver-license_000`
- `synthetic-usa-driver-license_001`
- `synthetic-usa-driver-license_002`
- `synthetic-usa-driver-license_003`
- `synthetic-usa-driver-license_004`

Additional experiment-ready variants:

- [E:\thesis\data\multidoc\pilot_train_strong.jsonl](/E:/thesis/data/multidoc/pilot_train_strong.jsonl) -> `9` stronger records
- [E:\thesis\data\multidoc\pilot_train_augmented.jsonl](/E:/thesis/data/multidoc/pilot_train_augmented.jsonl) -> `10` records including the low-quality `synthetic-usa-driver-license_004`
- after label cleanup, the current `pilot_train_strong.jsonl` contains `14` records
- safe launchers:
  - [E:\thesis\run_paddleocr_vl_lora_strong_safe.ps1](/E:/thesis/run_paddleocr_vl_lora_strong_safe.ps1)
  - [E:\thesis\run_paddleocr_vl_lora_augmented_safe.ps1](/E:/thesis/run_paddleocr_vl_lora_augmented_safe.ps1)

Validation currently contains:

- `1` invoice record (`invoice_val_1`)

## Annotation queue status

The queue still contains a mix of:

- historical invoice drafts
- verified passport reference
- heuristic identity drafts
- external records still needing manual review

Current queue size:

- `41` total rows
- `15` heuristic identity drafts
- `12` external rows still waiting for manual review

Current external-source mix in the queue:

- `ud-biometrics/passport-dataset` -> `10`
- `ud-biometrics/synthetic-usa-driver-license` -> `10`
- `ud-synthetic/synthetic-printed-usa-passports` -> `3`
- `ud-synthetic/synthetic-printed-german-passports` -> `3`
- `ud-synthetic/synthetic-french-passports` -> `3`
- `ud-synthetic/synthetic-japanese-passports` -> `3`
- `ud-synthetic/synthetic-russian-passports` -> `3`

Identity heuristic candidates are also exported to:

- [E:\thesis\data\multidoc\identity_review_candidates.jsonl](/E:/thesis/data/multidoc/identity_review_candidates.jsonl)
- [E:\thesis\IDENTITY_REVIEW_CANDIDATES.md](/E:/thesis/IDENTITY_REVIEW_CANDIDATES.md)

## Diverse passport import round

To reduce overfitting to a single UAE passport layout and a single California driver-license template, the external import flow was extended to pull additional synthetic passport layouts:

- `ud-synthetic/synthetic-printed-usa-passports`
- `ud-synthetic/synthetic-printed-german-passports`
- `ud-synthetic/synthetic-french-passports`
- `ud-synthetic/synthetic-japanese-passports`
- `ud-synthetic/synthetic-russian-passports`

The practical result is mixed but useful:

- `ud-biometrics/passport-dataset` remains the strongest weak-label bootstrap source;
- `ud-synthetic/synthetic-russian-passports` unexpectedly gave the best heuristic yield among the new diverse sources;
- `ud-synthetic/synthetic-french-passports` and `ud-synthetic/synthetic-japanese-passports` produce partial drafts that are useful for review, but are still noisy;
- `ud-synthetic/synthetic-printed-german-passports` currently fails as a heuristic bootstrap source and should be treated as manual-annotation-only for now.

Current prefill outcome by external source:

- `ud-biometrics/passport-dataset` -> `6 / 10` valid heuristic drafts
- `ud-biometrics/synthetic-usa-driver-license` -> `5 / 10` valid heuristic drafts
- `ud-synthetic/synthetic-printed-usa-passports` -> `1 / 3` valid heuristic drafts
- `ud-synthetic/synthetic-printed-german-passports` -> `0 / 3` valid heuristic drafts
- `ud-synthetic/synthetic-french-passports` -> `2 / 3` valid heuristic drafts
- `ud-synthetic/synthetic-japanese-passports` -> `1 / 3` valid heuristic drafts
- `ud-synthetic/synthetic-russian-passports` -> `3 / 3` valid heuristic drafts

Interpretation:

- the new synthetic layouts are already useful as a diversity pool for manual review and curated promotion;
- however, only part of them are strong enough for automatic weak-label prefill;
- the new Russian and French samples are the most promising immediate candidates for the next curated identity subset.

Two of these diverse passports were already manually curated and promoted into the strong training subset:

- `ud-synthetic__synthetic-russian-passports_000`
- `ud-synthetic__synthetic-french-passports_000`

This gives the train split a more useful mix than before:

- older UAE-style passports,
- a US passport,
- a French passport,
- a Russian passport,
- and California-style driver licenses.

## Best current direct extraction quality

### Passport

For Michelle Obama's passport, the current heuristic pipeline already extracts:

- `document_number`
- `surname`
- `given_names`
- `nationality`
- `date_of_birth`
- `place_of_birth`
- `date_of_issue`
- `date_of_expiry`
- `issuing_authority`

For external passport samples:

- `passport-dataset_000` improved from a broken surname (`SI`) to `AL FARSI`
- `passport-dataset_002` now yields a valid compact-identity parse with:
  - `document_number`
  - `nationality`
  - `date_of_birth`
  - `sex`
  - `date_of_expiry`

### Driver license

The strongest direct result remains:

- [E:\thesis\driver_license_direct_result.json](/E:/thesis/driver_license_direct_result.json)

Correctly extracted fields include:

- `license_number`
- `surname`
- `given_names`
- `date_of_birth`
- `address`
- `date_of_issue`
- `date_of_expiry`

## LoRA training progress

Recent pilot runs:

- `pilot v1`: best val loss `0.5075`
- `pilot v2`: best val loss `0.4798`
- `pilot v4`: best val loss `0.4554`
- `pilot v5`: best val loss `0.3946`
- `pilot v6`: best val loss `0.3981`
- `pilot v7_e1` (augmented, `10` train rows, `1` epoch): best val loss `0.8741`
- `strong_safe` (`9` train rows, `1` epoch, lighter config): best val loss `1.0217`
- `augmented_safe` (`10` train rows, `1` epoch, lighter config): best val loss `1.0171`
- `strong_safe_v3data` (`14` strong train rows, `1` epoch, passport-heavy, lighter config): best val loss `0.9176`

Latest checkpoint:

- [E:\thesis\outputs\paddleocr_vl_multidoc_lora_pilot_v5](/E:/thesis/outputs/paddleocr_vl_multidoc_lora_pilot_v5)

Latest summary:

- [training_summary.json](/E:/thesis/outputs/paddleocr_vl_multidoc_lora_pilot_v5/training_summary.json)

This is a meaningful sign that the training branch is improving as the identity subset becomes less sparse. The current best checkpoint is still `pilot v5`.
The new `v7_e1` short run confirms that simply adding one low-quality weak label does not automatically improve the extraction branch.
The later `safe` comparison shows a clearer result: the low-quality additional weak label does not help in the practical extraction benchmark, even though its one-epoch validation loss is slightly lower.
The new `strong_safe_v3data` run is a different kind of signal: adding more diverse verified passport records improves validation loss substantially, but that gain has not yet translated into a better practical identity benchmark than the best `strong_safe v3` parser/runtime point.

The next diversity-follow-up run used the enlarged `17`-record strong split and produced:

- [E:\thesis\outputs\paddleocr_vl_multidoc_lora_strong_safe_v8diverse](/E:/thesis/outputs/paddleocr_vl_multidoc_lora_strong_safe_v8diverse)
- best val loss: `0.8877`

This is the best low-memory validation loss of the strong-safe branch so far, which is a useful optimization signal.
However, benchmark quality still needs to be interpreted separately from loss.

To make that clearer, a new hold-out diversity probe was created:

- [E:\thesis\data\multidoc\identity_eval_diverse_v1.jsonl](/E:/thesis/data/multidoc/identity_eval_diverse_v1.jsonl)

This adds two unseen manually curated passport layouts:

- `ud-synthetic__synthetic-russian-passports_001_eval`
- `ud-synthetic__synthetic-french-passports_002_eval`

Results on the `7`-sample diverse eval set:

- older learned checkpoint `v4clean`:
  - field accuracy: `0.3603`
  - precision: `0.3997`
  - recall: `0.3603`
  - f1: `0.3731`
- newer diversity-trained checkpoint `v8diverse`:
  - field accuracy: `0.3460`
  - precision: `0.3854`
  - recall: `0.3460`
  - f1: `0.3588`

Interpretation:

- adding the new diverse passports improved training loss,
- but the one-epoch low-memory retrain still did not beat the older learned checkpoint on the new hold-out diverse eval set,
- which means the current bottleneck is no longer just data access, but the combination of:
  - very small supervised identity subset,
  - one-epoch low-memory fine-tuning,
  - and still-fragile structured decoding on non-UAE/non-California layouts.

## Low-memory controlled comparison

To adapt the experiments to a machine with `16 GB RAM`, a lighter configuration was introduced:

- `MAX_IMAGE_SIDE = 640`
- `MAX_LENGTH = 1536`
- `MAX_NEW_TOKENS = 192`
- `NUM_EPOCHS = 1`

Benchmark artifacts:

- [E:\thesis\multidoc_benchmark_identity_strong_safe.json](/E:/thesis/multidoc_benchmark_identity_strong_safe.json)
- [E:\thesis\multidoc_benchmark_identity_augmented_safe.json](/E:/thesis/multidoc_benchmark_identity_augmented_safe.json)
- [E:\thesis\multidoc_benchmark_identity_strong_safe_v3.json](/E:/thesis/multidoc_benchmark_identity_strong_safe_v3.json)
- [E:\thesis\multidoc_benchmark_identity_strong_safe_v4.json](/E:/thesis/multidoc_benchmark_identity_strong_safe_v4.json)
- [E:\thesis\multidoc_benchmark_identity_strong_safe_v3data.json](/E:/thesis/multidoc_benchmark_identity_strong_safe_v3data.json)

Fine-tuned results under this low-memory setup:

- `strong_safe`:
  - field accuracy: `0.3171`
  - precision: `0.3771`
  - recall: `0.3171`
  - f1: `0.3371`
- `augmented_safe`:
  - field accuracy: `0.2886`
  - precision: `0.3714`
  - recall: `0.2886`
  - f1: `0.3181`

This means the current best low-memory fine-tuned identity branch is `strong_safe`, not `augmented_safe`.

Post-reboot parser refinements pushed the low-memory branch further:

- `strong_safe v3`:
  - field accuracy: `0.4057`
  - precision: `0.5157`
  - recall: `0.4057`
  - f1: `0.4524`

This was the strongest low-memory identity benchmark so far and showed that parser/runtime improvements alone can still move the fine-tuned branch upward.

An additional cleanup pass (`v4`) changed the balance slightly:

- `strong_safe v4`:
  - field accuracy: `0.3657`
  - precision: `0.5257`
  - recall: `0.3657`
  - f1: `0.4257`

This suggests the current branch is approaching a parser-only plateau: precision can still be improved with cleanup, but larger gains will likely require either a better driver-license OCR path or stronger verified labels.

The newer passport-heavy `strong_safe_v3data` run changed the training picture again:

- train loss: `1.4997`
- val loss: `0.9176`

Benchmark metrics:

- `heuristic_pipeline`:
  - field accuracy: `0.3000`
  - precision: `0.4095`
  - recall: `0.3000`
  - f1: `0.3442`
- `fine_tuned_lora`:
  - field accuracy: `0.3486`
  - precision: `0.4129`
  - recall: `0.3486`
  - f1: `0.3762`

Interpretation:

- this is a useful mixed result;
- adding diverse passport examples clearly helps optimization and lowers validation loss;
- however, on the current `5`-sample identity benchmark the resulting checkpoint is still below the best `strong_safe v3` benchmark point;
- this means the new passport supervision is valuable, but the remaining bottleneck is still the driver-license side of the identity benchmark.

## Driver-license label cleanup

After manual visual review of the California synthetic cards, the driver-license supervision was cleaned up:

- `synthetic-usa-driver-license_002` was corrected from the bottom-right DOB-like numeric block to the visually plausible red DL number `18951467`;
- `synthetic-usa-driver-license_003` was corrected in the same way as a visual duplicate of `002`;
- `categories` for the clearly visible `CLASS C` cards were normalized to `["C"]`;
- the low-quality `synthetic-usa-driver-license_004` record was removed from the current train and pilot splits, because its visible text coverage is too weak for trustworthy supervision.

The annotation queue was also aligned with the new reality:

- `synthetic-usa-driver-license_000`
- `synthetic-usa-driver-license_001`
- `synthetic-usa-driver-license_002`
- `synthetic-usa-driver-license_003`

are now treated as `verified_reference` identity records rather than pending weak drafts.

## Clean-label benchmark re-score

Using the already saved predictions from the strongest low-memory benchmark point, but rescoring them against the cleaned `driver_license` labels, we get:

- artifact: [E:\thesis\multidoc_benchmark_identity_strong_safe_v3_cleanlabels.json](/E:/thesis/multidoc_benchmark_identity_strong_safe_v3_cleanlabels.json)

Metrics:

- `heuristic_pipeline`:
  - field accuracy: `0.3571`
  - precision: `0.5286`
  - recall: `0.3571`
  - f1: `0.4242`
- `fine_tuned_lora`:
  - field accuracy: `0.3807`
  - precision: `0.5157`
  - recall: `0.3807`
  - f1: `0.4364`

Interpretation:

- part of the earlier driver-license instability was indeed caused by supervision noise;
- after cleaning labels, the identity benchmark remains challenging, but it becomes more internally consistent;
- the best current low-memory checkpoint is still the `strong_safe v3` family, now with cleaner evaluation semantics.

## New clean-label strong-safe retrain

After the driver-license label cleanup, a new one-epoch low-memory retrain was run on the cleaned `14`-sample strong split:

- checkpoint: [E:\thesis\outputs\paddleocr_vl_multidoc_lora_strong_safe_v4clean](/E:/thesis/outputs/paddleocr_vl_multidoc_lora_strong_safe_v4clean)
- training summary: [training_summary.json](/E:/thesis/outputs/paddleocr_vl_multidoc_lora_strong_safe_v4clean/training_summary.json)

Training result:

- train loss: `1.4984`
- val loss: `0.9117`

Benchmark:

- [E:\thesis\multidoc_benchmark_identity_strong_safe_v4clean.json](/E:/thesis/multidoc_benchmark_identity_strong_safe_v4clean.json)

Metrics:

- `heuristic_pipeline`:
  - field accuracy: `0.2821`
  - precision: `0.4095`
  - recall: `0.2821`
  - f1: `0.3332`
- `fine_tuned_lora`:
  - field accuracy: `0.3307`
  - precision: `0.4129`
  - recall: `0.3307`
  - f1: `0.3667`

Interpretation:

- the retrained checkpoint still beats the corresponding heuristic branch on this run;
- however, it does **not** beat the rescored `strong_safe_v3_cleanlabels` point;
- this means the current limitation is no longer just label noise, but also model/runtime sensitivity on the small identity set.

## California-specific parser cleanup

After the clean-label retrain, the `driver_license` parser was further tightened for the California-style synthetic layout:

- `CLASS C` parsing no longer leaks the `A` from `California`;
- compact `MMDDYYYY` date strings are normalized more safely when OCR drops slashes;
- the license-number extraction keeps stronger preference for number candidates close to `DL / EXP / CLASS`.

This was evaluated without retraining again, using the same `v4clean` adapter but a refreshed benchmark:

- [E:\thesis\multidoc_benchmark_identity_strong_safe_v4clean_r2.json](/E:/thesis/multidoc_benchmark_identity_strong_safe_v4clean_r2.json)

Metrics:

- `heuristic_pipeline`:
  - field accuracy: `0.3071`
  - precision: `0.5095`
  - recall: `0.3071`
  - f1: `0.3732`
- `fine_tuned_lora`:
  - field accuracy: `0.3307`
  - precision: `0.4129`
  - recall: `0.3307`
  - f1: `0.3667`

Interpretation:

- the heuristic branch improved meaningfully on the same checkpoint family;
- the fine-tuned branch did not move, which suggests the current bottleneck is no longer category parsing, but OCR coverage and number recovery on the `driver_license` samples themselves;
- the best clean-label benchmark point is still `strong_safe_v3_cleanlabels`.

## Manual visual-duplicate expansion

Two additional driver-license records were added to the training subset after manual visual review:

- `synthetic-usa-driver-license_001` -> visual duplicate of `synthetic-usa-driver-license_000`
- `synthetic-usa-driver-license_003` -> visual duplicate of `synthetic-usa-driver-license_002`

These were added as manually curated visual duplicates to strengthen the identity subset without introducing the weak `synthetic-usa-driver-license_004` label into the strong split.

## New low-memory strong run on expanded curated subset

Checkpoint:

- [E:\thesis\outputs\paddleocr_vl_multidoc_lora_strong_safe_v2](/E:/thesis/outputs/paddleocr_vl_multidoc_lora_strong_safe_v2)

One-epoch training result:

- train loss: `1.5014`
- val loss: `1.0802`

Benchmark:

- [E:\thesis\multidoc_benchmark_identity_strong_safe_v2data.json](/E:/thesis/multidoc_benchmark_identity_strong_safe_v2data.json)

Metrics:

- `heuristic_pipeline`:
  - field accuracy: `0.3000`
  - precision: `0.4095`
  - recall: `0.3000`
  - f1: `0.3442`
- `fine_tuned_lora`:
  - field accuracy: `0.3486`
  - precision: `0.4129`
  - recall: `0.3486`
  - f1: `0.3762`

Interpretation:

- expanding the curated strong subset with near-duplicate driver-license examples did **not** improve the benchmark over the best `strong_safe v3` run;
- this is a useful negative result: more samples help only when they also increase informational diversity, not just count.

## What the latest benchmark shows

### Passport-only mini benchmark (`3` samples)

File:

- [E:\thesis\multidoc_benchmark_passport_v5.json](/E:/thesis/multidoc_benchmark_passport_v5.json)

Metrics:

- `out_of_the_box`: `0.00 / 0.00 / 0.00 / 0.00`
- `heuristic_pipeline`:
  - field accuracy: `0.9190`
  - precision: `0.8357`
  - recall: `0.9190`
  - f1: `0.8714`
- `fine_tuned_lora`: still unstable, but no longer purely zero-signal

### Identity benchmark (`5` samples)

File:

- [E:\thesis\multidoc_benchmark_identity_v5.json](/E:/thesis/multidoc_benchmark_identity_v5.json)

Metrics:

- `out_of_the_box`: `0.00 / 0.00 / 0.00 / 0.00`
- `heuristic_pipeline`:
  - field accuracy: `0.4305`
  - precision: `0.4829`
  - recall: `0.4305`
  - f1: `0.4504`
- `fine_tuned_lora`:
  - field accuracy: `0.1876`
  - precision: `0.3000`
  - recall: `0.1876`
  - f1: `0.2247`

## Honest interpretation

- The heuristic pipeline is currently the strongest practical identity extractor.
- The `LoRA` branch is improving in validation loss and now shows a stronger non-zero extraction signal on the identity mini-benchmark, but it still does not reliably produce clean final JSON on identity documents.
- The `LoRA` model is no longer completely hopeless:
  - on some samples it already starts generating JSON-like structured fragments
  - but the fields are still noisy or schema-inconsistent
- The runtime pipeline now includes a quality gate so obviously repetitive or low-quality structured adapter outputs fall back to the heuristic OCR branch instead of being accepted blindly.
- The latest benchmark also corrected an evaluation issue where the heuristic path could accidentally reuse adapter-influenced behavior; the current heuristic score is lower than older internal notes, but it is much more trustworthy.
- This means the next bottleneck is no longer "can we train at all?", but:
  - more verified identity data
  - stronger decoding constraints
  - better postprocessing and JSON recovery

## Next technical focus

1. Add a few more strong passport and driver-license records into verified training data.
2. Improve fine-tuned decoding and JSON recovery.
3. Re-run the identity benchmark after the next LoRA iteration.

## Driver-license OCR specialization update

After the clean-label retrain plateau, the next iteration targeted the runtime rather than another blind training loop.

The `driver_license` logic in [E:\thesis\backend\app\services\extraction.py](/E:/thesis/backend/app/services/extraction.py) was extended with:

- a broader top-header OCR crop for `DL / EXP / CLASS / LN / FN`
- a second crop focused on dates and detail fields
- stronger rejection of date-like numeric values when selecting `license_number`
- stricter category extraction to avoid false positives like `B` from `BRN`

Benchmark artifact:

- [E:\thesis\multidoc_benchmark_identity_strong_safe_v4clean_r4.json](/E:/thesis/multidoc_benchmark_identity_strong_safe_v4clean_r4.json)

Metrics:

- `heuristic_pipeline`
  - field accuracy: `0.5112`
  - precision: `0.5445`
  - recall: `0.5112`
  - f1: `0.5223`
- `fine_tuned_lora`
  - field accuracy: `0.3112`
  - precision: `0.4095`
  - recall: `0.3112`
  - f1: `0.3480`

Interpretation:

- this is the strongest heuristic identity result so far on the current evaluation set;
- the biggest practical gain came from better driver-license OCR recovery rather than from a new training pass;
- the `LoRA` branch remains useful as the research branch, but the best production-like path for identity documents is currently the stronger heuristic OCR pipeline.

## Structured-hybrid LoRA recovery

The next improvement targeted the fine-tuned branch itself instead of the OCR crops alone.

A hybrid recovery layer was added:

- structured `LoRA` output remains the primary source;
- missing or suspicious identity fields are backfilled from the stronger heuristic OCR branch;
- the same merge logic is now used both in runtime extraction and in the benchmark script, so evaluation matches actual application behavior.

Benchmark artifact:

- [E:\thesis\multidoc_benchmark_identity_strong_safe_v4clean_r5.json](/E:/thesis/multidoc_benchmark_identity_strong_safe_v4clean_r5.json)

Metrics:

- `heuristic_pipeline`
  - field accuracy: `0.5112`
  - precision: `0.5445`
  - recall: `0.5112`
  - f1: `0.5223`
- `fine_tuned_lora`
  - field accuracy: `0.4462`
  - precision: `0.4795`
  - recall: `0.4462`
  - f1: `0.4573`

Interpretation:

- this is the strongest identity result so far for the fine-tuned branch;
- the gap to the practical heuristic branch became much smaller;
- the heuristic pipeline is still stronger overall, but the `LoRA` branch is now substantially more defensible as a real thesis result rather than a weak side experiment.

## Prompt-constrained structured decoding

The next decoding improvement tightened the structured prompt itself.

Both runtime extraction and the benchmark runner now pass:

- an explicit JSON skeleton,
- `null` defaults for missing scalar fields,
- `[]` defaults for list fields,
- and a stricter instruction to return exactly one JSON object with no markdown or explanation text.

Benchmark artifact:

- [E:\thesis\multidoc_benchmark_identity_strong_safe_v4clean_r6.json](/E:/thesis/multidoc_benchmark_identity_strong_safe_v4clean_r6.json)

Metrics:

- `heuristic_pipeline`
  - field accuracy: `0.5112`
  - precision: `0.5445`
  - recall: `0.5112`
  - f1: `0.5223`
- `fine_tuned_lora`
  - field accuracy: `0.4862`
  - precision: `0.5195`
  - recall: `0.4862`
  - f1: `0.4973`

Interpretation:

- this is the strongest result so far for the fine-tuned identity branch;
- the heuristic branch stayed unchanged, which is expected;
- the stricter schema prompt improved the learned branch itself rather than the OCR fallback path;
- at this point the `LoRA` branch is close enough to the heuristic branch to be a credible learned extractor in the thesis narrative.

## Adding verified driver-license_004

The weak `synthetic-usa-driver-license_004` draft was manually reviewed and upgraded into a curated duplicate with verified labels:

- `license_number = 18951467`
- `surname = DAVIS`
- `given_names = MICHAEL`
- `date_of_birth = 05/19/1992`
- `address = 3524 BREMEN DRIVE`
- `date_of_issue = 01/16/2014`
- `date_of_expiry = 04/17/2018`
- `categories = ["C"]`

This increased the strong training split to `15` records.

New run:

- [E:\thesis\outputs\paddleocr_vl_multidoc_lora_strong_safe_v7dl](/E:/thesis/outputs/paddleocr_vl_multidoc_lora_strong_safe_v7dl)
- best val loss: `1.0929`

Benchmark artifact:

- [E:\thesis\multidoc_benchmark_identity_strong_safe_v7dl.json](/E:/thesis/multidoc_benchmark_identity_strong_safe_v7dl.json)

Metrics:

- `heuristic_pipeline`
  - field accuracy: `0.5112`
  - precision: `0.5445`
  - recall: `0.5112`
  - f1: `0.5223`
- `fine_tuned_lora`
  - field accuracy: `0.4862`
  - precision: `0.5195`
  - recall: `0.4862`
  - f1: `0.4973`

Interpretation:

- the extra verified duplicate did not improve the benchmark over the previous `r6` result;
- this is a useful negative finding: once the label noise is cleaned up, additional near-duplicate driver-license samples contribute little;
- the next data-side improvement should come from more diverse verified layouts, not from more copies of the same California template.

## Diverse external passport expansion

The external identity pool was then expanded beyond the earlier UAE/US-heavy mix. Additional synthetic passport layouts were imported from:

- `ud-synthetic/synthetic-printed-usa-passports`
- `ud-synthetic/synthetic-printed-german-passports`
- `ud-synthetic/synthetic-french-passports`
- `ud-synthetic/synthetic-japanese-passports`
- `ud-synthetic/synthetic-russian-passports`

Queue status after the import expansion:

- total queue rows: `51`
- heuristic drafts: `13`
- manual external rows still waiting for review: `22`

Practical weak-label yield by source:

- `ud-biometrics/passport-dataset` -> `6 / 10`
- `ud-biometrics/synthetic-usa-driver-license` -> `5 / 10`
- `ud-synthetic/synthetic-printed-usa-passports` -> `1 / 5`
- `ud-synthetic/synthetic-printed-german-passports` -> `0 / 5`
- `ud-synthetic/synthetic-french-passports` -> `2 / 5`
- `ud-synthetic/synthetic-japanese-passports` -> `1 / 5`
- `ud-synthetic/synthetic-russian-passports` -> `3 / 5`

Interpretation:

- Russian and French passports are currently the strongest new diversity sources.
- German passports are not currently usable as automatic weak-label sources.
- The broader import is still useful, because it turns the queue into a much stronger raw pool for manual curation.

## Curation plan

A dedicated curation plan was generated:

- [E:\thesis\IDENTITY_CURATION_PLAN.md](/E:/thesis/IDENTITY_CURATION_PLAN.md)
- [E:\thesis\data\multidoc\identity_curated_train_candidates.jsonl](/E:/thesis/data/multidoc/identity_curated_train_candidates.jsonl)
- [E:\thesis\data\multidoc\identity_holdout_eval_candidates.jsonl](/E:/thesis/data/multidoc/identity_holdout_eval_candidates.jsonl)
- [E:\thesis\data\multidoc\identity_manual_review_candidates.jsonl](/E:/thesis/data/multidoc/identity_manual_review_candidates.jsonl)

This separated the new raw pool into:

- candidates for future curated train promotion,
- candidates for hold-out diverse evaluation,
- and rows that should remain manual-review-only for now.

## New curated passport promotions

Three new diverse passports were manually curated and promoted into the strong training subset:

- `ud-synthetic__synthetic-russian-passports_000`
- `ud-synthetic__synthetic-french-passports_000`
- `ud-synthetic__synthetic-russian-passports_001`

This increased the current train split to `18` rows:

- `9` passport
- `5` driver_license
- `4` invoice

## Diverse low-memory runs

The first diversity-oriented low-memory run used the expanded strong split with two new curated passports:

- [E:\thesis\outputs\paddleocr_vl_multidoc_lora_strong_safe_v8diverse](/E:/thesis/outputs/paddleocr_vl_multidoc_lora_strong_safe_v8diverse)
- best val loss: `0.8877`

This is the best validation loss in the `strong_safe` family so far.

To evaluate whether the diversity actually helps beyond optimization loss, a new hold-out eval was created:

- [E:\thesis\data\multidoc\identity_eval_diverse_v1.jsonl](/E:/thesis/data/multidoc/identity_eval_diverse_v1.jsonl)

This extends the original identity mini-set with two unseen diverse passports:

- `ud-synthetic__synthetic-russian-passports_001_eval`
- `ud-synthetic__synthetic-french-passports_002_eval`

Benchmark results for the older learned checkpoint:

- artifact: [E:\thesis\multidoc_benchmark_identity_diverse_v1_v4clean.json](/E:/thesis/multidoc_benchmark_identity_diverse_v1_v4clean.json)
- `heuristic_pipeline`
  - field accuracy: `0.3781`
  - precision: `0.4175`
  - recall: `0.3781`
  - f1: `0.3909`
- `fine_tuned_lora (v4clean)`
  - field accuracy: `0.3603`
  - precision: `0.3997`
  - recall: `0.3603`
  - f1: `0.3731`

After adding one more curated Russian passport, another low-memory run was executed:

- [E:\thesis\outputs\paddleocr_vl_multidoc_lora_strong_safe_v9curated](/E:/thesis/outputs/paddleocr_vl_multidoc_lora_strong_safe_v9curated)
- best val loss: `0.9133`

Benchmark artifact:

- [E:\thesis\multidoc_benchmark_identity_diverse_v1_v9curated.json](/E:/thesis/multidoc_benchmark_identity_diverse_v1_v9curated.json)

Metrics:

- `heuristic_pipeline`
  - field accuracy: `0.3781`
  - precision: `0.4175`
  - recall: `0.3781`
  - f1: `0.3909`
- `fine_tuned_lora (v9curated)`
  - field accuracy: `0.3460`
  - precision: `0.3854`
  - recall: `0.3460`
  - f1: `0.3588`

Interpretation:

- adding diverse curated passports clearly improves dataset quality and often improves validation loss;
- however, under the current `1`-epoch low-memory regime, these gains still do not reliably transfer into better hold-out diverse extraction quality;
- this is an important signal that the next real improvement will likely require either:
  - a stronger training curriculum than `1` epoch,
  - a document-type-specific fine-tuning schedule,
  - or a more constrained field-grounded decoding path.

## Passport-first curriculum

To test whether a more focused training regime helps on the same local machine, a separate passport-only curriculum split was created:

- [E:\thesis\data\multidoc\passport_curriculum_train_v1.jsonl](/E:/thesis/data/multidoc/passport_curriculum_train_v1.jsonl) -> `8` passport train rows
- [E:\thesis\data\multidoc\passport_curriculum_val_v1.jsonl](/E:/thesis/data/multidoc/passport_curriculum_val_v1.jsonl) -> `1` passport validation row
- [E:\thesis\data\multidoc\passport_eval_diverse_v1.jsonl](/E:/thesis/data/multidoc/passport_eval_diverse_v1.jsonl) -> `5` passport hold-out rows

Then a `2`-epoch low-memory passport-only run was executed:

- [E:\thesis\outputs\paddleocr_vl_passport_curriculum_v1](/E:/thesis/outputs/paddleocr_vl_passport_curriculum_v1)

Training summary:

- epoch 1 train loss: `1.9214`
- epoch 1 val loss: `1.2978`
- epoch 2 train loss: `1.4599`
- epoch 2 val loss: `1.1676`

Benchmark artifact:

- [E:\thesis\multidoc_benchmark_passport_curriculum_v1.json](/E:/thesis/multidoc_benchmark_passport_curriculum_v1.json)

Passport-only diverse eval metrics:

- `heuristic_pipeline`
  - field accuracy: `0.3544`
  - precision: `0.4095`
  - recall: `0.3544`
  - f1: `0.3723`
- `fine_tuned_lora (passport_curriculum_v1)`
  - field accuracy: `0.4210`
  - precision: `0.4095`
  - recall: `0.4210`
  - f1: `0.4079`

Interpretation:

- this is the clearest sign so far that a document-family-specific curriculum helps;
- the learned passport branch now beats the heuristic branch on the passport-only diverse eval;
- therefore, on the current hardware, a narrower curriculum appears more promising than continuing to force all identity layouts through one tiny mixed low-memory run.

## Driver-license-first curriculum

The same idea was tested separately for `driver_license`.

Dedicated splits were created:

- [E:\thesis\data\multidoc\driver_license_curriculum_train_v1.jsonl](/E:/thesis/data/multidoc/driver_license_curriculum_train_v1.jsonl) -> `4` train rows
- [E:\thesis\data\multidoc\driver_license_curriculum_val_v1.jsonl](/E:/thesis/data/multidoc/driver_license_curriculum_val_v1.jsonl) -> `1` validation row
- [E:\thesis\data\multidoc\driver_license_eval_diverse_v1.jsonl](/E:/thesis/data/multidoc/driver_license_eval_diverse_v1.jsonl) -> `2` eval rows

Then a `3`-epoch low-memory run was executed:

- [E:\thesis\outputs\paddleocr_vl_driver_license_curriculum_v1](/E:/thesis/outputs/paddleocr_vl_driver_license_curriculum_v1)

Training dynamics:

- epoch 1 train loss: `1.6918`
- epoch 1 val loss: `1.4701`
- epoch 2 train loss: `1.2470`
- epoch 2 val loss: `1.1197`
- epoch 3 train loss: `1.0102`
- epoch 3 val loss: `1.0342`

Benchmark artifact:

- [E:\thesis\multidoc_benchmark_driver_license_curriculum_v1.json](/E:/thesis/multidoc_benchmark_driver_license_curriculum_v1.json)

Driver-license eval metrics:

- `heuristic_pipeline`
  - field accuracy: `0.4375`
  - precision: `0.4375`
  - recall: `0.4375`
  - f1: `0.4375`
- `fine_tuned_lora (driver_license_curriculum_v1)`
  - field accuracy: `0.3750`
  - precision: `0.3750`
  - recall: `0.3750`
  - f1: `0.3750`

Interpretation:

- the driver-license branch does learn under a focused curriculum;
- however, unlike passports, the current learned branch still trails the heuristic OCR pipeline on actual extraction quality;
- this suggests that driver licenses are now limited more by OCR/field grounding quality than by the lack of a narrow fine-tuning schedule.

## Driver-license crop inspection and geometry fix

To understand why the driver-license branch lagged, lightweight crop exports were generated for the California template:

- [E:\thesis\debug\driver_license_000_crops](/E:/thesis/debug/driver_license_000_crops)
- [E:\thesis\debug\driver_license_000_crops_v2](/E:/thesis/debug/driver_license_000_crops_v2)

This inspection showed that the old `number` and `issue/bottom_right` OCR crops were partially pointed at empty background. The extraction service was updated so that driver-license-specific crops are now anchored to a tighter internal focus region rather than the full page crop.

The same pass also tightened date normalization so that OCR years like `1692` are repaired to `1992` when supported by nearby compact numeric evidence.

Resulting driver-license benchmark:

- [E:\thesis\multidoc_benchmark_driver_license_curriculum_v1_r4.json](/E:/thesis/multidoc_benchmark_driver_license_curriculum_v1_r4.json)

Metrics:

- heuristic_pipeline
  - field accuracy: `0.6250`
  - precision: `0.7143`
  - recall: `0.6250`
  - f1: `0.6667`
- fine_tuned_lora
  - field accuracy: `0.6250`
  - precision: `0.7143`
  - recall: `0.6250`
  - f1: `0.6667`

Interpretation:

- the driver-license bottleneck was significantly caused by crop geometry;
- after re-anchoring the OCR regions, the heuristic and learned driver-license branches reached parity on the small focused eval;
- this makes the `driver_license-first curriculum` result much more meaningful than it looked before the geometry fix.

## Updated mixed identity checkpoint status

After the driver-license grounding refresh, the mixed identity benchmark for the current strong-safe learned branch was rerun:

- [E:\thesis\multidoc_benchmark_identity_strong_safe_v4clean_r7.json](/E:/thesis/multidoc_benchmark_identity_strong_safe_v4clean_r7.json)

Metrics:

- heuristic_pipeline
  - field accuracy: `0.4071`
  - precision: `0.5286`
  - recall: `0.4071`
  - f1: `0.4570`
- fine_tuned_lora
  - field accuracy: `0.4357`
  - precision: `0.5486`
  - recall: `0.4357`
  - f1: `0.4843`

Interpretation:

- the latest mixed identity learned branch is now again ahead of the heuristic baseline on the current `identity_eval_v5`;
- the main unresolved errors are concentrated in harder passport samples and the second synthetic driver-license example rather than in the core California crop geometry.
