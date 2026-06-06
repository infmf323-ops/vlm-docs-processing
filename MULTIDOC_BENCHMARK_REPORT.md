# Multi-Document Benchmark Report

Date: 2026-05-17

## Goal

The current benchmark compares three modes for the new multi-document extraction branch:

- `out_of_the_box`
- `heuristic_pipeline`
- `fine_tuned_lora`

The evaluation is still focused on a small pilot subset, because the identity-document training set is only now becoming large enough for meaningful LoRA experiments.

## Diversity expansion note

The latest iteration did not focus on another blind retrain first. Instead, the external identity pool was widened with multiple new passport layouts:

- USA
- Germany
- France
- Japan
- Russia

This matters because the current learned branch no longer benefits much from extra near-duplicates of the same California driver-license template. The current bottleneck is now identity-layout diversity rather than raw row count.

Practical weak-label outcome of the new import round:

| Source dataset | Valid heuristic drafts |
|---|---:|
| `ud-biometrics/passport-dataset` | `6 / 10` |
| `ud-biometrics/synthetic-usa-driver-license` | `5 / 10` |
| `ud-synthetic/synthetic-printed-usa-passports` | `1 / 3` |
| `ud-synthetic/synthetic-printed-german-passports` | `0 / 3` |
| `ud-synthetic/synthetic-french-passports` | `2 / 3` |
| `ud-synthetic/synthetic-japanese-passports` | `1 / 3` |
| `ud-synthetic/synthetic-russian-passports` | `3 / 3` |

Interpretation:

- Russian and French synthetic passports are currently the most promising new diversity sources;
- German passports currently do not work well as heuristic weak-label sources;
- the imported layouts are still valuable even when weak labels are noisy, because they increase the pool for manual review and curated promotion.

Two of the new diverse passport samples were manually curated and promoted into the strong training subset:

- `ud-synthetic__synthetic-russian-passports_000`
- `ud-synthetic__synthetic-french-passports_000`

This created a new diversity-oriented low-memory training run:

- checkpoint: [E:\thesis\outputs\paddleocr_vl_multidoc_lora_strong_safe_v8diverse](/E:/thesis/outputs/paddleocr_vl_multidoc_lora_strong_safe_v8diverse)
- validation loss: `0.8877`

That is the strongest validation-loss result of the `strong_safe` branch so far. However, the benchmark story is more nuanced.

## Hold-out diverse eval

To avoid reusing only the original identity mini-set, a new hold-out diversity probe was created:

- [E:\thesis\data\multidoc\identity_eval_diverse_v1.jsonl](/E:/thesis/data/multidoc/identity_eval_diverse_v1.jsonl)

This extends the original `5`-sample identity eval with two additional unseen passport layouts:

- `ud-synthetic__synthetic-russian-passports_001_eval`
- `ud-synthetic__synthetic-french-passports_002_eval`

Benchmarks:

- older learned checkpoint:
  - [E:\thesis\multidoc_benchmark_identity_diverse_v1_v4clean.json](/E:/thesis/multidoc_benchmark_identity_diverse_v1_v4clean.json)
- newer diversity-trained checkpoint:
  - [E:\thesis\multidoc_benchmark_identity_diverse_v1_v8diverse.json](/E:/thesis/multidoc_benchmark_identity_diverse_v1_v8diverse.json)

Metrics on the `7`-sample diverse eval:

| Mode | Field Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| heuristic_pipeline | 0.3781 | 0.4175 | 0.3781 | 0.3909 |
| fine_tuned_lora (`v4clean`) | 0.3603 | 0.3997 | 0.3603 | 0.3731 |
| fine_tuned_lora (`v8diverse`) | 0.3460 | 0.3854 | 0.3460 | 0.3588 |

Interpretation:

- adding more diverse passports clearly helped optimization and lowered validation loss;
- but under the current `1`-epoch low-memory regime, the newer diversity-trained checkpoint still did not beat the older learned checkpoint on the hold-out diverse eval;
- this is an important negative result:
  - extra data alone is not enough,
  - and the current bottleneck is now the combination of tiny supervised scale, low-memory fine-tuning, and fragile structured decoding.

## Further curated-passport follow-up

One more Russian passport was then manually curated and promoted into the strong training subset:

- `ud-synthetic__synthetic-russian-passports_001`

This produced another low-memory run:

- checkpoint: [E:\thesis\outputs\paddleocr_vl_multidoc_lora_strong_safe_v9curated](/E:/thesis/outputs/paddleocr_vl_multidoc_lora_strong_safe_v9curated)
- validation loss: `0.9133`

And a new hold-out diverse benchmark:

- [E:\thesis\multidoc_benchmark_identity_diverse_v1_v9curated.json](/E:/thesis/multidoc_benchmark_identity_diverse_v1_v9curated.json)

Metrics on the same `7`-sample diverse eval:

| Mode | Field Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| heuristic_pipeline | 0.3781 | 0.4175 | 0.3781 | 0.3909 |
| fine_tuned_lora (`v9curated`) | 0.3460 | 0.3854 | 0.3460 | 0.3588 |

Interpretation:

- this is effectively the same practical result as `v8diverse`;
- adding one more strong passport is still worth keeping in the dataset, but it does not materially move the benchmark under the current training regime;
- the strongest current conclusion is now clearer than before:
  - data diversity is necessary,
  - but tiny one-epoch low-memory LoRA is no longer enough to fully capitalize on it.

## Passport-first curriculum benchmark

To test a stronger local strategy without changing hardware, the next step was not another mixed identity run, but a narrower passport-only curriculum:

- train split: [E:\thesis\data\multidoc\passport_curriculum_train_v1.jsonl](/E:/thesis/data/multidoc/passport_curriculum_train_v1.jsonl)
- val split: [E:\thesis\data\multidoc\passport_curriculum_val_v1.jsonl](/E:/thesis/data/multidoc/passport_curriculum_val_v1.jsonl)
- hold-out eval: [E:\thesis\data\multidoc\passport_eval_diverse_v1.jsonl](/E:/thesis/data/multidoc/passport_eval_diverse_v1.jsonl)

This run used:

- `2` epochs
- the same low-memory image and token limits
- only passport documents

Checkpoint:

- [E:\thesis\outputs\paddleocr_vl_passport_curriculum_v1](/E:/thesis/outputs/paddleocr_vl_passport_curriculum_v1)

Benchmark:

- [E:\thesis\multidoc_benchmark_passport_curriculum_v1.json](/E:/thesis/multidoc_benchmark_passport_curriculum_v1.json)

Metrics on the `5`-sample passport-only diverse eval:

| Mode | Field Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| heuristic_pipeline | 0.3544 | 0.4095 | 0.3544 | 0.3723 |
| fine_tuned_lora (`passport_curriculum_v1`) | 0.4210 | 0.4095 | 0.4210 | 0.4079 |

Interpretation:

- this is the clearest positive learning result in the latest local cycle;
- a document-family-specific curriculum works better than another tiny mixed identity run on the same machine;
- therefore, on the current hardware, the next sensible direction is:
  - continue passport-specialized improvement separately,
  - and only later return to broader mixed identity fine-tuning.

## Driver-license-first curriculum benchmark

To test whether the same curriculum trick also helps the weaker license branch, a separate driver-license-only run was created:

- train split: [E:\thesis\data\multidoc\driver_license_curriculum_train_v1.jsonl](/E:/thesis/data/multidoc/driver_license_curriculum_train_v1.jsonl)
- val split: [E:\thesis\data\multidoc\driver_license_curriculum_val_v1.jsonl](/E:/thesis/data/multidoc/driver_license_curriculum_val_v1.jsonl)
- eval split: [E:\thesis\data\multidoc\driver_license_eval_diverse_v1.jsonl](/E:/thesis/data/multidoc/driver_license_eval_diverse_v1.jsonl)

This branch used:

- `3` epochs
- the same low-memory image/token settings
- only `driver_license` samples

Checkpoint:

- [E:\thesis\outputs\paddleocr_vl_driver_license_curriculum_v1](/E:/thesis/outputs/paddleocr_vl_driver_license_curriculum_v1)

Benchmark:

- [E:\thesis\multidoc_benchmark_driver_license_curriculum_v1.json](/E:/thesis/multidoc_benchmark_driver_license_curriculum_v1.json)

Metrics on the `2`-sample license eval:

| Mode | Field Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| heuristic_pipeline | 0.4375 | 0.4375 | 0.4375 | 0.4375 |
| fine_tuned_lora (`driver_license_curriculum_v1`) | 0.3750 | 0.3750 | 0.3750 | 0.3750 |

Interpretation:

- unlike the passport branch, the driver-license branch does not yet overtake the heuristic baseline;
- the training losses fall cleanly across `3` epochs, so the branch is learning;
- however, the practical bottleneck for licenses still appears to be OCR/grounding quality rather than the absence of a narrower curriculum.

This gives a useful split conclusion:

- `passport-first curriculum` is the strongest local learning direction so far;
- `driver_license-first curriculum` is still weaker than heuristic OCR, so its next gains likely need better region grounding or stronger OCR-targeted prompts rather than another small isolated LoRA run.

## Current benchmark artifacts

### Passport mini benchmark

- [E:\thesis\multidoc_benchmark_passport_v5.json](/E:/thesis/multidoc_benchmark_passport_v5.json)

Dataset:

- `passport_michelle_obama`
- `passport-dataset_000`
- `passport-dataset_002`

Metrics:

| Mode | Field Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| out_of_the_box | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| heuristic_pipeline | 0.9190 | 0.8357 | 0.9190 | 0.8714 |
| fine_tuned_lora (v5) | low and unstable | low and unstable | low and unstable | low and unstable |

### Identity mini benchmark

- [E:\thesis\multidoc_benchmark_identity_v5.json](/E:/thesis/multidoc_benchmark_identity_v5.json)

Dataset:

- `passport_michelle_obama`
- `passport-dataset_000`
- `passport-dataset_002`
- `synthetic-usa-driver-license_000`
- `synthetic-usa-driver-license_002`

Metrics:

| Mode | Field Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| out_of_the_box | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| heuristic_pipeline | 0.4305 | 0.4829 | 0.4305 | 0.4504 |
| fine_tuned_lora (v5) | 0.1876 | 0.3000 | 0.1876 | 0.2247 |

## Interpretation

### Out of the box

The zero-shot `PaddleOCR-VL` branch still fails as a structured extractor for these identity cases. It may produce OCR-like text or fragmented pseudo-JSON, but it does not yet yield usable normalized fields.

### Heuristic pipeline

The heuristic branch is still the strongest practical mode for the identity benchmark, but the score is now lower than earlier internal runs because the benchmark was recalibrated to use the true base `PaddleOCR-VL` OCR path instead of an unintentionally adapter-influenced path. This makes the current numbers stricter, but also much more trustworthy.

### Fine-tuned LoRA

The LoRA branch has improved strongly in validation loss, and after adding structured-text salvage it now shows clearer non-zero extraction metrics on the identity mini-benchmark. This still does **not** mean the model is ready. Instead, it means the model is currently stuck in an intermediate phase:

- it emits JSON-like output more often than before,
- but the output is still incomplete, repetitive, or schema-inconsistent,
- so final structured extraction remains unreliable.

## Important nuance

The latest `v5` adapter is already better than earlier runs in two concrete senses:

- it begins to emit partially structured fields on some identity samples,
- which did not happen consistently in earlier iterations;
- it now scores above zero on the identity mini-benchmark.

This is a sign of progress, even though the model still trails the heuristic branch by a large margin.

## Benchmark recalibration note

The latest benchmark also fixed an important evaluation bug:

- the base OCR heuristic path and the LoRA-adapted structured path are now separated more cleanly;
- empty adapter configuration no longer resolves to `Path('.')`;
- the benchmark no longer overestimates the heuristic branch by accidentally reusing the adapter path.

This is why the latest heuristic metrics are lower than some earlier notes: the new numbers are more honest.

## Controlled training note

To avoid further crashes on a 16 GB RAM machine, two lighter one-epoch comparison runs were added:

- [E:\thesis\outputs\paddleocr_vl_multidoc_lora_strong_safe](/E:/thesis/outputs/paddleocr_vl_multidoc_lora_strong_safe)
  - `best val loss = 1.0217`
- [E:\thesis\outputs\paddleocr_vl_multidoc_lora_augmented_safe](/E:/thesis/outputs/paddleocr_vl_multidoc_lora_augmented_safe)
  - `best val loss = 1.0171`

These runs use a safer configuration:

- `MAX_IMAGE_SIDE = 640`
- `MAX_LENGTH = 1536`
- `MAX_NEW_TOKENS = 192`
- `NUM_EPOCHS = 1`

The training-loss result alone is not enough, so both lighter checkpoints were also benchmarked on the identity mini-set:

- [E:\thesis\multidoc_benchmark_identity_strong_safe.json](/E:/thesis/multidoc_benchmark_identity_strong_safe.json)
- [E:\thesis\multidoc_benchmark_identity_augmented_safe.json](/E:/thesis/multidoc_benchmark_identity_augmented_safe.json)

Low-memory benchmark comparison:

| Mode | Field Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| heuristic_pipeline | 0.2800 | 0.3371 | 0.2800 | 0.3036 |
| fine_tuned_lora (`strong_safe`) | 0.3171 | 0.3771 | 0.3171 | 0.3371 |
| fine_tuned_lora (`augmented_safe`) | 0.2886 | 0.3714 | 0.2886 | 0.3181 |

This gives a more actionable conclusion than raw `val loss` alone:

- in the low-memory setup, `strong_safe` is the best current fine-tuned identity checkpoint;
- the extra low-quality weak label slightly improved `val loss`, but did **not** improve practical extraction quality.

## Post-reboot parser refinement benchmark

After a Windows reboot and a stricter low-memory runtime configuration (`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`), the `strong_safe` branch was re-benchmarked again after targeted parser fixes for:

- passport name cleanup,
- compact passport MRZ parsing,
- driver-license date normalization,
- driver-license issue-date extraction,
- filtering false `license_number` values such as `CALIFORNIA`.

Artifact:

- [E:\thesis\multidoc_benchmark_identity_strong_safe_v3.json](/E:/thesis/multidoc_benchmark_identity_strong_safe_v3.json)

Updated low-memory benchmark:

| Mode | Field Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| heuristic_pipeline | 0.3571 | 0.4886 | 0.3571 | 0.4094 |
| fine_tuned_lora (`strong_safe`, v3 parser) | 0.4057 | 0.5157 | 0.4057 | 0.4524 |

This is the strongest current low-memory identity benchmark in the project. It also changes the practical interpretation slightly:

- `strong_safe` now beats the heuristic branch by a clearer margin on the identity mini-set;
- the gain comes from parser/runtime improvements, not from a new training run;
- the benchmark remains small, so this should be treated as a promising intermediate result rather than a final production-quality conclusion.

Sample-level observations after `v3`:

- `passport_michelle_obama` improved on the fine-tuned path:
  - correct `document_number`
  - clean `given_names`
  - restored `date_of_birth`
- `passport-dataset_002` stopped emitting the obviously wrong fallback token `OF` as a given name
- `driver_license` samples still have unresolved issues:
  - `license_number` may collapse to the DOB-like bottom-right numeric block
  - `date_of_birth` still sometimes becomes `1892` instead of `1992`
  - `date_of_issue` is still fragile on synthetic cards

## Additional cleanup pass (`v4`)

After `v3`, one more cleanup-only pass was applied without any new training:

- disabled weak `categories` extraction unless category-specific markers are present,
- normalized structured `categories` strings more safely,
- dropped obviously fake `issuing_authority` values that were just dates.

Artifact:

- [E:\thesis\multidoc_benchmark_identity_strong_safe_v4.json](/E:/thesis/multidoc_benchmark_identity_strong_safe_v4.json)

Metrics:

| Mode | Field Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| heuristic_pipeline | 0.3571 | 0.5429 | 0.3571 | 0.4281 |
| fine_tuned_lora (`strong_safe`, v4 parser) | 0.3657 | 0.5257 | 0.3657 | 0.4257 |

Interpretation:

- `heuristic_pipeline` became a bit cleaner and more precise;
- `fine_tuned_lora` stayed above the earlier `strong_safe` benchmark, but no longer beats the heuristic branch as clearly as in `v3`;
- this is a sign that the current branch is reaching a parser-only plateau: low-cost cleanup still helps, but large gains now likely require either:
  - a better driver-license OCR path,
  - or stronger verified identity labels.

## Expanded curated strong-split experiment

To test whether a slightly larger identity subset helps, two additional manually curated visual-duplicate driver-license records were added:

- `synthetic-usa-driver-license_001`
- `synthetic-usa-driver-license_003`

These produced an expanded `strong` split with `11` records and a new one-epoch low-memory run:

- checkpoint: [E:\thesis\outputs\paddleocr_vl_multidoc_lora_strong_safe_v2](/E:/thesis/outputs/paddleocr_vl_multidoc_lora_strong_safe_v2)
- benchmark: [E:\thesis\multidoc_benchmark_identity_strong_safe_v2data.json](/E:/thesis/multidoc_benchmark_identity_strong_safe_v2data.json)

Metrics:

| Mode | Field Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| heuristic_pipeline | 0.3000 | 0.4095 | 0.3000 | 0.3442 |
| fine_tuned_lora (`strong_safe_v2data`) | 0.3486 | 0.4129 | 0.3486 | 0.3762 |

This run is still better than the earliest safe baselines, but worse than the best `v3` parser/runtime benchmark. The key takeaway is important:

- adding more curated samples is **not enough by itself** if the new records are too visually similar to existing ones;
- the branch now appears to benefit more from higher information diversity than from simple count growth.

## Driver-license supervision cleanup

During a manual review of the California synthetic licenses, an important supervision issue was found:

- some records were labeled with the bottom-right DOB-like numeric block as `license_number`,
- even though the visually plausible red DL number on the card is `18951467`.

For the extraction task, the labels were aligned with the visible document content:

- `synthetic-usa-driver-license_002` -> `license_number = 18951467`
- `synthetic-usa-driver-license_003` -> corrected as a visual duplicate of `002`
- visible `CLASS C` markers were normalized to `categories = ["C"]`
- the weak `synthetic-usa-driver-license_004` sample was removed from the current train/pilot splits

This matters because the previous benchmark was partly penalizing the extractor for disagreeing with a noisy target.

### Re-scored benchmark on cleaned labels

To isolate the supervision effect, the already saved predictions from the strongest low-memory benchmark point were re-scored against the cleaned evaluation labels without re-running inference:

- artifact: [E:\thesis\multidoc_benchmark_identity_strong_safe_v3_cleanlabels.json](/E:/thesis/multidoc_benchmark_identity_strong_safe_v3_cleanlabels.json)

Metrics:

| Mode | Field Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| heuristic_pipeline | 0.3571 | 0.5286 | 0.3571 | 0.4242 |
| fine_tuned_lora (`strong_safe v3`, clean labels) | 0.3807 | 0.5157 | 0.3807 | 0.4364 |

Interpretation:

- the cleanup does not magically solve the task;
- but it removes a real source of evaluation noise;
- after this correction, the `strong_safe v3` family still remains the best current low-memory identity point;
- the next gains should now come from parser/runtime improvements and cleaner `driver_license` OCR, not from arguing with inconsistent labels.

## Clean-label retrain (`v4clean`)

To test whether the cleaned supervision alone is enough to improve the learning branch, a new one-epoch low-memory retrain was run on the cleaned `14`-sample strong split:

- checkpoint: [E:\thesis\outputs\paddleocr_vl_multidoc_lora_strong_safe_v4clean](/E:/thesis/outputs/paddleocr_vl_multidoc_lora_strong_safe_v4clean)
- benchmark: [E:\thesis\multidoc_benchmark_identity_strong_safe_v4clean.json](/E:/thesis/multidoc_benchmark_identity_strong_safe_v4clean.json)

Training result:

- train loss: `1.4984`
- val loss: `0.9117`

Benchmark metrics:

| Mode | Field Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| heuristic_pipeline | 0.2821 | 0.4095 | 0.2821 | 0.3332 |
| fine_tuned_lora (`strong_safe_v4clean`) | 0.3307 | 0.4129 | 0.3307 | 0.3667 |

Interpretation:

- this retrain remains better than its paired heuristic branch;
- however, it is still worse than the rescored `strong_safe_v3_cleanlabels` point;
- so the current bottleneck is not solved by cleaned labels alone.

This is another useful negative result:

- lower `val loss` and cleaner supervision help,
- but they do not yet produce the best extraction checkpoint on the current identity mini-set.

## California-layout parser refresh (`v4clean_r2`)

The next step was intentionally cheaper than another retrain: the California-style `driver_license` parser was tightened to reduce two concrete failure modes:

- `CLASS C` should no longer accidentally produce categories like `A,C` because of the `California` header;
- compact date strings such as `05191992` should be normalized more safely when OCR drops separators.

This produced a refreshed benchmark on the same `v4clean` adapter:

- artifact: [E:\thesis\multidoc_benchmark_identity_strong_safe_v4clean_r2.json](/E:/thesis/multidoc_benchmark_identity_strong_safe_v4clean_r2.json)

Metrics:

| Mode | Field Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| heuristic_pipeline | 0.3071 | 0.5095 | 0.3071 | 0.3732 |
| fine_tuned_lora (`strong_safe_v4clean_r2`) | 0.3307 | 0.4129 | 0.3307 | 0.3667 |

Interpretation:

- the heuristic branch improved noticeably on the same evaluation set;
- the fine-tuned branch stayed flat, which is actually informative;
- this means the current bottleneck is no longer simple parser cleanup, but OCR coverage and number recovery on the driver-license samples themselves.

In practice, this narrows the next engineering target:

- not another blind retrain,
- but a more specialized `driver_license` OCR path or better direct number-region extraction.

## Passport-heavy strong-split experiment

To improve identity diversity without adding more weak driver-license labels, three additional manually verified passport records were added:

- `passport-dataset_001`
- `passport-dataset_003`
- `passport-dataset_004`

This expanded the strong split to `14` records and produced a new one-epoch low-memory run:

- checkpoint: [E:\thesis\outputs\paddleocr_vl_multidoc_lora_strong_safe_v3data](/E:/thesis/outputs/paddleocr_vl_multidoc_lora_strong_safe_v3data)
- benchmark: [E:\thesis\multidoc_benchmark_identity_strong_safe_v3data.json](/E:/thesis/multidoc_benchmark_identity_strong_safe_v3data.json)

Training result:

- train loss: `1.4997`
- val loss: `0.9176`

Benchmark metrics:

| Mode | Field Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| heuristic_pipeline | 0.3000 | 0.4095 | 0.3000 | 0.3442 |
| fine_tuned_lora (`strong_safe_v3data`) | 0.3486 | 0.4129 | 0.3486 | 0.3762 |

Interpretation:

- the training signal improved substantially compared with earlier safe runs;
- this strongly suggests that verified passport diversity is useful for optimization;
- however, the practical identity benchmark did **not** surpass the best `strong_safe v3` parser/runtime point;
- this is another useful negative result: better `val loss` alone is not enough if the evaluation bottleneck remains dominated by weak `driver_license` extraction.

Practical takeaway:

- more diverse verified passport records are worth keeping;
- but the next real gain will likely come from either:
  - stronger `driver_license` supervision,
  - or a more specialized `driver_license` OCR/extraction path.

## Practical conclusion

- The heuristic pipeline is no longer the only credible identity branch: the fine-tuned low-memory branch can already beat it on the best `strong_safe v3` benchmark point, and that remains true after cleaning the `driver_license` labels.
- At the same time, the branch is still unstable across checkpoints: better `val loss` does not always imply a better identity benchmark.
- The next quality gains will likely come from:
  - a more specialized `driver_license` OCR path,
  - better direct recovery of the red DL number region,
  - and continued structured-output cleanup for the fine-tuned branch.

## Runtime integration note

The backend extraction service now supports loading a local PaddleOCR-VL LoRA adapter through configuration. A runtime quality gate rejects obviously repetitive or low-quality structured outputs and falls back to the heuristic OCR branch when necessary.

## Recommended next step

1. Use the new split pair:
   - [pilot_train_strong.jsonl](/E:/thesis/data/multidoc/pilot_train_strong.jsonl)
   - [pilot_train_augmented.jsonl](/E:/thesis/data/multidoc/pilot_train_augmented.jsonl)
2. Compare `strong-only` against `augmented-with-low-quality-labels` instead of mixing them blindly.
3. Improve fine-tuned JSON decoding and driver-license OCR robustness before the next full adapter iteration.

## Driver-license targeted OCR pass

To address the remaining `driver_license` bottleneck, the extraction runtime was updated with a more specialized California-style OCR path:

- a wider top-header crop for `DL / EXP / CLASS / LN / FN`
- a dedicated dates/detail crop
- stronger rejection of date-like `license_number` candidates such as `05101992`
- stricter category extraction to avoid false positives like `B` from `BRN`

This produced a new benchmark artifact:

- [E:\thesis\multidoc_benchmark_identity_strong_safe_v4clean_r4.json](/E:/thesis/multidoc_benchmark_identity_strong_safe_v4clean_r4.json)

Metrics:

| Mode | Field Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| heuristic_pipeline | 0.5112 | 0.5445 | 0.5112 | 0.5223 |
| fine_tuned_lora (`strong_safe_v4clean_r4`) | 0.3112 | 0.4095 | 0.3112 | 0.3480 |

Interpretation:

- this is the strongest `heuristic_pipeline` result on the current `identity_eval_v5` set so far;
- the new OCR path materially improved practical extraction on the problematic driver-license samples;
- the `fine_tuned_lora` branch did **not** benefit from the same gain, which means the next bottleneck is no longer simple OCR coverage alone;
- at this point the project has a clear split:
  - `heuristic_pipeline` is the best practical identity extractor,
  - `fine_tuned_lora` remains the main research branch that still needs better supervision and more stable structured decoding.

## Structured-hybrid fine-tuned recovery

The next iteration targeted the `fine_tuned_lora` branch directly. A hybrid recovery step was added:

- the structured adapter output is still used as the primary source;
- missing or clearly suspicious identity fields are then merged with the stronger heuristic OCR branch;
- the same hybrid logic is now used both in runtime extraction and in the benchmark script, so evaluation matches application behavior.

New benchmark artifact:

- [E:\thesis\multidoc_benchmark_identity_strong_safe_v4clean_r5.json](/E:/thesis/multidoc_benchmark_identity_strong_safe_v4clean_r5.json)

Metrics:

| Mode | Field Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| heuristic_pipeline | 0.5112 | 0.5445 | 0.5112 | 0.5223 |
| fine_tuned_lora (`strong_safe_v4clean_r5`) | 0.4462 | 0.4795 | 0.4462 | 0.4573 |

Interpretation:

- this is the best `fine_tuned_lora` identity result obtained so far on the current evaluation set;
- the hybrid merge substantially reduced the gap between the research branch and the practical heuristic branch;
- the heuristic pipeline still remains the best production-like path overall;
- however, the fine-tuned branch is now much more credible as a thesis result because it no longer collapses as soon as the structured decoder misses a few key fields.

## Prompt-constrained structured decoding

The next refinement tightened the structured prompt itself:

- both runtime extraction and the benchmark runner now provide an explicit JSON skeleton;
- missing values are requested as `null`;
- list fields are constrained to `[]`;
- the model is told to return exactly one JSON object and no markdown/explanations.

New benchmark artifact:

- [E:\thesis\multidoc_benchmark_identity_strong_safe_v4clean_r6.json](/E:/thesis/multidoc_benchmark_identity_strong_safe_v4clean_r6.json)

Metrics:

| Mode | Field Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| heuristic_pipeline | 0.5112 | 0.5445 | 0.5112 | 0.5223 |
| fine_tuned_lora (`strong_safe_v4clean_r6`) | 0.4862 | 0.5195 | 0.4862 | 0.4973 |

Interpretation:

- this is the strongest `fine_tuned_lora` result obtained so far on the current identity evaluation set;
- the gap to the heuristic branch narrowed again;
- the stricter schema prompt did not change the heuristic path, but it clearly improved the research branch by making structured output more disciplined;
- the current picture is now much cleaner:
  - `heuristic_pipeline` remains the best practical extractor,
  - `fine_tuned_lora` is now close enough to be presented as a meaningful learned extraction branch rather than only a fallback experiment.

## Additional verified driver-license duplicate

One more California driver-license sample (`synthetic-usa-driver-license_004`) was manually upgraded from a weak heuristic draft into a curated duplicate with verified labels and added to the strong training split.

New training run:

- checkpoint: [E:\thesis\outputs\paddleocr_vl_multidoc_lora_strong_safe_v7dl](/E:/thesis/outputs/paddleocr_vl_multidoc_lora_strong_safe_v7dl)
- summary: [training_summary.json](/E:/thesis/outputs/paddleocr_vl_multidoc_lora_strong_safe_v7dl/training_summary.json)

Training result:

- train loss: `1.5392`
- val loss: `1.0929`

Benchmark artifact:

- [E:\thesis\multidoc_benchmark_identity_strong_safe_v7dl.json](/E:/thesis/multidoc_benchmark_identity_strong_safe_v7dl.json)

Metrics:

| Mode | Field Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| heuristic_pipeline | 0.5112 | 0.5445 | 0.5112 | 0.5223 |
| fine_tuned_lora (`strong_safe_v7dl`) | 0.4862 | 0.5195 | 0.4862 | 0.4973 |

Interpretation:

- adding one more near-duplicate verified driver-license sample did **not** improve the benchmark over `r6`;
- this is another useful negative result: once the obvious label noise is removed, more of the same template does not buy much;
- the next likely gain will come from either:
  - more diverse verified `driver_license` layouts,
  - or stronger learned decoding/field grounding rather than another duplicate-heavy identity increment.

## Driver-license grounding refresh

After inspecting exported driver-license crops, it became clear that two key OCR regions (`number` and `bottom_right`) were partially aimed at empty background instead of the license text itself. The driver-license-specific crop logic was re-anchored to a tighter internal `focus` region, and date normalization was hardened to repair implausible OCR years such as `1692 -> 1992` when a matching compact date signal exists nearby.

New driver-license benchmark artifacts:

- [E:\thesis\multidoc_benchmark_driver_license_curriculum_v1_r3.json](/E:/thesis/multidoc_benchmark_driver_license_curriculum_v1_r3.json)
- [E:\thesis\multidoc_benchmark_driver_license_curriculum_v1_r4.json](/E:/thesis/multidoc_benchmark_driver_license_curriculum_v1_r4.json)

Latest driver-license metrics (`r4`):

| Mode | Field Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| heuristic_pipeline | 0.6250 | 0.7143 | 0.6250 | 0.6667 |
| fine_tuned_lora (`driver_license_curriculum_v1_r4`) | 0.6250 | 0.7143 | 0.6250 | 0.6667 |

Interpretation:

- this is the first clean sign that the driver-license bottleneck was substantially geometric rather than purely model-side;
- the heuristic branch improved sharply once the OCR regions actually landed on the `DL / EXP / ISS` text;
- the fine-tuned branch now catches up to the heuristic branch on this narrow driver-license eval, which makes the curriculum result much more convincing than the original `v1` score suggested.

## Updated mixed identity benchmark after driver-license fixes

With the improved driver-license OCR path in place, the mixed identity benchmark was rerun against the best current strong-safe checkpoint.

New benchmark artifact:

- [E:\thesis\multidoc_benchmark_identity_strong_safe_v4clean_r7.json](/E:/thesis/multidoc_benchmark_identity_strong_safe_v4clean_r7.json)

Metrics:

| Mode | Field Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| heuristic_pipeline | 0.4071 | 0.5286 | 0.4071 | 0.4570 |
| fine_tuned_lora (`strong_safe_v4clean_r7`) | 0.4357 | 0.5486 | 0.4357 | 0.4843 |

Interpretation:

- the driver-license grounding fixes improve the overall mixed identity picture, not just the isolated license eval;
- the learned branch now stays ahead of the heuristic branch again on the current `identity_eval_v5` set;
- the main remaining weakness is no longer the California-style `DL` crop geometry, but the harder passport failures and the second synthetic license sample with weak OCR signal.
