# Training Results (2026-04-17)

## Current Model

`Bennet1996/donut-small`

Current fine-tuned checkpoint:

- `E:\thesis\outputs\Bennet1996_donut-small_ft_best`

## Is It Better Than Before?

Yes.

Previous stronger local run:

- train samples: `199`
- val samples: `49`
- epochs: `5`
- best val loss: `1.2529`

Improved run:

- train samples: `424`
- val samples: `49`
- epochs: `8`
- max length: `256`
- gradient accumulation steps: `4`
- learning rate: `1.5e-5`
- best val loss: `0.7833`

Validation loss improved from `1.2529` to `0.7833`.

## Improved Training Setup

- Dataset: `katanaml-org/invoices-donut-data-v1`
- Model: `Bennet1996/donut-small`
- Device: `NVIDIA GeForce RTX 3080`
- CUDA: enabled
- Image size: `640x480`
- Max target length: `256`
- Scheduler: `cosine`
- Warmup ratio: `0.08`
- Weight decay: `0.01`
- Gradient clipping: `1.0`
- Early stopping support: enabled in code

## Metrics by Epoch

| Epoch | Train Loss | Val Loss |
|---|---:|---:|
| 1 | 6.8848 | 3.6028 |
| 2 | 3.1864 | 2.2377 |
| 3 | 1.5537 | 1.2654 |
| 4 | 1.0692 | 0.9869 |
| 5 | 0.8178 | 0.8566 |
| 6 | 0.6814 | 0.8004 |
| 7 | 0.6246 | 0.7852 |
| 8 | 0.6102 | 0.7833 |

## Best Result

- Best validation loss: `0.7832700318219711`
- Best epoch: `8`

## Dataset Notes

- Training split used: `424` valid samples
- Validation split used: `49` valid samples
- Skipped during preprocessing:
  - `1` training sample
  - `1` validation sample
- Reason: `ground_truth` became empty after serialization

## Out-of-the-Box vs Fine-Tuned

Two comparisons were run:

- a quick qualitative comparison on `5` validation samples;
- a full field-level evaluation on the whole validation split (`49` valid documents).

Base model:

- `Bennet1996/donut-small`

Fine-tuned model:

- `E:\thesis\outputs\Bennet1996_donut-small_ft_best`

### Summary

- Full validation-set comparison confirms the fine-tuned model is much better than the base model.
- The base model did not recover the expected invoice schema on the validation set.
- The fine-tuned model recovered multiple key fields with high exact-match accuracy.

### Full Validation-Set Results

Validation documents evaluated: `49`

Document-level exact match:

- base model: `0.0000`
- fine-tuned model: `0.0000`

Field-level exact-match accuracy:

| Field | Base | Fine-tuned |
|---|---:|---:|
| `invoice_no` | 0.0000 | 1.0000 |
| `invoice_date` | 0.0000 | 1.0000 |
| `seller` | 0.0000 | 0.0000 |
| `client` | 0.0000 | 0.0000 |
| `seller_tax_id` | 0.0000 | 0.9592 |
| `client_tax_id` | 0.0000 | 1.0000 |
| `iban` | 0.0000 | 0.3878 |
| `total_net_worth` | 0.0000 | 0.9184 |
| `total_vat` | 0.0000 | 0.9184 |
| `total_gross_worth` | 0.0000 | 0.8571 |

Interpretation of full-set evaluation:

- the fine-tuned model is decisively better on every important structured field;
- invoice number and invoice date are perfect on this validation split;
- tax IDs are almost perfect;
- totals are strong;
- seller/client free-text addresses are still weak under strict exact match;
- IBAN is improved but still unstable.

### Quick Qualitative Comparison

- Exact full-string matches on sampled examples:
  - base model: `0 / 5`
  - fine-tuned model: `0 / 5`
- Despite that, the fine-tuned model is clearly better qualitatively.
- The base model often produces unrelated document structure and even irrelevant foreign-language fields.
- The fine-tuned model usually predicts the invoice structure correctly and often gets invoice number, date, tax IDs, and totals close or correct, though it still duplicates some closing tags and sometimes slightly distorts addresses or amounts.

### Sample 1

- Dataset index: `41`
- Ground truth key fields:
  - invoice_no: `32530472`
  - invoice_date: `08/27/2015`
  - gross: `$ 187,00`
- Base model:
  - generated unrelated fields like `StraГџe`, `Geburtsdatum`, `Gesamtbrutto`
- Fine-tuned model:
  - recovered `invoice_no = 32530472`
  - recovered `invoice_date = 08/27/2015`
  - recovered gross total close to correct: `$ 187,00`
  - still distorted seller/client text and duplicated closing tags

### Sample 2

- Dataset index: `7`
- Ground truth key fields:
  - invoice_no: `67583819`
  - invoice_date: `06/01/2012`
  - net: `$ 889,67`
  - gross: `978,64`
- Base model:
  - produced unrelated output and wrong structure
- Fine-tuned model:
  - recovered `invoice_no = 67583819`
  - recovered `invoice_date = 06/01/2012`
  - recovered `seller_tax_id = 911-94-3128`
  - recovered `client_tax_id = 911-97-3515`
  - gross amount was close but not exact

### Sample 3

- Dataset index: `1`
- Ground truth key fields:
  - invoice_no: `16220332`
  - invoice_date: `05/15/2017`
  - gross: `$ 69138,73`
- Base model:
  - output was largely irrelevant and structurally wrong
- Fine-tuned model:
  - recovered invoice number and date correctly
  - recovered tax IDs correctly
  - recovered IBAN correctly
  - gross amount stayed very close
  - extra duplicated suffix text still appears

### Sample 4

- Dataset index: `48`
- Ground truth key fields:
  - invoice_no: `37959814`
  - invoice_date: `07/12/2013`
  - net: `$6623,62`
  - gross: `$7285,98`
- Base model:
  - collapsed into unusable repetitive text
- Fine-tuned model:
  - recovered invoice number and date correctly
  - recovered seller/client tax IDs correctly
  - recovered IBAN correctly
  - recovered gross total correctly
  - VAT value had an error and output still had repeated tail text

### Sample 5

- Dataset index: `17`
- Ground truth key fields:
  - invoice_no: `64281058`
  - invoice_date: `11/08/2019`
  - net: `$ 14 014,99`
  - gross: `$ 15 416,49`
- Base model:
  - produced mostly irrelevant template-like output
- Fine-tuned model:
  - recovered invoice number and date correctly
  - recovered both tax IDs correctly
  - recovered IBAN correctly
  - gross amount was very close
  - still shows duplicated structural ending

## Practical Interpretation

The fine-tuned model is already much more useful than the out-of-the-box model for this invoice task.

Main improvement:

- the model learned the expected invoice schema;
- it now predicts task-relevant fields instead of generic unrelated text;
- core identifiers are often correct.

Main remaining issues:

- duplicated closing tags;
- occasional corruption of addresses;
- occasional small numeric mistakes in totals or VAT.

## Useful Next Step

The next high-value improvement would be to evaluate field-level accuracy automatically, for example:

- invoice number accuracy
- date accuracy
- seller/client tax ID accuracy
- gross/net/vat exact match rate

That would be more informative than loss alone.
