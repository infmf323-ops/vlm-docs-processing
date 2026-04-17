# Training Results (2026-04-17)

## Current Model

`Bennet1996/donut-small`

This model is currently used in the local training pipeline instead of the heavier `naver-clova-ix/donut-base`.

## Training Setup

- Dataset: `katanaml-org/invoices-donut-data-v1`
- Train samples used: `199`
- Validation samples used: `49`
- Epochs: `5`
- Image size: `640x480`
- Max target length: `192`
- Gradient accumulation steps: `2`
- Learning rate: `2e-5`
- Weight decay: `0.01`
- Scheduler: `cosine`
- Warmup ratio: `0.1`
- Gradient clipping: `1.0`
- Device: `NVIDIA GeForce RTX 3080`
- CUDA: enabled

## Notes

- During preprocessing, `1` training sample and `1` validation sample were skipped because `ground_truth` became empty after serialization.
- The training loop was adapted for local GPU execution and stabilized for this lighter model.

## Metrics by Epoch

| Epoch | Train Loss | Val Loss |
|---|---:|---:|
| 1 | 6.6495 | 3.6329 |
| 2 | 3.1270 | 2.1881 |
| 3 | 1.5517 | 1.3999 |
| 4 | 1.1504 | 1.2745 |
| 5 | 1.0323 | 1.2529 |

## Best Result

- Best validation loss: `1.2528536891450688`
- Best checkpoint saved locally to:
  - `E:\thesis\outputs\Bennet1996_donut-small_ft_best`

## Interpretation

The run is a good local baseline:

- training is stable;
- validation loss consistently improves across epochs;
- the model is now lightweight enough to train efficiently on the local RTX 3080.

A useful next step would be to evaluate actual field extraction quality on validation examples, because task quality is better judged by predicted invoice fields than by loss alone.
