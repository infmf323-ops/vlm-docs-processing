# Passport Transfer Version Comparison

Date fixed: 2026-05-21

## Purpose

This note compares the current passport transfer-learning split families `v1`, `v2`, and `v3`.

## Version summary

| Version | Pretrain idea | Pretrain train | Pretrain val | Russian finetune train | Russian finetune val |
|---|---:|---:|---:|---:|---:|
| `v1` | All synthetic passport sources, including Russian and printed variants | 36 | 9 | 3 | 1 |
| `v2` | All non-Russian synthetic passport sources, including printed variants | 32 | 8 | 3 | 1 |
| `v3` | Only non-Russian, non-printed synthetic passport sources | 24 | 6 | 3 | 1 |

## Pretrain source composition

### v1

- `ud-synthetic/synthetic-chinese-passports`: `4` rows
- `ud-synthetic/synthetic-french-passports`: `4` rows
- `ud-synthetic/synthetic-greek-passports`: `4` rows
- `ud-synthetic/synthetic-indian-passports`: `4` rows
- `ud-synthetic/synthetic-japanese-passports`: `4` rows
- `ud-synthetic/synthetic-printed-german-passports`: `4` rows (printed)
- `ud-synthetic/synthetic-printed-usa-passports`: `4` rows (printed)
- `ud-synthetic/synthetic-russian-passports`: `4` rows (russian)
- `ud-synthetic/synthetic-turkish-passports`: `4` rows

### v2

- `ud-synthetic/synthetic-chinese-passports`: `4` rows
- `ud-synthetic/synthetic-french-passports`: `4` rows
- `ud-synthetic/synthetic-greek-passports`: `4` rows
- `ud-synthetic/synthetic-indian-passports`: `4` rows
- `ud-synthetic/synthetic-japanese-passports`: `4` rows
- `ud-synthetic/synthetic-printed-german-passports`: `4` rows (printed)
- `ud-synthetic/synthetic-printed-usa-passports`: `4` rows (printed)
- `ud-synthetic/synthetic-turkish-passports`: `4` rows

### v3

- `ud-synthetic/synthetic-chinese-passports`: `4` rows
- `ud-synthetic/synthetic-french-passports`: `4` rows
- `ud-synthetic/synthetic-greek-passports`: `4` rows
- `ud-synthetic/synthetic-indian-passports`: `4` rows
- `ud-synthetic/synthetic-japanese-passports`: `4` rows
- `ud-synthetic/synthetic-turkish-passports`: `4` rows

## Recommendation

- `v3` is the cleanest mainline baseline for the next transfer experiment.
- `v2` is the wider non-Russian baseline if we want more source diversity and are willing to keep printed variants.
- `v1` should now be treated mostly as an ablation baseline because it leaks Russian structure into the broad pretrain stage.

## Practical meaning

- If `v3` beats `v2`, then printed variants were likely hurting more than helping.
- If `v2` beats `v3`, then printed variants may still provide useful passport-layout diversity.
- If `v1` beats both, then Russian overlap in the pretrain stage is doing a lot of work and the adaptation story is weaker than we want.

