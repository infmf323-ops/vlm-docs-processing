from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "multidoc"

TRAIN_V2_PATH = DATA_DIR / "passport_curriculum_train_v2.jsonl"
VAL_V2_PATH = DATA_DIR / "passport_curriculum_val_v2.jsonl"

TRAIN_V3_PATH = DATA_DIR / "passport_curriculum_train_v3.jsonl"
VAL_V3_PATH = DATA_DIR / "passport_curriculum_val_v3.jsonl"
PLAN_PATH = ROOT / "PASSPORT_CURRICULUM_V3_PLAN.md"


MANUAL_V3_ADDITIONS = [
    {
        "id": "ud-synthetic__synthetic-russian-passports_004",
        "image_path": "E:/thesis/data/multidoc/external/ud_biometrics/ud-synthetic__synthetic-russian-passports/ud-synthetic__synthetic-russian-passports_004.png",
        "document_type": "passport",
        "fields": {
            "document_number": "73 1960787",
            "surname": "PETROV",
            "given_names": "DMITRIY NIKOLAEVICH",
            "nationality": "RUSSIAN FEDERATION",
            "date_of_birth": "12.01.1995",
            "sex": "M",
            "place_of_birth": "KAZAN / RUSSIA",
            "date_of_issue": "04.08.2025",
            "date_of_expiry": "04.08.2035",
            "issuing_authority": "МВД 50001",
            "mrz": "P<RUSPETROV<<DMITRIY<<<<<<<<<<<<<<<<<<<<<<\n73<1960787RUS9501120M3508046<<<<<<<<<<<<00",
        },
        "source_dataset": "ud-synthetic/synthetic-russian-passports",
        "source": "manual_visual_review_v3",
        "draft_quality": "manual_verified_v3",
    },
    {
        "id": "ud-synthetic__synthetic-japanese-passports_000",
        "image_path": "E:/thesis/data/multidoc/external/ud_biometrics/ud-synthetic__synthetic-japanese-passports/ud-synthetic__synthetic-japanese-passports_000.png",
        "document_type": "passport",
        "fields": {
            "document_number": "XD5569211",
            "surname": "ISHIKAWA",
            "given_names": "MARIKO",
            "nationality": "JAPAN",
            "date_of_birth": "14 AUG 1984",
            "sex": "F",
            "place_of_birth": "SENDAI",
            "date_of_issue": "24 DEC 2019",
            "date_of_expiry": "24 DEC 2029",
            "issuing_authority": "MINISTRY OF FOREIGN AFFAIRS",
            "mrz": "P<JPNISHIKAWA<<MARIKO<<<<<<<<<<<<<<<<<<<<\nXD55692115JPN8408141F2912246<<<<<<<<<<<<06",
        },
        "source_dataset": "ud-synthetic/synthetic-japanese-passports",
        "source": "manual_visual_review_v3",
        "draft_quality": "manual_verified_v3",
    },
    {
        "id": "ud-synthetic__synthetic-japanese-passports_001",
        "image_path": "E:/thesis/data/multidoc/external/ud_biometrics/ud-synthetic__synthetic-japanese-passports/ud-synthetic__synthetic-japanese-passports_001.png",
        "document_type": "passport",
        "fields": {
            "document_number": "PF9368628",
            "surname": "SASAKI",
            "given_names": "EMI",
            "nationality": "JAPAN",
            "date_of_birth": "03 DEC 1982",
            "sex": "F",
            "place_of_birth": "SHIZUOKA",
            "date_of_issue": "05 JAN 2023",
            "date_of_expiry": "05 JAN 2033",
            "issuing_authority": "MINISTRY OF FOREIGN AFFAIRS",
            "mrz": "P<JPNSASAKI<<EMI<<<<<<<<<<<<<<<<<<<<<<<<<<\nPF93686282JPN8212030F3301052<<<<<<<<<<<<08",
        },
        "source_dataset": "ud-synthetic/synthetic-japanese-passports",
        "source": "manual_visual_review_v3",
        "draft_quality": "manual_verified_v3",
    },
    {
        "id": "ud-synthetic__synthetic-japanese-passports_002",
        "image_path": "E:/thesis/data/multidoc/external/ud_biometrics/ud-synthetic__synthetic-japanese-passports/ud-synthetic__synthetic-japanese-passports_002.png",
        "document_type": "passport",
        "fields": {
            "document_number": "QK7905188",
            "surname": "YOSHIDA",
            "given_names": "KAORI",
            "nationality": "JAPAN",
            "date_of_birth": "30 DEC 1979",
            "sex": "F",
            "place_of_birth": "SAITAMA",
            "date_of_issue": "12 OCT 2019",
            "date_of_expiry": "12 OCT 2029",
            "issuing_authority": "NAGOYA",
            "mrz": "P<JPNYOSHIDA<<KAORI<<<<<<<<<<<<<<<<<<<<<<\nQK79051886JPN7912300F2910127<<<<<<<<<<<<02",
        },
        "source_dataset": "ud-synthetic/synthetic-japanese-passports",
        "source": "manual_visual_review_v3",
        "draft_quality": "manual_verified_v3",
    },
    {
        "id": "ud-synthetic__synthetic-japanese-passports_003",
        "image_path": "E:/thesis/data/multidoc/external/ud_biometrics/ud-synthetic__synthetic-japanese-passports/ud-synthetic__synthetic-japanese-passports_003.png",
        "document_type": "passport",
        "fields": {
            "document_number": "QI7859558",
            "surname": "YAMAGUCHI",
            "given_names": "AKIRA",
            "nationality": "JAPAN",
            "date_of_birth": "26 SEP 1968",
            "sex": "M",
            "place_of_birth": "TOKYO",
            "date_of_issue": "06 OCT 2020",
            "date_of_expiry": "06 OCT 2030",
            "issuing_authority": "FUKUOKA",
            "mrz": "P<JPNYAMAGUCHI<<AKIRA<<<<<<<<<<<<<<<<<<<<\nQI78595581JPN6809261M3010068<<<<<<<<<<<<00",
        },
        "source_dataset": "ud-synthetic/synthetic-japanese-passports",
        "source": "manual_visual_review_v3",
        "draft_quality": "manual_verified_v3",
    },
    {
        "id": "ud-synthetic__synthetic-japanese-passports_004",
        "image_path": "E:/thesis/data/multidoc/external/ud_biometrics/ud-synthetic__synthetic-japanese-passports/ud-synthetic__synthetic-japanese-passports_004.png",
        "document_type": "passport",
        "fields": {
            "document_number": "ZU9873448",
            "surname": "YAMAZAKI",
            "given_names": "SHOTA",
            "nationality": "JAPAN",
            "date_of_birth": "21 MAR 1981",
            "sex": "M",
            "place_of_birth": "KOBE",
            "date_of_issue": "28 AUG 2021",
            "date_of_expiry": "28 AUG 2031",
            "issuing_authority": "FUKUOKA",
            "mrz": "P<JPNYAMAZAKI<<SHOTA<<<<<<<<<<<<<<<<<<<<\nZU98734482JPN8103217M3108284<<<<<<<<<<<<00",
        },
        "source_dataset": "ud-synthetic/synthetic-japanese-passports",
        "source": "manual_visual_review_v3",
        "draft_quality": "manual_verified_v3",
    },
    {
        "id": "ud-synthetic__synthetic-printed-usa-passports_001",
        "image_path": "E:/thesis/data/multidoc/external/ud_biometrics/ud-synthetic__synthetic-printed-usa-passports/ud-synthetic__synthetic-printed-usa-passports_001.png",
        "document_type": "passport",
        "fields": {
            "document_number": "J17236842",
            "surname": "KING",
            "given_names": "CYNTHIA",
            "nationality": "UNITED STATES OF AMERICA",
            "date_of_birth": "10 JUN 1965",
            "sex": "F",
            "place_of_birth": "TEXAS, U.S.A.",
            "date_of_issue": "21 OCT 2025",
            "date_of_expiry": "21 OCT 2035",
            "issuing_authority": "UNITED STATES DEPARTMENT OF STATE",
            "mrz": "P<USAKING<<CYNTHIA<<<<<<<<<<<<<<<<<<<<<<\nJ172368422USA6506102F3510214<<<<<<<<<<<<06",
        },
        "source_dataset": "ud-synthetic/synthetic-printed-usa-passports",
        "source": "manual_visual_review_v3",
        "draft_quality": "manual_verified_v3",
    },
    {
        "id": "ud-synthetic__synthetic-printed-german-passports_001",
        "image_path": "E:/thesis/data/multidoc/external/ud_biometrics/ud-synthetic__synthetic-printed-german-passports/ud-synthetic__synthetic-printed-german-passports_001.png",
        "document_type": "passport",
        "fields": {
            "document_number": "FV5661398",
            "surname": "SCHAEFER",
            "given_names": "ANGELIKA",
            "nationality": "GERMANY",
            "date_of_birth": "25 NOV 1996",
            "sex": "F",
            "place_of_birth": "ESSEN",
            "date_of_issue": "23 DEC 2022",
            "date_of_expiry": "23 DEC 2032",
            "issuing_authority": "STADT WUPPERTAL",
            "mrz": "PPD<<SCHAEFER<<ANGELIKA<<<<<<<<<<<<<<<<<<<<\nFV56613980D<<9611250F3212231<<<<<<<<<<<<06",
        },
        "source_dataset": "ud-synthetic/synthetic-printed-german-passports",
        "source": "manual_visual_review_v3",
        "draft_quality": "manual_verified_v3",
    },
    {
        "id": "ud-synthetic__synthetic-printed-german-passports_002",
        "image_path": "E:/thesis/data/multidoc/external/ud_biometrics/ud-synthetic__synthetic-printed-german-passports/ud-synthetic__synthetic-printed-german-passports_002.png",
        "document_type": "passport",
        "fields": {
            "document_number": "DX4785821",
            "surname": "HUBER",
            "given_names": "ERIKA",
            "nationality": "GERMANY",
            "date_of_birth": "03 AUG 1982",
            "sex": "F",
            "place_of_birth": "FRANKFURT AM MAIN",
            "date_of_issue": "18 APR 2016",
            "date_of_expiry": "18 APR 2026",
            "issuing_authority": "STADT WUPPERTAL",
            "mrz": "PPD<<HUBER<<ERIKA<<<<<<<<<<<<<<<<<<<<<<<\nDX47858215D<<8208031F2604181<<<<<<<<<<<<04",
        },
        "source_dataset": "ud-synthetic/synthetic-printed-german-passports",
        "source": "manual_visual_review_v3",
        "draft_quality": "manual_verified_v3",
    },
]


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def build_plan_text(base_rows: list[dict], additions: list[dict]) -> str:
    kept_ids = [row["id"] for row in base_rows]
    added_ids = [row["id"] for row in additions]
    return f"""# Passport Curriculum V3 Plan

Date: 2026-05-20

## Goal

Expand the passport-specific training branch from the leak-reduced `v2` subset (`8` train rows)
to a genuinely larger and more diverse `v3` subset without reintroducing overlap with the
current `passport_eval_diverse_v1.jsonl`.

## Resulting size

- `passport_curriculum_train_v3.jsonl` -> `17` train rows
- `passport_curriculum_val_v3.jsonl` -> `1` validation row

## Base rows kept from `v2`

{chr(10).join(f"- `{row_id}`" for row_id in kept_ids)}

## New rows promoted into `v3`

{chr(10).join(f"- `{row_id}`" for row_id in added_ids)}

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
"""


def main() -> None:
    train_v2 = load_jsonl(TRAIN_V2_PATH)
    val_v2 = load_jsonl(VAL_V2_PATH)

    seen_ids = {row["id"] for row in train_v2}
    additions = [row for row in MANUAL_V3_ADDITIONS if row["id"] not in seen_ids]
    train_v3 = [*train_v2, *additions]

    save_jsonl(TRAIN_V3_PATH, train_v3)
    save_jsonl(VAL_V3_PATH, val_v2)
    PLAN_PATH.write_text(build_plan_text(train_v2, additions), encoding="utf-8")

    print(TRAIN_V3_PATH)
    print(VAL_V3_PATH)
    print(PLAN_PATH)
    print(f"base_rows={len(train_v2)}")
    print(f"added_rows={len(additions)}")
    print(f"train_v3_rows={len(train_v3)}")
    print(f"val_v3_rows={len(val_v2)}")


if __name__ == "__main__":
    main()
