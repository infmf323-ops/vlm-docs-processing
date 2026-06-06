from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "multidoc"

TRAIN_V1_PATH = DATA_DIR / "passport_curriculum_train_v1.jsonl"
VAL_V1_PATH = DATA_DIR / "passport_curriculum_val_v1.jsonl"

TRAIN_V2_PATH = DATA_DIR / "passport_curriculum_train_v2.jsonl"
VAL_V2_PATH = DATA_DIR / "passport_curriculum_val_v2.jsonl"
PLAN_PATH = ROOT / "PASSPORT_CURRICULUM_V2_PLAN.md"


LEAKED_TRAIN_IDS = {
    "passport-dataset_000",
    "passport-dataset_002",
    "ud-synthetic__synthetic-russian-passports_001",
}


MANUAL_V2_ADDITIONS = [
    {
        "id": "ud-synthetic__synthetic-russian-passports_002",
        "image_path": "E:/thesis/data/multidoc/external/ud_biometrics/ud-synthetic__synthetic-russian-passports/ud-synthetic__synthetic-russian-passports_002.png",
        "document_type": "passport",
        "fields": {
            "document_number": "77 3781858",
            "surname": "IVANOV",
            "given_names": "ALEKSEY ALEKSANDROVICH",
            "nationality": "RUSSIAN FEDERATION",
            "date_of_birth": "23.01.1980",
            "sex": "M",
            "place_of_birth": "CHELYABINSK / USSR",
            "date_of_issue": "28.02.2019",
            "date_of_expiry": "28.02.2029",
            "issuing_authority": "??? 16001",
            "mrz": "P<RUSIVANOV<<ALEKSEY<<<<<<<<<<<<<<<<<<<<<<\n77<3781856RUS8001232M2902289<<<<<<<<<<<<08",
        },
        "source_dataset": "ud-synthetic/synthetic-russian-passports",
        "source": "manual_visual_review_v2",
        "draft_quality": "manual_verified_v2",
    },
    {
        "id": "ud-synthetic__synthetic-russian-passports_003",
        "image_path": "E:/thesis/data/multidoc/external/ud_biometrics/ud-synthetic__synthetic-russian-passports/ud-synthetic__synthetic-russian-passports_003.png",
        "document_type": "passport",
        "fields": {
            "document_number": "74 8572866",
            "surname": "MOROZOV",
            "given_names": "YEVGENIY OLEGOVICH",
            "nationality": "RUSSIAN FEDERATION",
            "date_of_birth": "05.05.1997",
            "sex": "M",
            "place_of_birth": "BARNAUL / RUSSIA",
            "date_of_issue": "23.06.2018",
            "date_of_expiry": "23.06.2028",
            "issuing_authority": "??? 63002",
            "mrz": "P<RUSMOROZOV<<YEVGENIY<<<<<<<<<<<<<<<<<<<<<\n74<8572863RUS9705054M2806239<<<<<<<<<<<<06",
        },
        "source_dataset": "ud-synthetic/synthetic-russian-passports",
        "source": "manual_visual_review_v2",
        "draft_quality": "manual_verified_v2",
    },
    {
        "id": "ud-synthetic__synthetic-french-passports_004",
        "image_path": "E:/thesis/data/multidoc/external/ud_biometrics/ud-synthetic__synthetic-french-passports/ud-synthetic__synthetic-french-passports_004.png",
        "document_type": "passport",
        "fields": {
            "document_number": "PQ8522424",
            "surname": "LAURENT",
            "given_names": "LOUIS JACQUES",
            "nationality": "FRAN?AISE",
            "date_of_birth": "19.09.2000",
            "sex": "M",
            "place_of_birth": "STRASBOURG",
            "date_of_issue": "02.10.2023",
            "date_of_expiry": "02.10.2033",
            "issuing_authority": "PR?FECTURE DU VAL-DE-MARNE",
            "mrz": "P<FRALAURENT<<DIDIER<<<<<<<<<<<<<<<<<<<<<<\nPQ85224242FRA0009195M3310023<<<<<<<<<<<<08",
        },
        "source_dataset": "ud-synthetic/synthetic-french-passports",
        "source": "manual_visual_review_v2",
        "draft_quality": "manual_verified_v2",
        "review_note": "Visible name fields disagree with MRZ given-name token; visible fields were preferred for structured labels and raw MRZ was preserved.",
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
    return f"""# Passport Curriculum V2 Plan

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

{chr(10).join(f"- `{row_id}`" for row_id in kept_ids)}

## New rows promoted into train

{chr(10).join(f"- `{row_id}`" for row_id in added_ids)}

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
"""


def main() -> None:
    train_v1 = load_jsonl(TRAIN_V1_PATH)
    val_v1 = load_jsonl(VAL_V1_PATH)

    base_rows = [row for row in train_v1 if row.get("id") not in LEAKED_TRAIN_IDS]
    train_v2 = [*base_rows, *MANUAL_V2_ADDITIONS]

    save_jsonl(TRAIN_V2_PATH, train_v2)
    save_jsonl(VAL_V2_PATH, val_v1)
    PLAN_PATH.write_text(build_plan_text(base_rows, MANUAL_V2_ADDITIONS), encoding="utf-8")

    print(TRAIN_V2_PATH)
    print(VAL_V2_PATH)
    print(PLAN_PATH)
    print(f"base_rows={len(base_rows)}")
    print(f"added_rows={len(MANUAL_V2_ADDITIONS)}")
    print(f"train_v2_rows={len(train_v2)}")
    print(f"val_v2_rows={len(val_v1)}")


if __name__ == "__main__":
    main()
