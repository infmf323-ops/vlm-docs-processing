from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from datasets import load_dataset


ROOT = Path("E:/thesis")
OUTPUT_QUEUE = ROOT / "data" / "multidoc" / "annotation_queue.jsonl"
EXTERNAL_ROOT = ROOT / "data" / "multidoc" / "external" / "ud_biometrics"


def load_queue() -> list[dict[str, Any]]:
    if not OUTPUT_QUEUE.exists():
        return []
    rows: list[dict[str, Any]] = []
    with OUTPUT_QUEUE.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_queue(rows: list[dict[str, Any]]) -> None:
    OUTPUT_QUEUE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_QUEUE.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def empty_passport_fields() -> dict[str, Any]:
    return {
        "document_number": None,
        "surname": None,
        "given_names": None,
        "nationality": None,
        "date_of_birth": None,
        "sex": None,
        "place_of_birth": None,
        "date_of_issue": None,
        "date_of_expiry": None,
        "issuing_authority": None,
        "mrz": None,
    }


def empty_driver_license_fields() -> dict[str, Any]:
    return {
        "document_number": None,
        "surname": None,
        "given_names": None,
        "date_of_birth": None,
        "date_of_issue": None,
        "date_of_expiry": None,
        "issuing_authority": None,
        "address": None,
        "categories": [],
    }


def slugify_dataset_name(dataset_name: str) -> str:
    slug = dataset_name.replace("/", "__")
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", slug)
    return slug.strip("-")


def infer_output_id(dataset_name: str, index: int) -> str:
    return f"{slugify_dataset_name(dataset_name)}_{index:03d}"


def export_samples(
    dataset_name: str,
    document_type: str,
    limit: int,
    rows: list[dict[str, Any]],
    existing_ids: set[str],
) -> int:
    dataset = load_dataset(dataset_name, split=f"train[:{limit}]")
    dataset_slug = slugify_dataset_name(dataset_name)
    output_dir = EXTERNAL_ROOT / dataset_slug
    output_dir.mkdir(parents=True, exist_ok=True)

    added = 0
    for index, sample in enumerate(dataset):
        row_id = infer_output_id(dataset_name, index)
        if row_id in existing_ids:
            continue

        image = sample["image"]
        image_path = output_dir / f"{row_id}.png"
        image.save(image_path)

        row = {
            "id": row_id,
            "image_path": str(image_path).replace("\\", "/"),
            "document_type": document_type,
            "fields": empty_passport_fields() if document_type == "passport" else empty_driver_license_fields(),
            "status": "needs_annotation_external",
            "source_dataset": dataset_name,
        }
        if "label" in sample:
            row["source_label"] = sample["label"]

        rows.append(row)
        existing_ids.add(row_id)
        added += 1

    return added


def main() -> None:
    rows = load_queue()
    existing_ids = {str(row.get("id")) for row in rows}

    dataset_specs = [
        ("ud-biometrics/passport-dataset", "passport", 5),
        ("ud-biometrics/synthetic-usa-driver-license", "driver_license", 5),
        ("ud-synthetic/synthetic-printed-usa-passports", "passport", 5),
        ("ud-synthetic/synthetic-printed-german-passports", "passport", 5),
        ("ud-synthetic/synthetic-french-passports", "passport", 5),
        ("ud-synthetic/synthetic-japanese-passports", "passport", 5),
        ("ud-synthetic/synthetic-russian-passports", "passport", 5),
    ]

    added_counts: dict[str, int] = {}
    for dataset_name, document_type, limit in dataset_specs:
        try:
            added_counts[dataset_name] = export_samples(
                dataset_name=dataset_name,
                document_type=document_type,
                limit=limit,
                rows=rows,
                existing_ids=existing_ids,
            )
        except Exception as exc:
            added_counts[dataset_name] = -1
            print(f"dataset_failed={dataset_name} error={exc}")

    save_queue(rows)
    print(OUTPUT_QUEUE)
    for dataset_name, count in added_counts.items():
        print(f"{dataset_name}={count}")
    print(f"total_rows={len(rows)}")


if __name__ == "__main__":
    main()
