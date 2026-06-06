from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path("E:/thesis")


def normalize_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def first_present(fields: dict[str, Any], candidates: list[str]) -> str | None:
    lowered = {str(key).lower(): value for key, value in fields.items()}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return normalize_string(lowered[candidate.lower()])
    return None


def infer_document_type(template_name: str) -> str:
    template = template_name.lower()
    if template.startswith("pp_") or "passport" in template:
        return "passport"
    if template.startswith("id_card") or "idcard" in template:
        return "id_card"
    if template.startswith("rp_") or "residence" in template:
        return "id_card"
    return "other"


def empty_fields(document_type: str) -> dict[str, Any]:
    if document_type == "passport":
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
    if document_type == "id_card":
        return {
            "document_number": None,
            "surname": None,
            "given_names": None,
            "date_of_birth": None,
            "sex": None,
            "nationality": None,
            "date_of_issue": None,
            "date_of_expiry": None,
            "issuing_authority": None,
            "address": None,
        }
    return {"raw_fields": {}}


def map_passport_fields(fields: dict[str, Any]) -> dict[str, Any]:
    mapped = empty_fields("passport")
    mapped["document_number"] = first_present(
        fields,
        [
            "document_number",
            "document no",
            "passport_number",
            "passport_no",
            "number",
            "documentnumber",
        ],
    )
    mapped["surname"] = first_present(fields, ["surname", "last_name", "family_name", "lastname"])
    mapped["given_names"] = first_present(
        fields,
        ["given_names", "given_names_1", "given_name", "first_name", "firstname", "names"],
    )
    mapped["nationality"] = first_present(fields, ["nationality", "country", "citizenship"])
    mapped["date_of_birth"] = first_present(
        fields,
        ["date_of_birth", "birth_date", "dob", "dateofbirth"],
    )
    mapped["sex"] = first_present(fields, ["sex", "gender"])
    mapped["place_of_birth"] = first_present(
        fields,
        ["place_of_birth", "birth_place", "placeofbirth"],
    )
    mapped["date_of_issue"] = first_present(
        fields,
        ["date_of_issue", "issue_date", "issued_on", "dateofissue"],
    )
    mapped["date_of_expiry"] = first_present(
        fields,
        ["date_of_expiry", "expiry_date", "expiration_date", "dateofexpiry"],
    )
    mapped["issuing_authority"] = first_present(
        fields,
        ["issuing_authority", "authority", "issuer"],
    )
    mapped["mrz"] = first_present(fields, ["mrz", "machine_readable_zone"])
    return mapped


def map_id_card_fields(fields: dict[str, Any]) -> dict[str, Any]:
    mapped = empty_fields("id_card")
    mapped["document_number"] = first_present(
        fields,
        ["document_number", "card_number", "id_number", "document no", "number"],
    )
    mapped["surname"] = first_present(fields, ["surname", "last_name", "family_name", "lastname"])
    mapped["given_names"] = first_present(
        fields,
        ["given_names", "given_name", "first_name", "firstname", "names"],
    )
    mapped["date_of_birth"] = first_present(fields, ["date_of_birth", "birth_date", "dob"])
    mapped["sex"] = first_present(fields, ["sex", "gender"])
    mapped["nationality"] = first_present(fields, ["nationality", "country", "citizenship"])
    mapped["date_of_issue"] = first_present(fields, ["date_of_issue", "issue_date", "issued_on"])
    mapped["date_of_expiry"] = first_present(fields, ["date_of_expiry", "expiry_date", "expiration_date"])
    mapped["issuing_authority"] = first_present(fields, ["issuing_authority", "authority", "issuer"])
    mapped["address"] = first_present(fields, ["address", "holder_address", "residence"])
    return mapped


def map_fields(document_type: str, raw_fields: dict[str, Any]) -> dict[str, Any]:
    if document_type == "passport":
        return map_passport_fields(raw_fields)
    if document_type == "id_card":
        return map_id_card_fields(raw_fields)
    return {"raw_fields": raw_fields}


def convert_document(
    document: dict[str, Any],
    images_dir: Path,
    output_images_dir: Path | None,
) -> dict[str, Any] | None:
    annotations = document.get("annotations") or []
    if not annotations:
        return None

    annotation = annotations[0]
    template_name = str(annotation.get("template") or "unknown")
    document_type = infer_document_type(template_name)
    raw_fields = annotation.get("fields") or {}
    mapped_fields = map_fields(document_type, raw_fields)

    filename = document.get("filename")
    if not filename:
        return None
    source_image = images_dir / filename
    if not source_image.exists():
        return None

    image_path = source_image
    if output_images_dir is not None:
        output_images_dir.mkdir(parents=True, exist_ok=True)
        destination = output_images_dir / source_image.name
        if not destination.exists():
            shutil.copy2(source_image, destination)
        image_path = destination

    document_id = document.get("_id", document.get("id", source_image.stem))
    return {
        "id": f"docxpand_{document_id}",
        "image_path": str(image_path).replace("\\", "/"),
        "document_type": document_type,
        "fields": mapped_fields,
        "source": "docxpand_25k",
        "source_template": template_name,
        "raw_source_fields": raw_fields,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-json", required=True, help="Path to DocXPand JSON dataset")
    parser.add_argument("--images-dir", required=True, help="Path to extracted DocXPand images directory")
    parser.add_argument("--output-jsonl", required=True, help="Target JSONL path in our multidoc format")
    parser.add_argument("--copy-images-dir", help="Optional destination directory for copied images")
    parser.add_argument("--limit", type=int, default=0, help="Optional row limit")
    args = parser.parse_args()

    dataset_json = Path(args.dataset_json)
    images_dir = Path(args.images_dir)
    output_jsonl = Path(args.output_jsonl)
    copy_images_dir = Path(args.copy_images_dir) if args.copy_images_dir else None

    dataset = json.loads(dataset_json.read_text(encoding="utf-8"))
    documents = dataset.get("documents") or []

    rows: list[dict[str, Any]] = []
    for document in documents:
        row = convert_document(document, images_dir=images_dir, output_images_dir=copy_images_dir)
        if row is None:
            continue
        rows.append(row)
        if args.limit and len(rows) >= args.limit:
            break

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    type_counts: dict[str, int] = {}
    for row in rows:
        type_counts[row["document_type"]] = type_counts.get(row["document_type"], 0) + 1

    print(output_jsonl)
    print(f"rows={len(rows)}")
    print("type_counts=" + json.dumps(type_counts, ensure_ascii=False))


if __name__ == "__main__":
    main()
