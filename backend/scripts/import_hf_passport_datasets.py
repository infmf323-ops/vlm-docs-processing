from __future__ import annotations

import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any

from huggingface_hub import list_repo_files, snapshot_download


ROOT = Path("E:/thesis")
DATA_DIR = ROOT / "data" / "multidoc"
OUTPUT_JSONL = DATA_DIR / "passport_hf_synthetic_source_v1.jsonl"
EXTERNAL_ROOT = DATA_DIR / "external" / "hf_passports_v1"

DATASET_REPOS = [
    "ud-synthetic/synthetic-russian-passports",
    "ud-synthetic/synthetic-japanese-passports",
    "ud-synthetic/synthetic-french-passports",
    "ud-synthetic/synthetic-printed-usa-passports",
    "ud-synthetic/synthetic-printed-german-passports",
    "ud-synthetic/synthetic-turkish-passports",
    "ud-synthetic/synthetic-greek-passports",
    "ud-synthetic/synthetic-indian-passports",
    "ud-synthetic/synthetic-chinese-passports",
]

NATIONALITY_MAP = {
    "RUS": "RUSSIAN FEDERATION",
    "RUSSIAN FEDERATION": "RUSSIAN FEDERATION",
    "JPN": "JAPAN",
    "JAPAN": "JAPAN",
    "FRA": "FRANCE",
    "FRANCE": "FRANCE",
    "USA": "UNITED STATES OF AMERICA",
    "UNITED STATES OF AMERICA": "UNITED STATES OF AMERICA",
    "D<<": "GERMANY",
    "DEU": "GERMANY",
    "GERMANY": "GERMANY",
    "GRC": "GREECE",
    "GREECE": "GREECE",
    "HELLENIC": "GREECE",
    "TUR": "TURKEY",
    "TURKEY": "TURKEY",
    "IND": "INDIA",
    "INDIA": "INDIA",
    "INDIAN": "INDIA",
    "CHN": "CHINA",
    "CHINA": "CHINA",
    "CHINESE": "CHINA",
}


def slugify_repo_name(repo_id: str) -> str:
    slug = repo_id.replace("/", "__")
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", slug)
    return slug.strip("-")


def normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).replace("\r", "\n").strip()
    if not text:
        return None
    text = re.sub(r"\s+", " ", text.replace("\n", " "))
    return text.strip() or None


def normalize_sex(value: Any) -> str | None:
    text = (normalize_text(value) or "").upper()
    if not text:
        return None
    if text.startswith("M") or "/M" in text:
        return "M"
    if text.startswith("F") or "/F" in text:
        return "F"
    return text[:1] if text else None


def normalize_name(value: Any) -> str | None:
    text = normalize_text(value)
    if not text:
        return None
    if " / " in text:
        parts = [part.strip() for part in text.split(" / ") if part.strip()]
        if len(parts) >= 2:
            return parts[-1]
    text = text.replace(",", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_mrz(value: Any) -> str | None:
    text = normalize_text(value)
    if not text:
        return None
    text = text.replace("\\r", "\n").replace("\\n", "\n")
    text = text.replace("\r", "\n")
    if "\n" not in text and len(text) > 44:
        first = text[:44]
        second = text[44:]
        if second:
            text = first + "\n" + second
    return text.strip()


def normalize_nationality(value: Any, fallback_code: Any = None) -> str | None:
    text = normalize_text(value)
    code = normalize_text(fallback_code)
    if text:
        mapped = NATIONALITY_MAP.get(text.upper())
        if mapped:
            return mapped
        return text.upper()
    if code:
        mapped = NATIONALITY_MAP.get(code.upper())
        if mapped:
            return mapped
        return code.upper()
    return None


def read_text_with_fallbacks(path: Path, encodings: list[str]) -> str:
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding)
        except Exception as exc:  # pragma: no cover - best effort decoding
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Unable to decode {path}")


def infer_gender_bucket(row: dict[str, Any], fallback_index: int) -> str:
    gender = (normalize_text(row.get("gender")) or "").lower()
    if gender in {"male", "female"}:
        return gender
    sex = (normalize_text(row.get("sex")) or "").upper()
    if sex.startswith("M") or "/M" in sex:
        return "male"
    if sex.startswith("F") or "/F" in sex:
        return "female"
    return "male" if fallback_index == 0 else "female"


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


def map_passport_fields(raw: dict[str, Any]) -> dict[str, Any]:
    fields = empty_passport_fields()
    fields["document_number"] = normalize_text(raw.get("pass_num") or raw.get("passport_number"))
    fields["surname"] = normalize_name(raw.get("surname"))
    fields["given_names"] = normalize_name(raw.get("given_name") or raw.get("given_names"))
    fields["nationality"] = normalize_nationality(raw.get("nationality"), raw.get("nationality_code"))
    fields["date_of_birth"] = normalize_text(raw.get("date_of_birth"))
    fields["sex"] = normalize_sex(raw.get("sex"))
    fields["place_of_birth"] = normalize_text(raw.get("place_of_birth"))
    fields["date_of_issue"] = normalize_text(raw.get("date_of_issue"))
    fields["date_of_expiry"] = normalize_text(raw.get("date_of_expiry"))
    fields["issuing_authority"] = normalize_text(raw.get("authority"))
    fields["mrz"] = normalize_mrz(raw.get("mrz"))
    return fields


def parse_csv_metadata(path: Path) -> list[dict[str, Any]]:
    text = read_text_with_fallbacks(path, ["utf-8-sig", "cp1251", "utf-8"])
    delimiter = ";" if ";" in (text.splitlines()[0] if text.splitlines() else "") else ","
    reader = csv.DictReader(text.splitlines(), delimiter=delimiter)
    rows: list[dict[str, Any]] = []
    for row in reader:
        if not any((str(value).strip() for value in row.values() if value is not None)):
            continue
        rows.append(dict(row))
    return rows


def parse_json_metadata(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        for gender_key in ("male", "female"):
            for row in payload.get(gender_key, []) or []:
                merged = dict(row)
                merged["gender"] = gender_key
                rows.append(merged)
    elif isinstance(payload, list):
        rows.extend(payload)
    return rows


def discover_metadata_file(snapshot_dir: Path, repo_files: list[str]) -> Path:
    metadata_candidates = [Path(name) for name in repo_files if name.startswith("metadata/")]
    if not metadata_candidates:
        raise FileNotFoundError(f"No metadata file found in {snapshot_dir}")
    metadata_candidates.sort()
    return snapshot_dir / metadata_candidates[0]


def discover_image_files(snapshot_dir: Path) -> list[Path]:
    images = sorted(snapshot_dir.rglob("*.jpg")) + sorted(snapshot_dir.rglob("*.png"))
    return [path for path in images if "metadata" not in path.parts]


def group_images_by_gender(images: list[Path]) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {"male": [], "female": [], "other": []}
    for image_path in images:
        name = image_path.name.lower()
        if "female_" in name:
            grouped["female"].append(image_path)
        elif "male_" in name:
            grouped["male"].append(image_path)
        else:
            grouped["other"].append(image_path)
    for key in grouped:
        grouped[key] = sorted(grouped[key])
    return grouped


def export_repo(repo_id: str) -> list[dict[str, Any]]:
    repo_files = list_repo_files(repo_id=repo_id, repo_type="dataset")
    snapshot_dir = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=["README.md", "metadata/*", "passport/*", "passport/**/*", "files/*"],
        )
    )

    metadata_path = discover_metadata_file(snapshot_dir, repo_files)
    if metadata_path.suffix.lower() == ".json":
        raw_rows = parse_json_metadata(metadata_path)
    else:
        raw_rows = parse_csv_metadata(metadata_path)

    images = discover_image_files(snapshot_dir)
    image_groups = group_images_by_gender(images)
    image_pointers = {"male": 0, "female": 0, "other": 0}

    dataset_slug = slugify_repo_name(repo_id)
    output_dir = EXTERNAL_ROOT / dataset_slug
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for index, raw_row in enumerate(raw_rows):
        bucket = infer_gender_bucket(raw_row, fallback_index=index % 2)
        bucket_images = image_groups.get(bucket) or image_groups["other"]
        pointer = image_pointers.get(bucket, 0)
        if pointer >= len(bucket_images):
            bucket_images = image_groups["other"] or images
            pointer = image_pointers.get("other", 0)
            bucket = "other"
        if pointer >= len(bucket_images):
            break

        source_image = bucket_images[pointer]
        image_pointers[bucket] = pointer + 1

        destination = output_dir / source_image.name
        if not destination.exists():
            shutil.copy2(source_image, destination)

        row = {
            "id": f"{dataset_slug}_{index:03d}",
            "image_path": str(destination).replace("\\", "/"),
            "document_type": "passport",
            "fields": map_passport_fields(raw_row),
            "source": "hf_synthetic_passports_v1",
            "source_dataset": repo_id,
            "source_metadata_file": metadata_path.name,
            "is_synthetic": True,
        }
        rows.append(row)

    return rows


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def main() -> None:
    all_rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}

    for repo_id in DATASET_REPOS:
        try:
            rows = export_repo(repo_id)
            all_rows.extend(rows)
            counts[repo_id] = len(rows)
        except Exception as exc:
            counts[repo_id] = -1
            print(f"repo_failed={repo_id} error={exc}")

    save_jsonl(OUTPUT_JSONL, all_rows)
    print(OUTPUT_JSONL)
    print(f"total_rows={len(all_rows)}")
    for repo_id, count in counts.items():
        print(f"{repo_id}={count}")


if __name__ == "__main__":
    main()
