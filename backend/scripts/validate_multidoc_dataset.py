from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        rows.append((line_number, json.loads(line)))
    return rows


def count_populated(value) -> int:
    if value is None:
        return 0
    if isinstance(value, dict):
        return sum(count_populated(v) for v in value.values())
    if isinstance(value, list):
        return sum(count_populated(v) for v in value)
    return 1 if str(value).strip() else 0


def validate_rows(path: Path) -> list[str]:
    issues: list[str] = []
    seen_ids: set[str] = set()

    for line_number, row in load_jsonl(path):
        row_id = row.get("id")
        image_path = row.get("image_path")
        document_type = row.get("document_type")
        fields = row.get("fields")

        if not row_id:
            issues.append(f"{path.name}:{line_number}: missing id")
        elif row_id in seen_ids:
            issues.append(f"{path.name}:{line_number}: duplicate id '{row_id}'")
        else:
            seen_ids.add(row_id)

        if not image_path:
            issues.append(f"{path.name}:{line_number}: missing image_path")
        elif not Path(image_path).exists():
            issues.append(f"{path.name}:{line_number}: image file does not exist -> {image_path}")

        if not document_type:
            issues.append(f"{path.name}:{line_number}: missing document_type")

        if not isinstance(fields, dict):
            issues.append(f"{path.name}:{line_number}: fields must be an object")
            continue

        populated = count_populated(fields)
        if populated == 0:
            issues.append(f"{path.name}:{line_number}: all fields are empty")

        if row.get("bootstrap_template") is True:
            issues.append(f"{path.name}:{line_number}: bootstrap_template record requires manual review")

    return issues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate multi-document JSONL dataset files.")
    parser.add_argument(
        "--paths",
        nargs="+",
        default=[
            "E:/thesis/data/multidoc/train.jsonl",
            "E:/thesis/data/multidoc/val.jsonl",
        ],
        help="JSONL dataset files to validate.",
    )
    parser.add_argument(
        "--include-templates",
        action="store_true",
        help="Also validate templates.jsonl. Useful for checking placeholder examples separately.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_issues: list[str] = []
    raw_paths = list(args.paths)
    if args.include_templates:
        raw_paths.append("E:/thesis/data/multidoc/templates.jsonl")

    for raw_path in raw_paths:
        path = Path(raw_path)
        if not path.exists():
            all_issues.append(f"{path}: file not found")
            continue
        all_issues.extend(validate_rows(path))

    if all_issues:
        print("DATASET_ISSUES")
        for issue in all_issues:
            print(issue)
    else:
        print("DATASET_OK")


if __name__ == "__main__":
    main()
