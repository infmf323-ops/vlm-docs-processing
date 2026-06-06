from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("E:/thesis")
QUEUE_PATH = ROOT / "data" / "multidoc" / "annotation_queue.jsonl"
TRAIN_PATH = ROOT / "data" / "multidoc" / "train.jsonl"

TRAIN_CANDIDATES_JSONL = ROOT / "data" / "multidoc" / "identity_curated_train_candidates.jsonl"
EVAL_CANDIDATES_JSONL = ROOT / "data" / "multidoc" / "identity_holdout_eval_candidates.jsonl"
MANUAL_REVIEW_JSONL = ROOT / "data" / "multidoc" / "identity_manual_review_candidates.jsonl"
SUMMARY_MD = ROOT / "IDENTITY_CURATION_PLAN.md"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def count_populated(value: Any) -> int:
    if isinstance(value, dict):
        return sum(count_populated(item) for item in value.values())
    if isinstance(value, list):
        return sum(count_populated(item) for item in value)
    return 1 if value is not None and str(value).strip() else 0


def looks_suspicious(text: str | None) -> bool:
    if not text:
        return False
    lowered = text.lower()
    suspicious_fragments = [
        "signature",
        "holder",
        "http://",
        "https://",
        "issuing authority",
        "the following is a sample",
        "class",
        "end none",
    ]
    return any(fragment in lowered for fragment in suspicious_fragments)


def issue_count(fields: dict[str, Any]) -> int:
    issues = 0
    for value in fields.values():
        if isinstance(value, list):
            continue
        if looks_suspicious(str(value) if value is not None else None):
            issues += 1
    return issues


def source_priority(source_dataset: str) -> int:
    if "synthetic-russian-passports" in source_dataset:
        return 0
    if "synthetic-french-passports" in source_dataset:
        return 1
    if "passport-dataset" in source_dataset:
        return 2
    if "synthetic-japanese-passports" in source_dataset:
        return 3
    if "synthetic-printed-usa-passports" in source_dataset:
        return 4
    if "synthetic-usa-driver-license" in source_dataset:
        return 5
    if "synthetic-printed-german-passports" in source_dataset:
        return 6
    return 99


def main() -> None:
    queue_rows = load_jsonl(QUEUE_PATH)
    train_rows = load_jsonl(TRAIN_PATH)
    train_ids = {str(row.get("id")) for row in train_rows}

    train_candidates: list[dict[str, Any]] = []
    eval_candidates: list[dict[str, Any]] = []
    manual_review: list[dict[str, Any]] = []

    for row in queue_rows:
        source_dataset = str(row.get("source_dataset") or "")
        document_type = str(row.get("document_type") or "")
        status = str(row.get("status") or "")
        row_id = str(row.get("id") or "")
        fields = row.get("fields") or {}
        populated = count_populated(fields)
        suspicious = issue_count(fields)

        if document_type not in {"passport", "driver_license", "id_card"}:
            continue
        if not source_dataset:
            continue
        if row_id in train_ids:
            continue

        candidate = {
            "id": row_id,
            "document_type": document_type,
            "source_dataset": source_dataset,
            "image_path": row.get("image_path"),
            "status": status,
            "heuristic_valid": row.get("heuristic_valid"),
            "populated_field_count": populated,
            "suspicious_field_count": suspicious,
            "fields": fields,
        }

        is_diverse_passport = (
            document_type == "passport"
            and source_dataset.startswith("ud-synthetic/")
        )
        is_good_heuristic = status == "draft_from_heuristic_pipeline" and populated >= 5 and suspicious <= 2

        if is_good_heuristic and source_dataset in {
            "ud-synthetic/synthetic-russian-passports",
            "ud-synthetic/synthetic-french-passports",
        }:
            train_candidates.append(candidate)
        elif is_diverse_passport and populated >= 4:
            eval_candidates.append(candidate)
        else:
            manual_review.append(candidate)

    train_candidates.sort(key=lambda row: (source_priority(row["source_dataset"]), -row["populated_field_count"], row["id"]))
    eval_candidates.sort(key=lambda row: (source_priority(row["source_dataset"]), -row["populated_field_count"], row["id"]))
    manual_review.sort(key=lambda row: (source_priority(row["source_dataset"]), -row["populated_field_count"], row["id"]))

    for path, rows in [
        (TRAIN_CANDIDATES_JSONL, train_candidates),
        (EVAL_CANDIDATES_JSONL, eval_candidates),
        (MANUAL_REVIEW_JSONL, manual_review),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
            encoding="utf-8",
        )

    lines = [
        "# Identity Curation Plan",
        "",
        f"Train candidates: `{len(train_candidates)}`",
        f"Eval candidates: `{len(eval_candidates)}`",
        f"Manual-review candidates: `{len(manual_review)}`",
        "",
        "## Train Candidates",
        "",
    ]

    for row in train_candidates:
        lines.extend(
            [
                f"- `{row['id']}` | `{row['source_dataset']}` | populated=`{row['populated_field_count']}` | suspicious=`{row['suspicious_field_count']}`",
            ]
        )

    lines.extend(["", "## Hold-out Eval Candidates", ""])
    for row in eval_candidates:
        lines.extend(
            [
                f"- `{row['id']}` | `{row['source_dataset']}` | populated=`{row['populated_field_count']}` | suspicious=`{row['suspicious_field_count']}`",
            ]
        )

    lines.extend(["", "## Manual Review", ""])
    for row in manual_review[:25]:
        lines.extend(
            [
                f"- `{row['id']}` | `{row['source_dataset']}` | status=`{row['status']}` | populated=`{row['populated_field_count']}` | suspicious=`{row['suspicious_field_count']}`",
            ]
        )

    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")

    print(TRAIN_CANDIDATES_JSONL)
    print(EVAL_CANDIDATES_JSONL)
    print(MANUAL_REVIEW_JSONL)
    print(SUMMARY_MD)
    print(f"train_candidates={len(train_candidates)}")
    print(f"eval_candidates={len(eval_candidates)}")
    print(f"manual_review={len(manual_review)}")


if __name__ == "__main__":
    main()
