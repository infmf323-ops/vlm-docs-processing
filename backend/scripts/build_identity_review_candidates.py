from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("E:/thesis")
QUEUE_PATH = ROOT / "data" / "multidoc" / "annotation_queue.jsonl"
OUTPUT_JSONL = ROOT / "data" / "multidoc" / "identity_review_candidates.jsonl"
OUTPUT_MD = ROOT / "IDENTITY_REVIEW_CANDIDATES.md"


def count_populated(value) -> int:
    if isinstance(value, dict):
        return sum(count_populated(item) for item in value.values())
    if isinstance(value, list):
        return sum(count_populated(item) for item in value)
    return 1 if value is not None and str(value).strip() else 0


def main() -> None:
    rows = [
        json.loads(line)
        for line in QUEUE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    candidates = []
    for row in rows:
        if row.get("status") != "draft_from_heuristic_pipeline":
            continue
        fields = row.get("fields") or {}
        candidates.append(
            {
                "id": row.get("id"),
                "document_type": row.get("document_type"),
                "image_path": row.get("image_path"),
                "source_dataset": row.get("source_dataset"),
                "draft_source": row.get("draft_source"),
                "populated_field_count": count_populated(fields),
                "fields": fields,
            }
        )

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSONL.open("w", encoding="utf-8") as fp:
        for row in candidates:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = [
        "# Identity Review Candidates",
        "",
        f"Total heuristic candidates: `{len(candidates)}`",
        "",
    ]
    for candidate in candidates:
        lines.extend(
            [
                f"## {candidate['id']}",
                "",
                f"- Document type: `{candidate['document_type']}`",
                f"- Source dataset: `{candidate.get('source_dataset')}`",
                f"- Populated fields: `{candidate['populated_field_count']}`",
                f"- Image: `{candidate['image_path']}`",
                "",
                "```json",
                json.dumps(candidate["fields"], ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(OUTPUT_JSONL)
    print(OUTPUT_MD)
    print(f"candidates={len(candidates)}")


if __name__ == "__main__":
    main()
