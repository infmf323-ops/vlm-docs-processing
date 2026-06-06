from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path("E:/thesis")
QUEUE_PATH = ROOT / "data" / "multidoc" / "annotation_queue.jsonl"
TRAIN_PATH = ROOT / "data" / "multidoc" / "train.jsonl"
VAL_PATH = ROOT / "data" / "multidoc" / "val.jsonl"


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


def strip_queue_only_fields(row: dict) -> dict:
    clean = dict(row)
    for key in [
        "status",
        "heuristic_engine",
        "heuristic_valid",
        "heuristic_elapsed_ms",
        "heuristic_raw_preview",
        "heuristic_issues",
        "draft_source",
        "source_label",
    ]:
        clean.pop(key, None)
    return clean


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs="+", required=True, help="Queue record ids to promote")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    args = parser.parse_args()

    queue_rows = load_jsonl(QUEUE_PATH)
    target_path = TRAIN_PATH if args.split == "train" else VAL_PATH
    target_rows = load_jsonl(target_path)
    target_ids = {row.get("id") for row in target_rows}
    requested_ids = set(args.ids)

    promoted = 0
    for row in queue_rows:
        row_id = row.get("id")
        if row_id not in requested_ids:
            continue
        if row_id in target_ids:
            continue
        target_rows.append(strip_queue_only_fields(row))
        target_ids.add(row_id)
        promoted += 1

    save_jsonl(target_path, target_rows)
    print(target_path)
    print(f"promoted={promoted}")


if __name__ == "__main__":
    main()
