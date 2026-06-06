from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("E:/thesis/data/multidoc")
TRAIN_FILE = ROOT / "train.jsonl"
VAL_FILE = ROOT / "val.jsonl"
PILOT_TRAIN = ROOT / "pilot_train.jsonl"
PILOT_VAL = ROOT / "pilot_val.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def count_populated(value) -> int:
    if value is None:
        return 0
    if isinstance(value, dict):
        return sum(count_populated(v) for v in value.values())
    if isinstance(value, list):
        return sum(count_populated(v) for v in value)
    return 1 if str(value).strip() else 0


def is_real_record(row: dict) -> bool:
    if row.get("bootstrap_template") is True:
        return False
    if not row.get("image_path") or not Path(row["image_path"]).exists():
        return False
    fields = row.get("fields")
    if not isinstance(fields, dict):
        return False
    return count_populated(fields) >= 4


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    train_rows = [row for row in load_jsonl(TRAIN_FILE) if is_real_record(row)]
    val_rows = [row for row in load_jsonl(VAL_FILE) if is_real_record(row)]

    if not val_rows and len(train_rows) > 1:
        val_rows = [train_rows.pop()]

    write_jsonl(PILOT_TRAIN, train_rows)
    write_jsonl(PILOT_VAL, val_rows)

    print(PILOT_TRAIN)
    print(PILOT_VAL)
    print(f"pilot_train={len(train_rows)}")
    print(f"pilot_val={len(val_rows)}")


if __name__ == "__main__":
    main()
