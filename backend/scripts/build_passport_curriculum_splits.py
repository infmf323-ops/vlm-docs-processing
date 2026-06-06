from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("E:/thesis")
DATA_DIR = ROOT / "data" / "multidoc"

TRAIN_PATH = DATA_DIR / "train.jsonl"
IDENTITY_DIVERSE_EVAL_PATH = DATA_DIR / "identity_eval_diverse_v1.jsonl"

PASSPORT_TRAIN_OUT = DATA_DIR / "passport_curriculum_train_v1.jsonl"
PASSPORT_VAL_OUT = DATA_DIR / "passport_curriculum_val_v1.jsonl"
PASSPORT_EVAL_OUT = DATA_DIR / "passport_eval_diverse_v1.jsonl"

VAL_ID = "passport_michelle_obama"


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


def main() -> None:
    train_rows = load_jsonl(TRAIN_PATH)
    diverse_eval_rows = load_jsonl(IDENTITY_DIVERSE_EVAL_PATH)

    passport_rows = [row for row in train_rows if row.get("document_type") == "passport"]
    train_out: list[dict] = []
    val_out: list[dict] = []

    for row in passport_rows:
        if row.get("id") == VAL_ID:
            val_out.append(row)
        else:
            train_out.append(row)

    passport_eval = [row for row in diverse_eval_rows if row.get("document_type") == "passport"]

    save_jsonl(PASSPORT_TRAIN_OUT, train_out)
    save_jsonl(PASSPORT_VAL_OUT, val_out)
    save_jsonl(PASSPORT_EVAL_OUT, passport_eval)

    print(PASSPORT_TRAIN_OUT)
    print(PASSPORT_VAL_OUT)
    print(PASSPORT_EVAL_OUT)
    print(f"train_rows={len(train_out)}")
    print(f"val_rows={len(val_out)}")
    print(f"eval_rows={len(passport_eval)}")


if __name__ == "__main__":
    main()
