from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("E:/thesis")
DATA_DIR = ROOT / "data" / "multidoc"

TRAIN_PATH = DATA_DIR / "train.jsonl"
IDENTITY_DIVERSE_EVAL_PATH = DATA_DIR / "identity_eval_diverse_v1.jsonl"

DL_TRAIN_OUT = DATA_DIR / "driver_license_curriculum_train_v1.jsonl"
DL_VAL_OUT = DATA_DIR / "driver_license_curriculum_val_v1.jsonl"
DL_EVAL_OUT = DATA_DIR / "driver_license_eval_diverse_v1.jsonl"

VAL_ID = "synthetic-usa-driver-license_000"


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

    dl_rows = [row for row in train_rows if row.get("document_type") == "driver_license"]
    train_out: list[dict] = []
    val_out: list[dict] = []

    for row in dl_rows:
        if row.get("id") == VAL_ID:
            val_out.append(row)
        else:
            train_out.append(row)

    dl_eval = [row for row in diverse_eval_rows if row.get("document_type") == "driver_license"]

    save_jsonl(DL_TRAIN_OUT, train_out)
    save_jsonl(DL_VAL_OUT, val_out)
    save_jsonl(DL_EVAL_OUT, dl_eval)

    print(DL_TRAIN_OUT)
    print(DL_VAL_OUT)
    print(DL_EVAL_OUT)
    print(f"train_rows={len(train_out)}")
    print(f"val_rows={len(val_out)}")
    print(f"eval_rows={len(dl_eval)}")


if __name__ == "__main__":
    main()
