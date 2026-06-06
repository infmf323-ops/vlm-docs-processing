from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("E:/thesis")
DATA_DIR = ROOT / "data" / "multidoc"

PRETRAIN_VAL_V2 = DATA_DIR / "passport_pretrain_hf_val_v2.jsonl"
PRETRAIN_VAL_V3 = DATA_DIR / "passport_pretrain_hf_val_v3.jsonl"
RUS_VAL_V3 = DATA_DIR / "passport_russian_finetune_val_v3.jsonl"

TRANSFER_HOLDOUT_OUT = DATA_DIR / "passport_transfer_holdout_eval_v1.jsonl"
PRINTED_SHIFT_OUT = DATA_DIR / "passport_transfer_printed_shift_eval_v1.jsonl"
CROSSCOUNTRY_OUT = DATA_DIR / "passport_transfer_crosscountry_eval_v1.jsonl"


def load_jsonl(path: Path) -> list[dict]:
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


def is_printed_row(row: dict) -> bool:
    return "printed-" in str(row.get("source_dataset", "")).lower()


def main() -> None:
    pretrain_val_v2 = load_jsonl(PRETRAIN_VAL_V2)
    pretrain_val_v3 = load_jsonl(PRETRAIN_VAL_V3)
    russian_val_v3 = load_jsonl(RUS_VAL_V3)

    transfer_holdout = [*pretrain_val_v3, *russian_val_v3]
    printed_shift = [row for row in pretrain_val_v2 if is_printed_row(row)]
    crosscountry = [*pretrain_val_v2, *russian_val_v3]

    save_jsonl(TRANSFER_HOLDOUT_OUT, transfer_holdout)
    save_jsonl(PRINTED_SHIFT_OUT, printed_shift)
    save_jsonl(CROSSCOUNTRY_OUT, crosscountry)

    print(TRANSFER_HOLDOUT_OUT)
    print(PRINTED_SHIFT_OUT)
    print(CROSSCOUNTRY_OUT)
    print(f"transfer_holdout_rows={len(transfer_holdout)}")
    print(f"printed_shift_rows={len(printed_shift)}")
    print(f"crosscountry_rows={len(crosscountry)}")


if __name__ == "__main__":
    main()
