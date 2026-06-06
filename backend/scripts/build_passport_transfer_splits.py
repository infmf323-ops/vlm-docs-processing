from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("E:/thesis")
DATA_DIR = ROOT / "data" / "multidoc"

HF_SOURCE_PATH = DATA_DIR / "passport_hf_synthetic_source_v1.jsonl"
PASSPORT_V3_PATH = DATA_DIR / "passport_curriculum_train_v3.jsonl"

RUS_FINETUNE_VAL_ID = "ud-synthetic__synthetic-russian-passports_004"
RUS_SOURCE_TOKEN = "synthetic-russian-passports"

SPLIT_OUTPUTS = {
    "v1": {
        "pretrain_train": DATA_DIR / "passport_pretrain_hf_train_v1.jsonl",
        "pretrain_val": DATA_DIR / "passport_pretrain_hf_val_v1.jsonl",
        "russian_train": DATA_DIR / "passport_russian_finetune_train_v1.jsonl",
        "russian_val": DATA_DIR / "passport_russian_finetune_val_v1.jsonl",
    },
    "v2": {
        "pretrain_train": DATA_DIR / "passport_pretrain_hf_train_v2.jsonl",
        "pretrain_val": DATA_DIR / "passport_pretrain_hf_val_v2.jsonl",
        "russian_train": DATA_DIR / "passport_russian_finetune_train_v2.jsonl",
        "russian_val": DATA_DIR / "passport_russian_finetune_val_v2.jsonl",
    },
    "v3": {
        "pretrain_train": DATA_DIR / "passport_pretrain_hf_train_v3.jsonl",
        "pretrain_val": DATA_DIR / "passport_pretrain_hf_val_v3.jsonl",
        "russian_train": DATA_DIR / "passport_russian_finetune_train_v3.jsonl",
        "russian_val": DATA_DIR / "passport_russian_finetune_val_v3.jsonl",
    },
}


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


def is_russian_row(row: dict) -> bool:
    return RUS_SOURCE_TOKEN in str(row.get("source_dataset", "")).lower() or str(
        (row.get("fields") or {}).get("nationality", "")
    ).upper() == "RUSSIAN FEDERATION"


def is_printed_variant(row: dict) -> bool:
    return "printed-" in str(row.get("source_dataset", "")).lower()


def build_grouped_train_val(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("source_dataset")), []).append(row)

    train_rows: list[dict] = []
    val_rows: list[dict] = []
    for _, dataset_rows in sorted(grouped.items()):
        dataset_rows = sorted(dataset_rows, key=lambda item: str(item.get("id")))
        if len(dataset_rows) >= 2:
            val_rows.append(dataset_rows[-1])
            train_rows.extend(dataset_rows[:-1])
        elif dataset_rows:
            train_rows.extend(dataset_rows)
    return train_rows, val_rows


def build_russian_finetune_splits(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    russian_rows = [row for row in rows if is_russian_row(row)]

    train_rows: list[dict] = []
    val_rows: list[dict] = []
    for row in sorted(russian_rows, key=lambda item: str(item.get("id"))):
        if row.get("id") == RUS_FINETUNE_VAL_ID:
            val_rows.append(row)
        else:
            train_rows.append(row)
    return train_rows, val_rows


def build_split_versions(hf_rows: list[dict], v3_rows: list[dict]) -> dict[str, dict[str, list[dict]]]:
    v1_pretrain_train, v1_pretrain_val = build_grouped_train_val(hf_rows)
    v2_source_rows = [row for row in hf_rows if not is_russian_row(row)]
    v2_pretrain_train, v2_pretrain_val = build_grouped_train_val(v2_source_rows)
    v3_source_rows = [row for row in hf_rows if not is_russian_row(row) and not is_printed_variant(row)]
    v3_pretrain_train, v3_pretrain_val = build_grouped_train_val(v3_source_rows)

    russian_train, russian_val = build_russian_finetune_splits(v3_rows)

    return {
        "v1": {
            "pretrain_train": v1_pretrain_train,
            "pretrain_val": v1_pretrain_val,
            "russian_train": russian_train,
            "russian_val": russian_val,
        },
        "v2": {
            "pretrain_train": v2_pretrain_train,
            "pretrain_val": v2_pretrain_val,
            "russian_train": russian_train,
            "russian_val": russian_val,
        },
        "v3": {
            "pretrain_train": v3_pretrain_train,
            "pretrain_val": v3_pretrain_val,
            "russian_train": russian_train,
            "russian_val": russian_val,
        },
    }


def main() -> None:
    hf_rows = load_jsonl(HF_SOURCE_PATH)
    v3_rows = load_jsonl(PASSPORT_V3_PATH)
    split_versions = build_split_versions(hf_rows, v3_rows)

    for version, outputs in SPLIT_OUTPUTS.items():
        rows = split_versions[version]
        save_jsonl(outputs["pretrain_train"], rows["pretrain_train"])
        save_jsonl(outputs["pretrain_val"], rows["pretrain_val"])
        save_jsonl(outputs["russian_train"], rows["russian_train"])
        save_jsonl(outputs["russian_val"], rows["russian_val"])

        print(outputs["pretrain_train"])
        print(outputs["pretrain_val"])
        print(outputs["russian_train"])
        print(outputs["russian_val"])
        print(f"{version}_pretrain_train_rows={len(rows['pretrain_train'])}")
        print(f"{version}_pretrain_val_rows={len(rows['pretrain_val'])}")
        print(f"{version}_russian_finetune_train_rows={len(rows['russian_train'])}")
        print(f"{version}_russian_finetune_val_rows={len(rows['russian_val'])}")


if __name__ == "__main__":
    main()
