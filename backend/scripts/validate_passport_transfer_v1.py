from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path("E:/thesis")
DATA_DIR = ROOT / "data" / "multidoc"

SPLIT_VERSION_PATHS = {
    "v1": {
        "pretrain_train": DATA_DIR / "passport_pretrain_hf_train_v1.jsonl",
        "pretrain_val": DATA_DIR / "passport_pretrain_hf_val_v1.jsonl",
        "russian_finetune_train": DATA_DIR / "passport_russian_finetune_train_v1.jsonl",
        "russian_finetune_val": DATA_DIR / "passport_russian_finetune_val_v1.jsonl",
    },
    "v2": {
        "pretrain_train": DATA_DIR / "passport_pretrain_hf_train_v2.jsonl",
        "pretrain_val": DATA_DIR / "passport_pretrain_hf_val_v2.jsonl",
        "russian_finetune_train": DATA_DIR / "passport_russian_finetune_train_v2.jsonl",
        "russian_finetune_val": DATA_DIR / "passport_russian_finetune_val_v2.jsonl",
    },
    "v3": {
        "pretrain_train": DATA_DIR / "passport_pretrain_hf_train_v3.jsonl",
        "pretrain_val": DATA_DIR / "passport_pretrain_hf_val_v3.jsonl",
        "russian_finetune_train": DATA_DIR / "passport_russian_finetune_train_v3.jsonl",
        "russian_finetune_val": DATA_DIR / "passport_russian_finetune_val_v3.jsonl",
    },
}

SUMMARY_PATH = DATA_DIR / "passport_transfer_v1_summary.json"


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def summarize_rows(rows: list[dict]) -> dict:
    nationality_counter = Counter()
    source_counter = Counter()
    missing_images: list[str] = []
    for row in rows:
        fields = row.get("fields") or {}
        nationality_counter[str(fields.get("nationality"))] += 1
        source_counter[str(row.get("source_dataset"))] += 1
        image_path = Path(str(row.get("image_path", "")))
        if not image_path.exists():
            missing_images.append(str(image_path))

    return {
        "row_count": len(rows),
        "nationalities": dict(sorted(nationality_counter.items())),
        "source_datasets": dict(sorted(source_counter.items())),
        "missing_images": missing_images,
    }


def main() -> None:
    summary = {"versions": {}}
    total_missing = 0
    has_overlap = False

    for version, split_paths in SPLIT_VERSION_PATHS.items():
        split_rows = {name: load_jsonl(path) for name, path in split_paths.items()}
        split_ids = {name: {str(row.get("id")) for row in rows} for name, rows in split_rows.items()}

        pretrain_overlap = sorted(split_ids["pretrain_train"] & split_ids["pretrain_val"])
        russian_overlap = sorted(split_ids["russian_finetune_train"] & split_ids["russian_finetune_val"])
        split_summary = {name: summarize_rows(rows) for name, rows in split_rows.items()}
        total_missing += sum(len(item["missing_images"]) for item in split_summary.values())
        has_overlap = has_overlap or bool(pretrain_overlap or russian_overlap)

        summary["versions"][version] = {
            "splits": split_summary,
            "overlap_checks": {
                "pretrain_train_vs_val": pretrain_overlap,
                "russian_finetune_train_vs_val": russian_overlap,
            },
        }

    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(SUMMARY_PATH)
    for version, version_summary in summary["versions"].items():
        for split_name, split_summary in version_summary["splits"].items():
            print(
                f"{version}:{split_name}: rows={split_summary['row_count']}, "
                f"missing_images={len(split_summary['missing_images'])}"
            )
        print(
            f"{version}:pretrain_train_vs_val_overlap="
            f"{len(version_summary['overlap_checks']['pretrain_train_vs_val'])}"
        )
        print(
            f"{version}:russian_finetune_train_vs_val_overlap="
            f"{len(version_summary['overlap_checks']['russian_finetune_train_vs_val'])}"
        )

    if total_missing:
        raise SystemExit(f"Found {total_missing} missing images across passport transfer splits.")
    if has_overlap:
        raise SystemExit("Found overlapping IDs between train/val splits.")


if __name__ == "__main__":
    main()
