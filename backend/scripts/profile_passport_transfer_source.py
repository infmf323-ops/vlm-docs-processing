from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("E:/thesis")
DATA_DIR = ROOT / "data" / "multidoc"
SOURCE_PATH = DATA_DIR / "passport_hf_synthetic_source_v1.jsonl"
OUTPUT_PATH = DATA_DIR / "passport_hf_synthetic_source_v1_profile.json"


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def source_flags(source_dataset: str) -> dict[str, bool]:
    lower = source_dataset.lower()
    return {
        "is_russian": "russian" in lower,
        "is_printed_variant": "printed-" in lower,
    }


def main() -> None:
    rows = load_jsonl(SOURCE_PATH)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("source_dataset"))].append(row)

    profile = {"total_rows": len(rows), "sources": {}}

    for source_dataset, source_rows in sorted(grouped.items()):
        nationality_counter = Counter()
        field_coverage = Counter()
        sex_counter = Counter()
        image_ext_counter = Counter()

        for row in source_rows:
            fields = row.get("fields") or {}
            nationality_counter[str(fields.get("nationality"))] += 1
            sex_counter[str(fields.get("sex"))] += 1
            image_ext_counter[Path(str(row.get("image_path", ""))).suffix.lower()] += 1
            for key, value in fields.items():
                if value not in (None, ""):
                    field_coverage[key] += 1

        profile["sources"][source_dataset] = {
            "row_count": len(source_rows),
            "flags": source_flags(source_dataset),
            "nationalities": dict(sorted(nationality_counter.items())),
            "sexes": dict(sorted(sex_counter.items())),
            "image_extensions": dict(sorted(image_ext_counter.items())),
            "field_coverage": dict(sorted(field_coverage.items())),
            "sample_ids": [str(row.get("id")) for row in sorted(source_rows, key=lambda item: str(item.get("id")))],
        }

    OUTPUT_PATH.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUTPUT_PATH)
    print(f"total_rows={profile['total_rows']}")
    for source_dataset, source_profile in profile["sources"].items():
        print(
            f"{source_dataset}: rows={source_profile['row_count']}, "
            f"printed={source_profile['flags']['is_printed_variant']}, "
            f"russian={source_profile['flags']['is_russian']}"
        )


if __name__ == "__main__":
    main()
