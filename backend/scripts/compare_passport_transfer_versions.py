from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("E:/thesis")
DATA_DIR = ROOT / "data" / "multidoc"
SUMMARY_PATH = DATA_DIR / "passport_transfer_v1_summary.json"
PROFILE_PATH = DATA_DIR / "passport_hf_synthetic_source_v1_profile.json"
REPORT_PATH = ROOT / "PASSPORT_TRANSFER_VERSION_COMPARISON.md"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    summary = load_json(SUMMARY_PATH)
    profile = load_json(PROFILE_PATH)

    versions = summary["versions"]
    source_profiles = profile["sources"]

    lines: list[str] = []
    lines.append("# Passport Transfer Version Comparison")
    lines.append("")
    lines.append("Date fixed: 2026-05-21")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("This note compares the current passport transfer-learning split families `v1`, `v2`, and `v3`.")
    lines.append("")
    lines.append("## Version summary")
    lines.append("")
    lines.append("| Version | Pretrain idea | Pretrain train | Pretrain val | Russian finetune train | Russian finetune val |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    descriptions = {
        "v1": "All synthetic passport sources, including Russian and printed variants",
        "v2": "All non-Russian synthetic passport sources, including printed variants",
        "v3": "Only non-Russian, non-printed synthetic passport sources",
    }

    for version in ("v1", "v2", "v3"):
        version_summary = versions[version]["splits"]
        lines.append(
            f"| `{version}` | {descriptions[version]} | "
            f"{version_summary['pretrain_train']['row_count']} | "
            f"{version_summary['pretrain_val']['row_count']} | "
            f"{version_summary['russian_finetune_train']['row_count']} | "
            f"{version_summary['russian_finetune_val']['row_count']} |"
        )

    lines.append("")
    lines.append("## Pretrain source composition")
    lines.append("")
    for version in ("v1", "v2", "v3"):
        lines.append(f"### {version}")
        lines.append("")
        pretrain_sources = versions[version]["splits"]["pretrain_train"]["source_datasets"]
        for source_name, row_count in sorted(pretrain_sources.items()):
            flags = source_profiles[source_name]["flags"]
            flag_bits = []
            if flags["is_russian"]:
                flag_bits.append("russian")
            if flags["is_printed_variant"]:
                flag_bits.append("printed")
            flag_text = f" ({', '.join(flag_bits)})" if flag_bits else ""
            lines.append(f"- `{source_name}`: `{row_count}` rows{flag_text}")
        lines.append("")

    lines.append("## Recommendation")
    lines.append("")
    lines.append("- `v3` is the cleanest mainline baseline for the next transfer experiment.")
    lines.append("- `v2` is the wider non-Russian baseline if we want more source diversity and are willing to keep printed variants.")
    lines.append("- `v1` should now be treated mostly as an ablation baseline because it leaks Russian structure into the broad pretrain stage.")
    lines.append("")
    lines.append("## Practical meaning")
    lines.append("")
    lines.append("- If `v3` beats `v2`, then printed variants were likely hurting more than helping.")
    lines.append("- If `v2` beats `v3`, then printed variants may still provide useful passport-layout diversity.")
    lines.append("- If `v1` beats both, then Russian overlap in the pretrain stage is doing a lot of work and the adaptation story is weaker than we want.")
    lines.append("")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
