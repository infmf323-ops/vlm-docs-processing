from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "artifacts" / "kaggle_bundle"
ZIP_PATH = ROOT / "artifacts" / "kaggle_bundle.zip"

REQUIRED_JSONL = [
    "data/multidoc/passport_curriculum_train_v1.jsonl",
    "data/multidoc/passport_curriculum_val_v1.jsonl",
    "data/multidoc/passport_curriculum_train_v2.jsonl",
    "data/multidoc/passport_curriculum_val_v2.jsonl",
    "data/multidoc/passport_curriculum_train_v3.jsonl",
    "data/multidoc/passport_curriculum_val_v3.jsonl",
    "data/multidoc/passport_hf_synthetic_source_v1.jsonl",
    "data/multidoc/passport_pretrain_hf_train_v1.jsonl",
    "data/multidoc/passport_pretrain_hf_val_v1.jsonl",
    "data/multidoc/passport_russian_finetune_train_v1.jsonl",
    "data/multidoc/passport_russian_finetune_val_v1.jsonl",
    "data/multidoc/passport_pretrain_hf_train_v2.jsonl",
    "data/multidoc/passport_pretrain_hf_val_v2.jsonl",
    "data/multidoc/passport_russian_finetune_train_v2.jsonl",
    "data/multidoc/passport_russian_finetune_val_v2.jsonl",
    "data/multidoc/passport_pretrain_hf_train_v3.jsonl",
    "data/multidoc/passport_pretrain_hf_val_v3.jsonl",
    "data/multidoc/passport_russian_finetune_train_v3.jsonl",
    "data/multidoc/passport_russian_finetune_val_v3.jsonl",
    "data/multidoc/passport_russian_finetune_train_v3_aug.jsonl",
    "data/multidoc/passport_eval_diverse_v1.jsonl",
    "data/multidoc/passport_transfer_holdout_eval_v1.jsonl",
    "data/multidoc/passport_transfer_printed_shift_eval_v1.jsonl",
    "data/multidoc/passport_transfer_crosscountry_eval_v1.jsonl",
    "data/multidoc/driver_license_curriculum_train_v1.jsonl",
    "data/multidoc/driver_license_curriculum_val_v1.jsonl",
    "data/multidoc/driver_license_eval_diverse_v1.jsonl",
    "data/multidoc/pilot_train_strong.jsonl",
    "data/multidoc/pilot_val.jsonl",
    "data/multidoc/identity_eval_v5.jsonl",
    "data/multidoc/identity_eval_diverse_v1.jsonl",
]

REQUIRED_FILES = [
    "data/multidoc/passport_transfer_v1_summary.json",
    "data/multidoc/passport_hf_synthetic_source_v1_profile.json",
    "PASSPORT_TRANSFER_VERSION_COMPARISON.md",
]

REQUIRED_TREES = [
    "backend/app",
    "backend/scripts/train_paddleocr_vl_lora.py",
    "backend/scripts/evaluate_multidoc_engines.py",
    "backend/scripts/import_hf_passport_datasets.py",
    "backend/scripts/build_passport_transfer_splits.py",
    "backend/scripts/prepare_passport_transfer_v1.py",
    "backend/scripts/validate_passport_transfer_v1.py",
    "backend/scripts/run_passport_transfer_experiment.py",
    "backend/scripts/run_passport_mrz_experiment.py",
    "backend/scripts/analyze_passport_eval.py",
    "backend/scripts/profile_passport_transfer_source.py",
    "backend/scripts/compare_passport_transfer_versions.py",
    "backend/scripts/build_passport_transfer_eval_sets.py",
    "backend/scripts/compare_passport_eval_runs.py",
    "backend/scripts/run_passport_eval_suite.py",
    "backend/scripts/augment_russian_passport_data.py",
    "kaggle",
]


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def collect_image_paths(jsonl_paths: list[Path]) -> set[Path]:
    image_paths: set[Path] = set()
    for path in jsonl_paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            image_path = Path(row["image_path"])
            image_paths.add(image_path)
    return image_paths


def copy_tree(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        copy_file(src, dst)


def write_manifest(bundle_root: Path, jsonl_paths: list[Path], image_paths: set[Path]) -> None:
    manifest = {
        "bundle_root": str(bundle_root),
        "jsonl_files": [str(path.relative_to(ROOT)).replace("\\", "/") for path in jsonl_paths],
        "image_count": len(image_paths),
        "image_roots": sorted(
            {
                str(path.relative_to(ROOT).parts[0]).replace("\\", "/")
                for path in image_paths
                if path.is_relative_to(ROOT)
            }
        ),
    }
    (bundle_root / "kaggle_bundle_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    jsonl_paths = [ROOT / rel for rel in REQUIRED_JSONL]
    required_files = [ROOT / rel for rel in REQUIRED_FILES]
    image_paths = collect_image_paths(jsonl_paths)

    for rel in REQUIRED_TREES:
        src = ROOT / rel
        dst = OUTPUT_ROOT / rel
        copy_tree(src, dst)

    for jsonl_path in jsonl_paths:
        copy_file(jsonl_path, OUTPUT_ROOT / jsonl_path.relative_to(ROOT))

    for file_path in required_files:
        copy_file(file_path, OUTPUT_ROOT / file_path.relative_to(ROOT))

    for image_path in sorted(image_paths):
        if image_path.is_relative_to(ROOT):
            copy_file(image_path, OUTPUT_ROOT / image_path.relative_to(ROOT))

    write_manifest(OUTPUT_ROOT, jsonl_paths, image_paths)

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.make_archive(str(ZIP_PATH.with_suffix("")), "zip", root_dir=OUTPUT_ROOT)
    print(ZIP_PATH)


if __name__ == "__main__":
    main()
