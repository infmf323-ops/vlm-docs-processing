from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HF_CACHE = ROOT / ".hf-cache"
DATA_DIR = ROOT / "data" / "multidoc"

EVAL_SETS = {
    "legacy_diverse": DATA_DIR / "passport_eval_diverse_v1.jsonl",
    "transfer_holdout": DATA_DIR / "passport_transfer_holdout_eval_v1.jsonl",
    "printed_shift": DATA_DIR / "passport_transfer_printed_shift_eval_v1.jsonl",
    "crosscountry": DATA_DIR / "passport_transfer_crosscountry_eval_v1.jsonl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a passport adapter across multiple eval sets.")
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["flat", "mrz"], default="flat")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "mrz":
        eval_script = ROOT / "kaggle" / "eval_passport_mrz_on_kaggle.py"
    else:
        eval_script = ROOT / "kaggle" / "eval_passport_flat_on_kaggle.py"

    for dataset_name, dataset_path in EVAL_SETS.items():
        output_path = args.output_dir / f"{dataset_name}_{args.mode}.json"
        env = os.environ.copy()
        env["HF_HOME"] = str(HF_CACHE)
        env["TRANSFORMERS_CACHE"] = str(HF_CACHE / "transformers")
        env["HF_HUB_CACHE"] = str(HF_CACHE / "hub")
        env["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE / "hub")
        env["PASSPORT_ADAPTER_DIR"] = str(args.adapter_dir)
        env["PASSPORT_EVAL_DATASET"] = str(dataset_path)
        env["PASSPORT_EVAL_OUTPUT"] = str(output_path)

        command = [sys.executable, str(eval_script)]
        print(f"[{dataset_name}] Running:", " ".join(command))
        subprocess.run(command, cwd=str(ROOT), env=env, check=True)

    print("Saved suite outputs to:", args.output_dir)


if __name__ == "__main__":
    main()
