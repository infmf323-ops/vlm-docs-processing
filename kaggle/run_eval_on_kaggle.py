from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
DATA_DIR = ROOT / "data" / "multidoc"
HF_CACHE = ROOT / ".hf-cache"


EVAL_MODES = {
    "passport_curriculum": DATA_DIR / "passport_eval_diverse_v1.jsonl",
    "driver_license_curriculum": DATA_DIR / "driver_license_eval_diverse_v1.jsonl",
    "mixed_identity": DATA_DIR / "identity_eval_v5.jsonl",
    "mixed_identity_diverse": DATA_DIR / "identity_eval_diverse_v1.jsonl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluation on Kaggle.")
    parser.add_argument(
        "--mode",
        choices=sorted(EVAL_MODES.keys()),
        default="passport_curriculum",
        help="Which prepared eval split to use.",
    )
    parser.add_argument(
        "--adapter-dir",
        required=True,
        help="Path to trained adapter directory inside the Kaggle workspace.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output JSON file path.",
    )
    parser.add_argument("--max-image-side", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-generation-time", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or str(ROOT / f"benchmark_{args.mode}.json")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(BACKEND_DIR)
    env["HF_HOME"] = str(HF_CACHE)
    env["TRANSFORMERS_CACHE"] = str(HF_CACHE / "transformers")
    env["HF_HUB_CACHE"] = str(HF_CACHE / "hub")
    env["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE / "hub")
    env["MAX_IMAGE_SIDE"] = str(args.max_image_side)
    env["MAX_NEW_TOKENS"] = str(args.max_new_tokens)
    env["MAX_GENERATION_TIME"] = str(args.max_generation_time)
    env["OPENBLAS_NUM_THREADS"] = env.get("OPENBLAS_NUM_THREADS", "1")
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
    env["MKL_NUM_THREADS"] = env.get("MKL_NUM_THREADS", "1")

    command = [
        sys.executable,
        str(BACKEND_DIR / "scripts" / "evaluate_multidoc_engines.py"),
        "--datasets",
        str(EVAL_MODES[args.mode]),
        "--adapter-dir",
        args.adapter_dir,
        "--output",
        output,
    ]
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=str(ROOT), env=env, check=True)


if __name__ == "__main__":
    main()
