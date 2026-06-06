from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "multidoc"
HF_CACHE = ROOT / ".hf-cache"
BACKEND_DIR = ROOT / "backend"

PRETRAIN_TRAIN = DATA_DIR / "passport_pretrain_hf_train_v3.jsonl"
PRETRAIN_VAL = DATA_DIR / "passport_pretrain_hf_val_v3.jsonl"
RUS_TRAIN = DATA_DIR / "passport_russian_finetune_train_v3.jsonl"
RUS_VAL = DATA_DIR / "passport_russian_finetune_val_v3.jsonl"

DEFAULT_PRETRAIN_OUTPUT = ROOT / "outputs" / "paddleocr_vl_passport_pretrain_hf_mrz_v3_local"
DEFAULT_FINETUNE_OUTPUT = ROOT / "outputs" / "paddleocr_vl_passport_russian_finetune_mrz_v3_local"
DEFAULT_EVAL_OUTPUT = ROOT / "outputs" / "passport_mrz_v3_eval.json"


def run_python(script_path: Path, env: dict[str, str] | None = None) -> None:
    command = [sys.executable, str(script_path)]
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=str(ROOT), env=env, check=True)


def build_training_env(
    train_file: Path,
    val_file: Path,
    output_dir: Path,
    epochs: int,
    *,
    max_image_side: int,
    max_length: int,
    max_new_tokens: int,
    learning_rate: float,
    grad_accum: int,
    resume_from_adapter: Path | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(BACKEND_DIR)
    env["HF_HOME"] = str(HF_CACHE)
    env["TRANSFORMERS_CACHE"] = str(HF_CACHE / "transformers")
    env["HF_HUB_CACHE"] = str(HF_CACHE / "hub")
    env["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE / "hub")
    env["MULTIDOC_DATA_DIR"] = str(DATA_DIR)
    env["MULTIDOC_TRAIN_FILE"] = str(train_file)
    env["MULTIDOC_VAL_FILE"] = str(val_file)
    env["OUTPUT_DIR"] = str(output_dir)
    env["NUM_EPOCHS"] = str(epochs)
    env["MAX_IMAGE_SIDE"] = str(max_image_side)
    env["MAX_LENGTH"] = str(max_length)
    env["MAX_NEW_TOKENS"] = str(max_new_tokens)
    env["BATCH_SIZE"] = "1"
    env["GRADIENT_ACCUMULATION_STEPS"] = str(grad_accum)
    env["LEARNING_RATE"] = str(learning_rate)
    env["TARGET_FORMAT"] = "passport_mrz"
    env["OPENBLAS_NUM_THREADS"] = env.get("OPENBLAS_NUM_THREADS", "1")
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
    env["MKL_NUM_THREADS"] = env.get("MKL_NUM_THREADS", "1")
    if resume_from_adapter is not None:
        env["BASE_ADAPTER_DIR"] = str(resume_from_adapter)
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run passport MRZ-first experiment locally.")
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument("--skip-finetune", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--pretrain-epochs", type=int, default=6)
    parser.add_argument("--finetune-epochs", type=int, default=4)
    parser.add_argument("--max-image-side", type=int, default=768)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--pretrain-learning-rate", type=float, default=2e-4)
    parser.add_argument("--finetune-learning-rate", type=float, default=1e-4)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--pretrain-output", type=Path, default=DEFAULT_PRETRAIN_OUTPUT)
    parser.add_argument("--finetune-output", type=Path, default=DEFAULT_FINETUNE_OUTPUT)
    parser.add_argument("--eval-output", type=Path, default=DEFAULT_EVAL_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.prepare_data:
        run_python(BACKEND_DIR / "scripts" / "prepare_passport_transfer_v1.py")

    run_python(BACKEND_DIR / "scripts" / "validate_passport_transfer_v1.py")

    train_script = BACKEND_DIR / "scripts" / "train_paddleocr_vl_lora.py"
    eval_script = ROOT / "kaggle" / "eval_passport_mrz_on_kaggle.py"

    if not args.skip_pretrain:
        env = build_training_env(
            PRETRAIN_TRAIN,
            PRETRAIN_VAL,
            args.pretrain_output,
            args.pretrain_epochs,
            max_image_side=args.max_image_side,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            learning_rate=args.pretrain_learning_rate,
            grad_accum=args.grad_accum,
        )
        run_python(train_script, env=env)

    finetune_adapter = args.pretrain_output
    if not args.skip_finetune:
        env = build_training_env(
            RUS_TRAIN,
            RUS_VAL,
            args.finetune_output,
            args.finetune_epochs,
            max_image_side=args.max_image_side,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            learning_rate=args.finetune_learning_rate,
            grad_accum=args.grad_accum,
            resume_from_adapter=finetune_adapter,
        )
        run_python(train_script, env=env)
        finetune_adapter = args.finetune_output

    if not args.skip_eval:
        env = os.environ.copy()
        env["HF_HOME"] = str(HF_CACHE)
        env["TRANSFORMERS_CACHE"] = str(HF_CACHE / "transformers")
        env["HF_HUB_CACHE"] = str(HF_CACHE / "hub")
        env["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE / "hub")
        env["PASSPORT_ADAPTER_DIR"] = str(finetune_adapter)
        env["PASSPORT_EVAL_OUTPUT"] = str(args.eval_output)
        run_python(eval_script, env=env)


if __name__ == "__main__":
    main()
