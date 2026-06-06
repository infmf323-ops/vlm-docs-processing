from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
DATA_DIR = ROOT / "data" / "multidoc"
OUTPUT_DIR = ROOT / "outputs"
HF_CACHE = ROOT / ".hf-cache"


TRAINING_MODES = {
    "passport_curriculum": {
        "train": DATA_DIR / "passport_curriculum_train_v1.jsonl",
        "val": DATA_DIR / "passport_curriculum_val_v1.jsonl",
        "output": OUTPUT_DIR / "paddleocr_vl_passport_curriculum_kaggle",
        "num_epochs": "3",
        "max_image_side": "768",
        "max_length": "2048",
        "max_new_tokens": "256",
        "grad_accum": "2",
    },
    "passport_curriculum_flat": {
        "train": DATA_DIR / "passport_curriculum_train_v1.jsonl",
        "val": DATA_DIR / "passport_curriculum_val_v1.jsonl",
        "output": OUTPUT_DIR / "paddleocr_vl_passport_curriculum_flat_kaggle",
        "num_epochs": "3",
        "max_image_side": "768",
        "max_length": "2048",
        "max_new_tokens": "192",
        "grad_accum": "2",
        "target_format": "passport_flat",
    },
    "passport_curriculum_flat_v2": {
        "train": DATA_DIR / "passport_curriculum_train_v2.jsonl",
        "val": DATA_DIR / "passport_curriculum_val_v2.jsonl",
        "output": OUTPUT_DIR / "paddleocr_vl_passport_curriculum_flat_v2_kaggle",
        "num_epochs": "6",
        "max_image_side": "768",
        "max_length": "2048",
        "max_new_tokens": "192",
        "grad_accum": "2",
        "target_format": "passport_flat",
    },
    "passport_curriculum_flat_v3": {
        "train": DATA_DIR / "passport_curriculum_train_v3.jsonl",
        "val": DATA_DIR / "passport_curriculum_val_v3.jsonl",
        "output": OUTPUT_DIR / "paddleocr_vl_passport_curriculum_flat_v3_kaggle",
        "num_epochs": "6",
        "max_image_side": "768",
        "max_length": "2048",
        "max_new_tokens": "192",
        "grad_accum": "2",
        "target_format": "passport_flat",
    },
    "passport_pretrain_hf_v1": {
        "train": DATA_DIR / "passport_pretrain_hf_train_v1.jsonl",
        "val": DATA_DIR / "passport_pretrain_hf_val_v1.jsonl",
        "output": OUTPUT_DIR / "paddleocr_vl_passport_pretrain_hf_v1_kaggle",
        "num_epochs": "6",
        "max_image_side": "768",
        "max_length": "2048",
        "max_new_tokens": "192",
        "grad_accum": "2",
        "target_format": "passport_flat",
    },
    "passport_russian_finetune_v1": {
        "train": DATA_DIR / "passport_russian_finetune_train_v1.jsonl",
        "val": DATA_DIR / "passport_russian_finetune_val_v1.jsonl",
        "output": OUTPUT_DIR / "paddleocr_vl_passport_russian_finetune_v1_kaggle",
        "num_epochs": "6",
        "max_image_side": "768",
        "max_length": "2048",
        "max_new_tokens": "192",
        "grad_accum": "1",
        "target_format": "passport_flat",
        # Lower LR for adaptation stage: prevents catastrophic forgetting of pretrain
        "learning_rate": "5e-5",
    },
    "passport_pretrain_hf_v2": {
        "train": DATA_DIR / "passport_pretrain_hf_train_v2.jsonl",
        "val": DATA_DIR / "passport_pretrain_hf_val_v2.jsonl",
        "output": OUTPUT_DIR / "paddleocr_vl_passport_pretrain_hf_v2_kaggle",
        "num_epochs": "6",
        "max_image_side": "768",
        "max_length": "2048",
        "max_new_tokens": "192",
        "grad_accum": "2",
        "target_format": "passport_flat",
    },
    "passport_russian_finetune_v2": {
        "train": DATA_DIR / "passport_russian_finetune_train_v2.jsonl",
        "val": DATA_DIR / "passport_russian_finetune_val_v2.jsonl",
        "output": OUTPUT_DIR / "paddleocr_vl_passport_russian_finetune_v2_kaggle",
        "num_epochs": "6",
        "max_image_side": "768",
        "max_length": "2048",
        "max_new_tokens": "192",
        "grad_accum": "1",
        "target_format": "passport_flat",
        "learning_rate": "5e-5",
    },
    "passport_pretrain_hf_v3": {
        "train": DATA_DIR / "passport_pretrain_hf_train_v3.jsonl",
        "val": DATA_DIR / "passport_pretrain_hf_val_v3.jsonl",
        "output": OUTPUT_DIR / "paddleocr_vl_passport_pretrain_hf_v3_kaggle",
        "num_epochs": "6",
        "max_image_side": "768",
        "max_length": "2048",
        "max_new_tokens": "192",
        "grad_accum": "2",
        "target_format": "passport_flat",
    },
    "passport_russian_finetune_v3": {
        "train": DATA_DIR / "passport_russian_finetune_train_v3.jsonl",
        "val": DATA_DIR / "passport_russian_finetune_val_v3.jsonl",
        "output": OUTPUT_DIR / "paddleocr_vl_passport_russian_finetune_v3_kaggle",
        "num_epochs": "6",
        "max_image_side": "768",
        "max_length": "2048",
        "max_new_tokens": "192",
        "grad_accum": "1",
        "target_format": "passport_flat",
        # Lower LR for adaptation stage: prevents catastrophic forgetting of pretrain
        "learning_rate": "5e-5",
    },
    # Augmented variant: 3 originals + 18 augmented = 21 training rows.
    # Run augment_russian_passport_data.py first to generate the JSONL.
    "passport_russian_finetune_v3_aug": {
        "train": DATA_DIR / "passport_russian_finetune_train_v3_aug.jsonl",
        "val": DATA_DIR / "passport_russian_finetune_val_v3.jsonl",
        "output": OUTPUT_DIR / "paddleocr_vl_passport_russian_finetune_v3_aug_kaggle",
        "num_epochs": "8",
        "max_image_side": "768",
        "max_length": "2048",
        "max_new_tokens": "192",
        "grad_accum": "1",
        "target_format": "passport_flat",
        "learning_rate": "3e-5",
    },
    "passport_pretrain_hf_mrz_v3": {
        "train": DATA_DIR / "passport_pretrain_hf_train_v3.jsonl",
        "val": DATA_DIR / "passport_pretrain_hf_val_v3.jsonl",
        "output": OUTPUT_DIR / "paddleocr_vl_passport_pretrain_hf_mrz_v3_kaggle",
        "num_epochs": "6",
        "max_image_side": "768",
        "max_length": "2048",
        "max_new_tokens": "128",
        "grad_accum": "2",
        "target_format": "passport_mrz",
    },
    "passport_russian_finetune_mrz_v3": {
        "train": DATA_DIR / "passport_russian_finetune_train_v3.jsonl",
        "val": DATA_DIR / "passport_russian_finetune_val_v3.jsonl",
        "output": OUTPUT_DIR / "paddleocr_vl_passport_russian_finetune_mrz_v3_kaggle",
        "num_epochs": "6",
        "max_image_side": "768",
        "max_length": "2048",
        "max_new_tokens": "128",
        "grad_accum": "1",
        "target_format": "passport_mrz",
        "learning_rate": "5e-5",
    },
    "driver_license_curriculum": {
        "train": DATA_DIR / "driver_license_curriculum_train_v1.jsonl",
        "val": DATA_DIR / "driver_license_curriculum_val_v1.jsonl",
        "output": OUTPUT_DIR / "paddleocr_vl_driver_license_curriculum_kaggle",
        "num_epochs": "4",
        "max_image_side": "768",
        "max_length": "2048",
        "max_new_tokens": "256",
        "grad_accum": "2",
    },
    "mixed_identity": {
        "train": DATA_DIR / "pilot_train_strong.jsonl",
        "val": DATA_DIR / "pilot_val.jsonl",
        "output": OUTPUT_DIR / "paddleocr_vl_mixed_identity_kaggle",
        "num_epochs": "2",
        "max_image_side": "768",
        "max_length": "2048",
        "max_new_tokens": "256",
        "grad_accum": "2",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PaddleOCR-VL LoRA training on Kaggle.")
    parser.add_argument(
        "--mode",
        choices=sorted(TRAINING_MODES.keys()),
        default="passport_curriculum",
        help="Which prepared training split to use.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Optional override for NUM_EPOCHS.")
    parser.add_argument("--max-image-side", type=int, default=None, help="Optional override for MAX_IMAGE_SIDE.")
    parser.add_argument("--max-length", type=int, default=None, help="Optional override for MAX_LENGTH.")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Optional override for MAX_NEW_TOKENS.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size.")
    parser.add_argument("--grad-accum", type=int, default=None, help="Optional override for gradient accumulation.")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate override. If omitted, uses the mode's default (2e-4 for pretrain, 5e-5 for finetune).",
    )
    parser.add_argument("--output-dir", default=None, help="Optional custom output directory.")
    parser.add_argument(
        "--resume-from-adapter",
        default=None,
        help="Optional adapter directory to continue training from.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only validate one forward path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = TRAINING_MODES[args.mode]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(BACKEND_DIR)
    env["HF_HOME"] = str(HF_CACHE)
    env["TRANSFORMERS_CACHE"] = str(HF_CACHE / "transformers")
    env["HF_HUB_CACHE"] = str(HF_CACHE / "hub")
    env["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE / "hub")
    env["MULTIDOC_DATA_DIR"] = str(DATA_DIR)
    env["MULTIDOC_TRAIN_FILE"] = str(spec["train"])
    env["MULTIDOC_VAL_FILE"] = str(spec["val"])
    env["OUTPUT_DIR"] = args.output_dir or str(spec["output"])
    env["NUM_EPOCHS"] = str(args.epochs or spec["num_epochs"])
    env["MAX_IMAGE_SIDE"] = str(args.max_image_side or spec["max_image_side"])
    env["MAX_LENGTH"] = str(args.max_length or spec["max_length"])
    env["MAX_NEW_TOKENS"] = str(args.max_new_tokens or spec["max_new_tokens"])
    env["BATCH_SIZE"] = str(args.batch_size)
    env["GRADIENT_ACCUMULATION_STEPS"] = str(args.grad_accum or spec["grad_accum"])
    # CLI --learning-rate overrides spec; spec overrides global default (2e-4).
    effective_lr = args.learning_rate or spec.get("learning_rate", "2e-4")
    env["LEARNING_RATE"] = str(effective_lr)
    env["TARGET_FORMAT"] = str(spec.get("target_format", "json"))
    if args.resume_from_adapter:
        env["BASE_ADAPTER_DIR"] = str(args.resume_from_adapter)
    env["OPENBLAS_NUM_THREADS"] = env.get("OPENBLAS_NUM_THREADS", "1")
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
    env["MKL_NUM_THREADS"] = env.get("MKL_NUM_THREADS", "1")
    if args.dry_run:
        env["DRY_RUN"] = "1"

    command = [sys.executable, str(BACKEND_DIR / "scripts" / "train_paddleocr_vl_lora.py")]
    print("Running:", " ".join(command))
    print("Mode:", args.mode)
    print("Train:", env["MULTIDOC_TRAIN_FILE"])
    print("Val:", env["MULTIDOC_VAL_FILE"])
    print("Output:", env["OUTPUT_DIR"])
    subprocess.run(command, cwd=str(ROOT), env=env, check=True)


if __name__ == "__main__":
    main()
