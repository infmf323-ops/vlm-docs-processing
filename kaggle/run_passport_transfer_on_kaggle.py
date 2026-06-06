from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run passport transfer learning stages on Kaggle.")
    parser.add_argument("--pipeline-version", choices=["v1", "v2", "v3"], default="v3")
    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument("--skip-finetune", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--pretrain-epochs", type=int, default=6)
    parser.add_argument("--finetune-epochs", type=int, default=4)
    parser.add_argument("--pretrain-output", default=None)
    parser.add_argument("--finetune-output", default=None)
    parser.add_argument("--eval-output", default=str(ROOT / "passport_transfer_eval_kaggle.json"))
    return parser.parse_args()


def run(command: list[str], env: dict[str, str] | None = None) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=str(ROOT), env=env, check=True)


def main() -> None:
    args = parse_args()
    python = sys.executable
    if args.pipeline_version == "v3":
        pretrain_mode = "passport_pretrain_hf_v3"
        finetune_mode = "passport_russian_finetune_v3"
        default_pretrain_output = str(ROOT / "outputs" / "paddleocr_vl_passport_pretrain_hf_v3_kaggle")
        default_finetune_output = str(ROOT / "outputs" / "paddleocr_vl_passport_russian_finetune_v3_kaggle")
    elif args.pipeline_version == "v2":
        pretrain_mode = "passport_pretrain_hf_v2"
        finetune_mode = "passport_russian_finetune_v2"
        default_pretrain_output = str(ROOT / "outputs" / "paddleocr_vl_passport_pretrain_hf_v2_kaggle")
        default_finetune_output = str(ROOT / "outputs" / "paddleocr_vl_passport_russian_finetune_v2_kaggle")
    else:
        pretrain_mode = "passport_pretrain_hf_v1"
        finetune_mode = "passport_russian_finetune_v1"
        default_pretrain_output = str(ROOT / "outputs" / "paddleocr_vl_passport_pretrain_hf_v1_kaggle")
        default_finetune_output = str(ROOT / "outputs" / "paddleocr_vl_passport_russian_finetune_v1_kaggle")

    pretrain_output = args.pretrain_output or default_pretrain_output
    finetune_output = args.finetune_output or default_finetune_output

    if not args.skip_pretrain:
        run(
            [
                python,
                str(ROOT / "kaggle" / "run_train_on_kaggle.py"),
                "--mode",
                pretrain_mode,
                "--epochs",
                str(args.pretrain_epochs),
                "--output-dir",
                pretrain_output,
            ]
        )

    if not args.skip_finetune:
        run(
            [
                python,
                str(ROOT / "kaggle" / "run_train_on_kaggle.py"),
                "--mode",
                finetune_mode,
                "--epochs",
                str(args.finetune_epochs),
                "--output-dir",
                finetune_output,
                "--resume-from-adapter",
                pretrain_output,
            ]
        )

    if not args.skip_eval:
        env = os.environ.copy()
        env["PASSPORT_ADAPTER_DIR"] = finetune_output
        env["PASSPORT_EVAL_OUTPUT"] = args.eval_output
        run([python, str(ROOT / "kaggle" / "eval_passport_flat_on_kaggle.py")], env=env)


if __name__ == "__main__":
    main()
