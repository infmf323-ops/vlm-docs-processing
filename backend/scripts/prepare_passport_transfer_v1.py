from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def run(script_name: str) -> None:
    script_path = ROOT / "backend" / "scripts" / script_name
    print(f"Running {script_path}")
    subprocess.run([sys.executable, str(script_path)], cwd=str(ROOT), check=True)


def main() -> None:
    run("import_hf_passport_datasets.py")
    run("profile_passport_transfer_source.py")
    run("build_passport_transfer_splits.py")
    run("build_passport_transfer_eval_sets.py")
    run("validate_passport_transfer_v1.py")


if __name__ == "__main__":
    main()
