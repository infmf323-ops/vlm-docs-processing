from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


ROOT = Path("E:/thesis")
DEFAULT_EVAL_PATH = ROOT / "passport_curriculum_flat_eval_kaggle.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize passport eval JSON outputs.")
    parser.add_argument("--input", type=Path, default=DEFAULT_EVAL_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))

    print("Exact metrics:")
    print(json.dumps(payload.get("metrics", {}), ensure_ascii=False, indent=2))
    print()
    print("Normalized metrics:")
    print(json.dumps(payload.get("normalized_metrics", {}), ensure_ascii=False, indent=2))
    print()

    exact_field_counter = Counter()
    normalized_field_counter = Counter()
    sample_counter = len(payload.get("samples", []))

    for sample in payload.get("samples", []):
        for field, field_payload in (sample.get("field_breakdown") or {}).items():
            if field_payload.get("match"):
                exact_field_counter[field] += 1
        for field, field_payload in (sample.get("normalized_field_breakdown") or {}).items():
            if field_payload.get("match"):
                normalized_field_counter[field] += 1

    print("Per-field exact matches:")
    for field, count in sorted(exact_field_counter.items()):
        print(f"{field}: {count}/{sample_counter}")
    print()

    print("Per-field normalized matches:")
    for field, count in sorted(normalized_field_counter.items()):
        print(f"{field}: {count}/{sample_counter}")
    print()

    print("Worst samples by exact F1:")
    worst_samples = sorted(
        payload.get("samples", []),
        key=lambda item: float(((item.get("metrics") or {}).get("f1_score", 0.0))),
    )
    for sample in worst_samples[:5]:
        metrics = sample.get("metrics") or {}
        norm_metrics = sample.get("normalized_metrics") or {}
        print(
            f"{sample.get('id')}: "
            f"exact_f1={metrics.get('f1_score', 0.0):.4f}, "
            f"normalized_f1={norm_metrics.get('f1_score', 0.0):.4f}"
        )


if __name__ == "__main__":
    main()
