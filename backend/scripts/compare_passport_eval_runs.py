from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two passport eval JSON outputs.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    return parser.parse_args()


def load_eval(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sample_map(payload: dict) -> dict[str, dict]:
    return {str(sample.get("id")): sample for sample in payload.get("samples", [])}


def metric_value(sample: dict, key: str, *, normalized: bool = False) -> float:
    bucket = sample.get("normalized_metrics") if normalized else sample.get("metrics")
    bucket = bucket or {}
    return float(bucket.get(key, 0.0))


def print_summary(label: str, payload: dict) -> None:
    print(label)
    print(" exact      :", json.dumps(payload.get("metrics", {}), ensure_ascii=False))
    print(" normalized :", json.dumps(payload.get("normalized_metrics", {}), ensure_ascii=False))
    print()


def main() -> None:
    args = parse_args()
    baseline = load_eval(args.baseline)
    candidate = load_eval(args.candidate)

    print_summary("Baseline", baseline)
    print_summary("Candidate", candidate)

    baseline_samples = sample_map(baseline)
    candidate_samples = sample_map(candidate)
    shared_ids = sorted(set(baseline_samples) & set(candidate_samples))

    print("Per-sample exact F1 delta:")
    deltas = []
    for sample_id in shared_ids:
        base_f1 = metric_value(baseline_samples[sample_id], "f1_score")
        cand_f1 = metric_value(candidate_samples[sample_id], "f1_score")
        delta = cand_f1 - base_f1
        deltas.append((delta, sample_id, base_f1, cand_f1))
    for delta, sample_id, base_f1, cand_f1 in sorted(deltas):
        print(f"{sample_id}: baseline={base_f1:.4f}, candidate={cand_f1:.4f}, delta={delta:+.4f}")
    print()

    print("Per-sample normalized F1 delta:")
    norm_deltas = []
    for sample_id in shared_ids:
        base_f1 = metric_value(baseline_samples[sample_id], "f1_score", normalized=True)
        cand_f1 = metric_value(candidate_samples[sample_id], "f1_score", normalized=True)
        delta = cand_f1 - base_f1
        norm_deltas.append((delta, sample_id, base_f1, cand_f1))
    for delta, sample_id, base_f1, cand_f1 in sorted(norm_deltas):
        print(f"{sample_id}: baseline={base_f1:.4f}, candidate={cand_f1:.4f}, delta={delta:+.4f}")


if __name__ == "__main__":
    main()
