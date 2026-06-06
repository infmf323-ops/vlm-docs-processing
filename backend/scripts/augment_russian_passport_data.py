"""
augment_russian_passport_data.py
=================================
Expands the tiny Russian passport finetune training split through image
augmentation.  Because the supervised Russian subset contains only 3 images
the model cannot learn stable adaptation from it.  This script creates
*N* augmented copies of every image while keeping the same field annotations
so the JSONL training format stays identical.

Augmentation is document-realistic: we simulate variation in scan angle,
brightness, print density and JPEG artefacts — the kinds of degradation seen
in real passport photographs/scans.  We explicitly avoid any distortion that
would make text unreadable (no heavy blur, no heavy geometric warp).

Usage (run from the E:/thesis root):
    python backend/scripts/augment_russian_passport_data.py

Outputs:
    data/multidoc/external/augmented_russian/   – augmented JPEG files
    data/multidoc/passport_russian_finetune_train_v3_aug.jsonl  – expanded split

The augmented split can be used as a drop-in replacement for the original
finetune train split.  The validation split is intentionally left unchanged
to keep evaluation honest.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "multidoc"
AUG_IMAGE_DIR = DATA_DIR / "external" / "augmented_russian"

# Source split (v3 is the cleanest transfer baseline)
SOURCE_JSONL = DATA_DIR / "passport_russian_finetune_train_v3.jsonl"
OUTPUT_JSONL = DATA_DIR / "passport_russian_finetune_train_v3_aug.jsonl"

# How many augmented copies to generate per original image.
# 3 originals × 6 augmentations = 18 training rows (more tractable for finetune).
AUGMENTATIONS_PER_IMAGE = 6
SEED = 42


def resolve_image_path(raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.exists():
        return candidate
    normalized = raw_path.replace("\\", "/")
    for prefix in ["E:/thesis/", "/kaggle/working/thesis_bundle/"]:
        if normalized.startswith(prefix):
            relative = normalized[len(prefix):].lstrip("/")
            remapped = PROJECT_ROOT / Path(relative)
            if remapped.exists():
                return remapped
    return candidate


def load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def random_rotate(img: Image.Image, rng: random.Random) -> Image.Image:
    """Slight rotation (±8°) with white fill — simulates non-straight scan."""
    angle = rng.uniform(-8.0, 8.0)
    return img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(255, 255, 255))


def random_brightness(img: Image.Image, rng: random.Random) -> Image.Image:
    """Brightness variation ±25 %."""
    factor = rng.uniform(0.75, 1.25)
    return ImageEnhance.Brightness(img).enhance(factor)


def random_contrast(img: Image.Image, rng: random.Random) -> Image.Image:
    """Contrast variation ±20 %."""
    factor = rng.uniform(0.80, 1.20)
    return ImageEnhance.Contrast(img).enhance(factor)


def random_sharpness(img: Image.Image, rng: random.Random) -> Image.Image:
    """Sharpness variation — from slightly blurred to slightly over-sharpened."""
    factor = rng.uniform(0.5, 2.0)
    return ImageEnhance.Sharpness(img).enhance(factor)


def random_color(img: Image.Image, rng: random.Random) -> Image.Image:
    """Mild color saturation shift — simulates different scanner colour profiles."""
    factor = rng.uniform(0.85, 1.15)
    return ImageEnhance.Color(img).enhance(factor)


def random_crop_pad(img: Image.Image, rng: random.Random) -> Image.Image:
    """Random small crop followed by resize back — simulates framing variance."""
    w, h = img.size
    margin_x = int(w * rng.uniform(0.0, 0.06))
    margin_y = int(h * rng.uniform(0.0, 0.06))
    left = rng.randint(0, margin_x)
    top = rng.randint(0, margin_y)
    right = w - rng.randint(0, margin_x)
    bottom = h - rng.randint(0, margin_y)
    cropped = img.crop((left, top, right, bottom))
    return cropped.resize((w, h), Image.Resampling.LANCZOS)


def random_jpeg_quality(img: Image.Image, rng: random.Random) -> Image.Image:
    """Re-compress with random JPEG quality — simulates scan/save artefacts."""
    import io
    quality = rng.randint(70, 97)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


# Available augmentation transforms (each applied with some probability)
TRANSFORMS = [
    random_rotate,
    random_brightness,
    random_contrast,
    random_sharpness,
    random_color,
    random_crop_pad,
    random_jpeg_quality,
]


def augment_image(img: Image.Image, rng: random.Random, aug_index: int) -> Image.Image:
    """
    Apply a deterministic-but-varied augmentation pipeline.
    Each aug_index selects a different combination of transforms.
    """
    # Always apply brightness + contrast (most realistic)
    result = random_brightness(img, rng)
    result = random_contrast(result, rng)

    # Conditionally apply others based on aug_index to ensure diversity
    if aug_index % 2 == 0:
        result = random_rotate(result, rng)
    if aug_index % 3 == 0:
        result = random_sharpness(result, rng)
    if aug_index % 2 == 1:
        result = random_crop_pad(result, rng)
    if aug_index % 4 == 0:
        result = random_color(result, rng)
    # Always finish with JPEG re-compression to simulate real scan pipeline
    result = random_jpeg_quality(result, rng)
    return result


def main() -> None:
    rng = random.Random(SEED)
    AUG_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    source_rows = load_rows(SOURCE_JSONL)
    if not source_rows:
        raise SystemExit(f"Source JSONL is empty: {SOURCE_JSONL}")

    print(f"Source rows: {len(source_rows)}")
    print(f"Augmentations per image: {AUGMENTATIONS_PER_IMAGE}")
    print(f"Expected total rows: {len(source_rows) * (1 + AUGMENTATIONS_PER_IMAGE)}")

    output_rows: list[dict] = []

    for row in source_rows:
        # Keep the original row as-is
        output_rows.append(row)

        img_path = resolve_image_path(str(row["image_path"]))
        if not img_path.exists():
            print(f"WARNING: image not found: {img_path}  — skipping augmentation for this row")
            continue

        original_img = Image.open(img_path).convert("RGB")
        base_id = row["id"]

        for aug_idx in range(AUGMENTATIONS_PER_IMAGE):
            aug_img = augment_image(original_img, rng, aug_idx)

            aug_filename = f"{base_id}__aug{aug_idx:02d}.jpg"
            aug_img_path = AUG_IMAGE_DIR / aug_filename
            aug_img.save(str(aug_img_path), format="JPEG", quality=95)

            aug_row = dict(row)
            aug_row["id"] = f"{base_id}__aug{aug_idx:02d}"
            aug_row["image_path"] = str(aug_img_path)
            aug_row["is_augmented"] = True
            aug_row["augmentation_index"] = aug_idx
            aug_row["source_id"] = base_id
            output_rows.append(aug_row)

    OUTPUT_JSONL.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in output_rows),
        encoding="utf-8",
    )

    print(f"\nDone.")
    print(f"Augmented images → {AUG_IMAGE_DIR}")
    print(f"Output JSONL     → {OUTPUT_JSONL}")
    print(f"Total rows       : {len(output_rows)}  ({len(source_rows)} original + {len(output_rows) - len(source_rows)} augmented)")


if __name__ == "__main__":
    main()
