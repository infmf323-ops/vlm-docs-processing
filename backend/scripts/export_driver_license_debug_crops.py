from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageOps


def crop_document_region(image: Image.Image) -> Image.Image:
    grayscale = ImageOps.grayscale(image)
    mask = grayscale.point(lambda px: 255 if px < 245 else 0)
    bbox = mask.getbbox()
    if not bbox:
        return image
    left, top, right, bottom = bbox
    width, height = image.size
    if (right - left) < width * 0.35 or (bottom - top) < height * 0.25:
        return image
    pad_x = max(8, int((right - left) * 0.02))
    pad_y = max(8, int((bottom - top) * 0.02))
    cropped_box = (
        max(0, left - pad_x),
        max(0, top - pad_y),
        min(width, right + pad_x),
        min(height, bottom + pad_y),
    )
    return image.crop(cropped_box)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    image_path = Path(args.image)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path).convert("RGB") as raw_image:
        image = crop_document_region(raw_image)
        width, height = image.size

        focus_box = (int(width * 0.10), int(height * 0.12), int(width * 0.97), int(height * 0.90))
        fx0, fy0, fx1, fy1 = focus_box
        fw = fx1 - fx0
        fh = fy1 - fy0
        regions = {
            "cropped_document": (0, 0, width, height),
            "focus": focus_box,
            "header": (fx0 + int(fw * 0.26), fy0 + int(fh * 0.04), fx0 + int(fw * 0.92), fy0 + int(fh * 0.34)),
            "text_block": (fx0 + int(fw * 0.26), fy0 + int(fh * 0.14), fx0 + int(fw * 0.72), fy0 + int(fh * 0.80)),
            "detail": (fx0 + int(fw * 0.26), fy0 + int(fh * 0.10), fx0 + int(fw * 0.98), fy0 + int(fh * 0.96)),
            "dates": (fx0 + int(fw * 0.28), fy0 + int(fh * 0.46), fx0 + int(fw * 0.98), fy0 + int(fh * 0.96)),
            "issue": (fx0 + int(fw * 0.78), fy0 + int(fh * 0.72), fx0 + int(fw * 0.98), fy0 + int(fh * 0.98)),
            "bottom_right": (fx0 + int(fw * 0.70), fy0 + int(fh * 0.50), fx0 + int(fw * 0.99), fy0 + int(fh * 0.96)),
            "number": (fx0 + int(fw * 0.24), fy0 + int(fh * 0.12), fx0 + int(fw * 0.82), fy0 + int(fh * 0.42)),
        }

        manifest: dict[str, dict[str, int | str]] = {}
        for name, box in regions.items():
            crop = image.crop(box)
            output_path = out_dir / f"{name}.png"
            crop.save(output_path)
            manifest[name] = {
                "path": str(output_path).replace("\\", "/"),
                "left": box[0],
                "top": box[1],
                "right": box[2],
                "bottom": box[3],
            }

        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(manifest_path)


if __name__ == "__main__":
    main()
