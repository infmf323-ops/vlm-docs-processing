from __future__ import annotations

import json
import os
from pathlib import Path

from PIL import Image

from app.schemas.documents import DocumentType, ProcessingEngine
from app.schemas.jobs import JobRequestOptions
from app.services.extraction import PaddleOCRVLEngine


ROOT = Path("E:/thesis")
QUEUE_PATH = ROOT / "data" / "multidoc" / "annotation_queue.jsonl"
MAX_IMAGE_SIDE = int(os.getenv("MAX_IMAGE_SIDE", "1280"))
LIMIT = int(os.getenv("LIMIT", "0"))


def prepare_image(image: Image.Image, *, max_side: int) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image

    scale = max_side / float(longest)
    target_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(target_size, Image.Resampling.LANCZOS)


def load_rows() -> list[dict]:
    rows: list[dict] = []
    with QUEUE_PATH.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_rows(rows: list[dict]) -> None:
    with QUEUE_PATH.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    rows = load_rows()
    engine = PaddleOCRVLEngine()
    processed = 0
    improved = 0

    for row in rows:
        if row.get("status") != "needs_annotation_external":
            continue
        if row.get("heuristic_engine"):
            continue

        document_type = str(row.get("document_type") or "").strip()
        if document_type not in {"passport", "id_card", "driver_license"}:
            continue

        image_path = Path(str(row.get("image_path")))
        if not image_path.exists():
            row["heuristic_error"] = "image_not_found"
            continue

        request = JobRequestOptions(
            document_type=DocumentType(document_type),
            extraction_engine=ProcessingEngine.PADDLEOCR_VL,
        )

        with Image.open(image_path) as image:
            image = prepare_image(image, max_side=MAX_IMAGE_SIDE)
            output = engine.predict(
                image=image,
                source_filename=image_path.name,
                request=request,
            )

        fields = output.normalized_result.fields.model_dump() if output.normalized_result.fields else {}
        raw_text = output.normalized_result.raw_text or ""
        validation = output.normalized_result.validation
        processing_meta = output.normalized_result.processing_meta

        row["heuristic_engine"] = "paddleocr_vl"
        row["heuristic_valid"] = validation.is_valid
        row["heuristic_elapsed_ms"] = processing_meta.elapsed_ms if processing_meta else None
        row["heuristic_raw_preview"] = raw_text[:500]
        processed += 1

        if validation.is_valid:
            row["fields"] = fields
            row["status"] = "draft_from_heuristic_pipeline"
            row["draft_source"] = "paddleocr_vl_heuristic"
            improved += 1
        else:
            row["heuristic_issues"] = [issue.message for issue in validation.issues]

        save_rows(rows)
        print(f"processed_row={row['id']} valid={validation.is_valid}")

        if LIMIT and processed >= LIMIT:
            break

    save_rows(rows)
    print(QUEUE_PATH)
    print(f"processed={processed}")
    print(f"improved={improved}")


if __name__ == "__main__":
    main()
