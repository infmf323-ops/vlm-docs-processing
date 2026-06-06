from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from PIL import Image

from app.schemas.documents import DocumentType, ProcessingEngine
from app.schemas.jobs import JobRequestOptions
from app.services.extraction import DonutEngineAdapter, PaddleOCRVLEngine, Qwen25VLEngine


def configure_hf_cache(cache_root: Path) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_root / "hub")
    os.environ["HF_HUB_CACHE"] = str(cache_root / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_root / "transformers")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a direct document extraction demo.")
    parser.add_argument("--file", required=True, help="Absolute path to the source document image.")
    parser.add_argument(
        "--document-type",
        required=True,
        choices=[value.value for value in DocumentType],
        help="Document type to use for field mapping.",
    )
    parser.add_argument(
        "--engine",
        default=ProcessingEngine.PADDLEOCR_VL.value,
        choices=[value.value for value in ProcessingEngine],
        help="Extraction engine to run.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path. Defaults to <input_stem>_<engine>_result.json next to the source file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = Path(args.file)
    if not source_path.exists():
        raise SystemExit(f"Source file not found: {source_path}")

    output_path = (
        Path(args.output)
        if args.output
        else source_path.with_name(f"{source_path.stem}_{args.engine}_result.json")
    )

    configure_hf_cache(Path("E:/thesis/.hf-cache"))

    image = Image.open(source_path).convert("RGB")
    request = JobRequestOptions(
        document_type=DocumentType(args.document_type),
        extraction_engine=ProcessingEngine(args.engine),
    )

    engine_map = {
        ProcessingEngine.DONUT: DonutEngineAdapter,
        ProcessingEngine.PADDLEOCR_VL: PaddleOCRVLEngine,
        ProcessingEngine.QWEN2_5_VL: Qwen25VLEngine,
    }
    engine_cls = engine_map[request.extraction_engine]
    engine = engine_cls()
    result = engine.predict(
        image=image,
        source_filename=source_path.name,
        request=request,
    )

    payload = {
        "source_file": str(source_path),
        "engine": args.engine,
        "document_type": args.document_type,
        "normalized_result": result.normalized_result.model_dump(mode="json"),
        "raw_result": result.raw_result,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
