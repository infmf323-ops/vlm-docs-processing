"""Оркестрация обработки MRZ: локализация -> OCR -> коррекция -> проверка -> разбор.

Распознавание подаётся как callable ocr_fn(image)->str, что развязывает пайплайн
с конкретной моделью и делает его полностью тестируемым.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from PIL import Image

from .checksum import validate_td3
from .localize import localize_mrz
from .parser import parse_td3, split_mrz_lines
from .postprocess import postcorrect_mrz


@dataclass
class MrzPipelineResult:
    raw_prediction: str
    line1: str
    line2: str
    mrz: str | None
    corrected: bool
    n_corrections: int
    valid: bool
    fields: dict = field(default_factory=dict)
    crop_size: tuple | None = None


def run_mrz_pipeline(
    image: Image.Image,
    ocr_fn: Callable[[Image.Image], str],
    *,
    localize: bool = True,
    postcorrect: bool = True,
    deskew: bool = True,
) -> MrzPipelineResult:
    crop = localize_mrz(image, deskew=deskew) if localize else image.convert("RGB")
    prediction = ocr_fn(crop) or ""
    line1, line2 = split_mrz_lines(prediction)
    n_corr = 0
    corrected = False
    if postcorrect and line2:
        line1, fixed, n = postcorrect_mrz(line1, line2)
        if n > 0:
            line2, corrected, n_corr = fixed, True, n
    mrz = (line1 + "\n" + line2) if (line1 and line2) else (line2 or line1 or None)
    valid = validate_td3(mrz)["valid"] if mrz else False
    fields = parse_td3(mrz) if mrz else {}
    return MrzPipelineResult(
        raw_prediction=prediction, line1=line1, line2=line2, mrz=mrz,
        corrected=corrected, n_corrections=n_corr, valid=valid,
        fields=fields, crop_size=getattr(crop, "size", None),
    )
