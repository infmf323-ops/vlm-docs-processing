"""Самодостаточный модуль обработки машиночитаемой зоны (MRZ) паспортов TD3.

Состав:
- checksum  — алгоритм и проверка контрольных цифр ICAO 9303 (TD3);
- postprocess — коррекция одиночных ошибок распознавания по контрольным цифрам;
- parser    — разбор MRZ в нормализованные поля;
- localize  — выделение и выравнивание полосы MRZ на изображении;
- pipeline  — оркестрация (локализация -> OCR -> коррекция -> проверка -> разбор).

Модуль не зависит от тяжёлых ML-библиотек: распознавание подаётся извне как
callable, что делает пайплайн тестируемым и переносимым.
"""
from .checksum import check_digit, validate_td3, td3_line2_valid
from .postprocess import postcorrect_line2, postcorrect_mrz
from .parser import parse_td3, split_mrz_lines
from .pipeline import MrzPipelineResult, run_mrz_pipeline

__all__ = [
    "check_digit", "validate_td3", "td3_line2_valid",
    "postcorrect_line2", "postcorrect_mrz",
    "parse_td3", "split_mrz_lines",
    "MrzPipelineResult", "run_mrz_pipeline",
]
