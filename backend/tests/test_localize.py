"""Тесты локализации и выравнивания MRZ-полосы (app.mrz.localize).

Запускаются без torch/transformers — нужны только numpy, Pillow, OpenCV.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from app.mrz.localize import _order_points, localize_mrz

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


def _make_passport_with_mrz(skew: float = 0.0) -> Image.Image:
    """Белый «паспорт» с двумя строками MRZ в нижней части (опц. перспектива)."""
    W, H = 1000, 700
    canvas = Image.new("RGB", (W, H), (250, 250, 248))
    draw = ImageDraw.Draw(canvas)
    # немного «контента» сверху, чтобы низ не был единственной тёмной зоной
    draw.rectangle([60, 60, 300, 260], outline=(120, 120, 120), width=3)
    # две строки MRZ внизу
    line = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
    line2 = "L898902C36UTO7408122F1204159ZE184226B<<<<<10"
    y0 = 560
    for i, text in enumerate((line, line2)):
        x = 70
        for ch in text:
            draw.rectangle(
                [x, y0 + i * 46, x + 16, y0 + i * 46 + 30],
                fill=(20, 20, 20) if ch != "<" else (60, 60, 60),
            )
            x += 20
    if skew and cv2 is not None:
        src = np.array(canvas)
        d = skew
        pts1 = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
        pts2 = np.float32([[d, d * 0.5], [W - d, 0], [W, H - d], [d * 0.3, H]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        warped = cv2.warpPerspective(src, M, (W, H), borderValue=(250, 250, 248))
        canvas = Image.fromarray(warped)
    return canvas


def test_order_points():
    pts = np.array([[10, 10], [100, 12], [98, 60], [8, 58]], dtype="float32")
    tl, tr, br, bl = _order_points(pts[[2, 0, 3, 1]])  # перемешанный порядок
    assert tuple(tl) == (10, 10)
    assert tuple(tr) == (100, 12)
    assert tuple(br) == (98, 60)
    assert tuple(bl) == (8, 58)


def test_localize_returns_landscape_band():
    img = _make_passport_with_mrz(skew=0.0)
    crop = localize_mrz(img)
    assert isinstance(crop, Image.Image)
    # MRZ-полоса широкая и низкая
    assert crop.width > crop.height
    assert crop.width / crop.height > 2.5
    # это именно кроп, а не весь кадр
    assert crop.height < img.height


def test_localize_handles_perspective_skew():
    img = _make_passport_with_mrz(skew=70.0)
    crop = localize_mrz(img, deskew=True)
    assert isinstance(crop, Image.Image)
    # после перспективной развёртки полоса должна стать горизонтальной
    assert crop.width > crop.height


def test_localize_fallback_without_band():
    # Пустой белый кадр: детектор ничего не находит -> нижний кроп.
    blank = Image.new("RGB", (800, 600), (255, 255, 255))
    crop = localize_mrz(blank, fallback_fraction=0.25)
    assert isinstance(crop, Image.Image)
    # высота ~ четверть исходной (нижний кроп)
    assert crop.height <= 600 * 0.25 + 2


if __name__ == "__main__":
    test_order_points()
    test_localize_returns_landscape_band()
    test_localize_handles_perspective_skew()
    test_localize_fallback_without_band()
    print("OK: все тесты локализации пройдены")
