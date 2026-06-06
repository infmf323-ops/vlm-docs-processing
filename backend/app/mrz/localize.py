"""Локализация и выравнивание полосы машиночитаемой зоны на изображении.

Конвейер локализации (OpenCV):

1. Детекция — blackhat по тёмному тексту, горизонтальный градиент и
   морфологическое замыкание сливают строки MRZ в один широкий прямоугольник,
   который отбирается по геометрии (широкий, низкий, в нижней части кадра).
2. Выравнивание (deskew) — по найденному прямоугольнику строится повёрнутый
   ограничивающий прямоугольник (`minAreaRect`), и полоса приводится к
   прямоугольному виду перспективным преобразованием по четырём углам
   (`warpPerspective`). Это исправляет не только наклон, но и перспективные
   искажения «документ на столе», которые были узким местом базового подхода.

При отсутствии OpenCV или неудаче детекции используется запасной нижний кроп.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


def _bottom_crop(image: Image.Image, fraction: float, max_width: int) -> Image.Image:
    w, h = image.size
    crop = image.convert("RGB").crop((0, int(h * (1 - fraction)), w, h))
    return _bound_width(crop, max_width)


def _bound_width(crop: Image.Image, max_width: int) -> Image.Image:
    if crop.width > max_width:
        nh = int(crop.height * max_width / crop.width)
        crop = crop.resize((max_width, nh), Image.LANCZOS)
    return crop


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Упорядочивает 4 точки как (tl, tr, br, bl)."""
    pts = pts.astype("float32")
    by_x = pts[np.argsort(pts[:, 0])]
    left, right = by_x[:2], by_x[2:]
    tl, bl = left[np.argsort(left[:, 1])]
    tr, br = right[np.argsort(right[:, 1])]
    return np.array([tl, tr, br, bl], dtype="float32")


def _four_point_warp(rgb: np.ndarray, box: np.ndarray) -> np.ndarray | None:
    """Перспективное выравнивание четырёхугольника `box` к прямоугольнику."""
    rect = _order_points(box)
    tl, tr, br, bl = rect
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if width < 5 or height < 5:
        return None
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(
        rgb, matrix, (width, height), borderValue=(255, 255, 255)
    )
    if height > width:  # MRZ-полоса всегда горизонтальная
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return warped


def _detect_band(gray: np.ndarray):
    """Возвращает контур наиболее вероятной MRZ-полосы или None."""
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    rect_k = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_k)
    grad = np.abs(cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=-1))
    mn, mx = float(grad.min()), float(grad.max())
    grad = (
        ((grad - mn) / (mx - mn) * 255).astype("uint8")
        if mx > mn
        else grad.astype("uint8")
    )
    grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, rect_k)
    thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    sq_k = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sq_k)
    thresh = cv2.erode(thresh, None, iterations=2)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gh, gw = gray.shape
    best = None
    best_score = 0.0
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if not h:
            continue
        ar = w / float(h)
        cover = w / float(gw)
        if ar > 4.0 and cover > 0.5 and y > gh * 0.4:
            # Чем шире полоса и чем ниже в кадре — тем вероятнее это MRZ.
            score = cover * (1.0 + (y / gh))
            if score > best_score:
                best_score = score
                best = c
    return best


def localize_mrz(
    image: Image.Image,
    *,
    deskew: bool = True,
    fallback_fraction: float = 0.25,
    max_width: int = 1100,
) -> Image.Image:
    """Возвращает кроп полосы MRZ. При неудаче — нижнюю часть изображения."""
    if cv2 is None:
        return _bottom_crop(image, fallback_fraction, max_width)

    rgb = np.array(image.convert("RGB"))
    H, W = rgb.shape[:2]
    scale = min(1.0, 1000.0 / max(W, 1))
    work = cv2.resize(rgb, (int(W * scale), int(H * scale))) if scale < 1 else rgb
    gray = cv2.cvtColor(work, cv2.COLOR_RGB2GRAY)

    contour = _detect_band(gray)
    if contour is None:
        return _bottom_crop(image, fallback_fraction, max_width)

    inv = 1.0 / scale

    if deskew:
        # Повёрнутый прямоугольник с небольшим запасом по краям и сильнее по высоте
        # (чтобы гарантированно вошли обе строки MRZ), затем перспективная развёртка.
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(contour)
        rw_p, rh_p = rw * 1.06 + 8.0, rh * 1.6 + 10.0
        box = cv2.boxPoints(((cx, cy), (rw_p, rh_p), ang)) * inv
        box[:, 0] = np.clip(box[:, 0], 0, W - 1)
        box[:, 1] = np.clip(box[:, 1], 0, H - 1)
        warped = _four_point_warp(rgb, box)
        if warped is not None:
            return _bound_width(Image.fromarray(warped), max_width)

    # Запасной путь: осепараллельный кроп по bounding box с паддингом.
    x, y, w, h = cv2.boundingRect(contour)
    pad_x = int(w * 0.04 * inv) + 10
    pad_y = int(h * 0.6 * inv) + 8
    x0 = int(max(0, x * inv - pad_x))
    y0 = int(max(0, y * inv - pad_y))
    x1 = int(min(W, (x + w) * inv + pad_x))
    y1 = int(min(H, (y + h) * inv + pad_y))
    crop = image.convert("RGB").crop((x0, y0, x1, y1))
    return _bound_width(crop, max_width)
