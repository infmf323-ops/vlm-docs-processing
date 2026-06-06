"""Внешний baseline: готовый OCR (Tesseract / EasyOCR) + наш TD3-разбор и проверка.

Назначение
----------
Честное сравнение «альтернативной философии» (коробочный OCR + детерминированный
парсер MRZ) с дообученной моделью PaddleOCR-VL + LoRA на ТОЙ ЖЕ отложенной выборке.
Ключевой принцип: распознавание берётся внешнее, а локализация/постобработка/проверка
по контрольным цифрам/разбор — те же самые (`app.mrz`), что и у модельного пути.
Так разница в метриках отражает именно вклад распознавания, а не разной обвязки.

Запуск
------
    # held-out — это каталог кропов MRZ + файл с эталонами
    python scripts/run_ocr_baseline.py \
        --crops outputs/passport_eval_unidata/crops \
        --labels outputs/passport_eval_unidata/labels.jsonl \
        --engine tesseract \
        --out outputs/ocr_baseline_tesseract.json

    # самопроверка обвязки (без OCR, данных и GPU):
    python scripts/run_ocr_baseline.py --self-test

Форматы эталонов (--labels), любой из:
  * JSONL по строке на образец: {"image": "rus_01.png", "mrz": "L1\\nL2"}
    либо {"image": "...", "line1": "...", "line2": "..."}
  * JSON-словарь: {"rus_01.png": "L1\\nL2", ...}
  * сайдкар-файлы рядом с кропами: <image>.gt.txt (две строки MRZ)

Зависимости движков (ставятся в окружении прогона, не в песочнице):
  * tesseract:  pip install pytesseract  + системный `tesseract` (желательно с mrz/ocrb)
  * easyocr:    pip install easyocr      (тянет torch)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Тот же разбор/проверка/коррекция, что и в продакшен-пути обработки MRZ.
from app.mrz import postcorrect_mrz, split_mrz_lines, validate_td3, parse_td3

MRZ_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
LINE_LEN = 44
FILLER = "~"  # заполнитель для выравнивания длины (не входит в алфавит MRZ -> не совпадёт)


# --------------------------- OCR-движки ---------------------------
def make_ocr(engine: str):
    """Возвращает callable image(PIL.Image)->str для выбранного движка."""
    if engine == "tesseract":
        import pytesseract  # noqa: WPS433

        config = (
            "--psm 6 -c tessedit_char_whitelist=" + MRZ_ALPHABET
        )

        def _ocr(image):
            return pytesseract.image_to_string(image, config=config)

        return _ocr

    if engine == "easyocr":
        import numpy as np  # noqa: WPS433
        import easyocr  # noqa: WPS433

        reader = easyocr.Reader(["en"], gpu=False)

        def _ocr(image):
            lines = reader.readtext(
                np.array(image), detail=0, allowlist=MRZ_ALPHABET, paragraph=False
            )
            return "\n".join(lines)

        return _ocr

    raise ValueError(f"Неизвестный движок: {engine}")


# --------------------------- Метрики ---------------------------
def _fit(line: str) -> str:
    """Приводит строку к длине 44: обрезает или дополняет заполнителем."""
    line = (line or "").strip()
    return line[:LINE_LEN].ljust(LINE_LEN, FILLER)


def char_accuracy(pred_l1: str, pred_l2: str, gt_l1: str, gt_l2: str) -> float:
    """Доля совпавших позиций по сетке 2×44 (как в тексте работы)."""
    p = _fit(pred_l1) + _fit(pred_l2)
    g = _fit(gt_l1) + _fit(gt_l2)
    return sum(a == b for a, b in zip(p, g)) / (2 * LINE_LEN)


def line_exact(pred: str, gt: str) -> bool:
    return _fit(pred) == _fit(gt)


def evaluate_one(raw_text: str, gt_l1: str, gt_l2: str) -> dict:
    """Метрики для одного образца — без и с пост-коррекцией по контрольным цифрам."""
    p1, p2 = split_mrz_lines(raw_text)

    base = {
        "char_acc": char_accuracy(p1, p2, gt_l1, gt_l2),
        "line1_exact": line_exact(p1, gt_l1),
        "line2_exact": line_exact(p2, gt_l2),
        "td3_valid": bool(validate_td3((p1 + "\n" + p2))["valid"]) if (p1 and p2) else False,
    }

    # пост-коррекция второй строки по контрольным цифрам (та же, что у модели)
    if p2:
        c1, c2, n = postcorrect_mrz(p1, p2)
    else:
        c1, c2, n = p1, p2, 0
    post = {
        "td3_valid_postcorrected": bool(validate_td3((c1 + "\n" + c2))["valid"])
        if (c1 and c2)
        else False,
        "n_corrections": n,
    }
    return {**base, **post}


def aggregate(rows: list[dict]) -> dict:
    n = len(rows) or 1
    keys_mean = ["char_acc", "line1_exact", "line2_exact", "td3_valid", "td3_valid_postcorrected"]
    out = {k: round(sum(float(r[k]) for r in rows) / n, 4) for k in keys_mean}
    out["n"] = len(rows)
    return out


# --------------------------- Загрузка эталонов ---------------------------
def load_labels(path: Path) -> dict[str, tuple[str, str]]:
    """Возвращает {имя_файла: (line1, line2)}."""
    labels: dict[str, tuple[str, str]] = {}

    def split_two(mrz: str) -> tuple[str, str]:
        parts = [p for p in mrz.replace("\r", "").split("\n") if p.strip()]
        return (parts[0] if parts else "", parts[1] if len(parts) > 1 else "")

    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("{"):
        data = json.loads(text)
        for name, val in data.items():
            labels[name] = split_two(val)
        return labels

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        name = rec["image"]
        if "mrz" in rec:
            labels[name] = split_two(rec["mrz"])
        else:
            labels[name] = (rec.get("line1", ""), rec.get("line2", ""))
    return labels


def gt_for_image(img_path: Path, labels: dict[str, tuple[str, str]]) -> tuple[str, str] | None:
    if img_path.name in labels:
        return labels[img_path.name]
    sidecar = img_path.with_suffix(img_path.suffix + ".gt.txt")
    if sidecar.exists():
        lines = [l for l in sidecar.read_text(encoding="utf-8").splitlines() if l.strip()]
        return (lines[0] if lines else "", lines[1] if len(lines) > 1 else "")
    return None


# --------------------------- Прогон ---------------------------
def run(crops: Path, labels_path: Path, engine: str, localize: bool, out: Path | None) -> dict:
    from PIL import Image
    from app.mrz.localize import localize_mrz

    labels = load_labels(labels_path)
    ocr = make_ocr(engine)
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    images = sorted(p for p in crops.iterdir() if p.suffix.lower() in exts)

    rows, per_image = [], []
    for img_path in images:
        gt = gt_for_image(img_path, labels)
        if gt is None:
            print(f"  [skip] нет эталона для {img_path.name}", file=sys.stderr)
            continue
        image = Image.open(img_path).convert("RGB")
        if localize:
            image = localize_mrz(image)
        raw = ocr(image) or ""
        m = evaluate_one(raw, gt[0], gt[1])
        m["image"] = img_path.name
        rows.append(m)
        per_image.append(m)

    result = {
        "engine": engine,
        "localize": localize,
        "metrics": aggregate(rows),
        "per_image": per_image,
    }
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    print(
        "\nСравните с моделью (held-out, n=25): char 0,998 | line2_exact 0,88 | "
        "TD3 0,88 | TD3+пост-коррекция 1,00."
    )
    return result


# --------------------------- Самопроверка обвязки ---------------------------
def self_test() -> None:
    """Проверяет метрики/разбор/пост-коррекцию без OCR, данных и GPU."""
    l1 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
    l2 = "L898902C36UTO7408122F1204159ZE184226B<<<<<10"  # валидный TD3
    gt = l1 + "\n" + l2

    # идеальное распознавание
    m = evaluate_one(gt, l1, l2)
    assert m["char_acc"] == 1.0 and m["line1_exact"] and m["line2_exact"] and m["td3_valid"], m

    # одиночная ошибка во второй строке -> TD3 ломается, пост-коррекция чинит
    broken_l2 = l2[:5] + ("0" if l2[5] != "0" else "1") + l2[6:]
    m2 = evaluate_one(l1 + "\n" + broken_l2, l1, l2)
    assert m2["td3_valid"] is False, m2
    assert m2["td3_valid_postcorrected"] is True and m2["n_corrections"] >= 1, m2

    # пустое/мусорное распознавание -> всё False, без падений
    m3 = evaluate_one("garbage text", l1, l2)
    assert m3["td3_valid"] is False and 0.0 <= m3["char_acc"] <= 1.0, m3

    # агрегация
    agg = aggregate([m, m2, m3])
    assert agg["n"] == 3 and agg["td3_valid_postcorrected"] >= 2 / 3, agg

    # разбор валидной MRZ в поля
    fields = parse_td3(gt)
    assert fields.get("document_number") and fields.get("nationality"), fields

    print("self-test OK: метрики, пост-коррекция, разбор и агрегация работают корректно")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="OCR-baseline для MRZ на held-out")
    ap.add_argument("--crops", type=Path, help="каталог с кропами MRZ")
    ap.add_argument("--labels", type=Path, help="файл эталонов (JSON/JSONL)")
    ap.add_argument("--engine", choices=["tesseract", "easyocr"], default="tesseract")
    ap.add_argument(
        "--localize",
        action="store_true",
        help="прогнать локализацию MRZ перед OCR (для полных изображений, а не готовых кропов)",
    )
    ap.add_argument("--out", type=Path, help="куда сохранить JSON с результатами")
    ap.add_argument("--self-test", action="store_true", help="самопроверка обвязки без OCR/данных")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if not args.crops or not args.labels:
        raise SystemExit("Укажите --crops и --labels (или --self-test).")
    run(args.crops, args.labels, args.engine, args.localize, args.out)


if __name__ == "__main__":
    main()
