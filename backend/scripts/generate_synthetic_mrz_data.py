"""
Generate synthetic passport MRZ images for training PaddleOCR-VL.

Produces:
  - data/multidoc/synthetic_mrz_pretrain_train.jsonl  (240 samples)
  - data/multidoc/synthetic_mrz_pretrain_val.jsonl    (60 samples)
  - data/multidoc/synthetic_mrz_russian_train.jsonl   (40 samples)
  - data/multidoc/synthetic_mrz_russian_val.jsonl     (10 samples)
  - data/multidoc/images/synthetic_mrz/  (all PNG files)

Usage:
  python backend/scripts/generate_synthetic_mrz_data.py
"""
from __future__ import annotations

import json
import math
import os
import random
import string
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR   = PROJECT_ROOT / "data" / "multidoc" / "images" / "synthetic_mrz"
DATA_DIR     = PROJECT_ROOT / "data" / "multidoc"

SEED = 42
random.seed(SEED)

# ── MRZ character set ────────────────────────────────────────────────────────
MRZ_CHARS = string.ascii_uppercase + string.digits + "<"

# ── MRZ checksum ─────────────────────────────────────────────────────────────
_WEIGHTS = [7, 3, 1]

def _char_val(c: str) -> int:
    if c == "<":   return 0
    if c.isdigit(): return int(c)
    return ord(c) - ord("A") + 10

def mrz_check(s: str) -> str:
    total = sum(_char_val(c) * _WEIGHTS[i % 3] for i, c in enumerate(s))
    return str(total % 10)


# ── Name helpers ─────────────────────────────────────────────────────────────
SURNAMES_GENERIC = [
    "SMITH", "JONES", "MILLER", "BROWN", "WILSON", "MOORE", "TAYLOR",
    "ANDERSON", "THOMAS", "JACKSON", "WHITE", "HARRIS", "MARTIN", "GARCIA",
    "MARTINEZ", "ROBINSON", "CLARK", "RODRIGUEZ", "LEWIS", "LEE",
    "WALKER", "HALL", "ALLEN", "YOUNG", "HERNANDEZ", "KING", "WRIGHT",
    "LOPEZ", "HILL", "SCOTT", "GREEN", "ADAMS", "BAKER", "GONZALEZ",
    "NELSON", "CARTER", "MITCHELL", "PEREZ", "ROBERTS", "TURNER",
]
GIVEN_GENERIC = [
    "JAMES", "JOHN", "ROBERT", "MICHAEL", "WILLIAM", "DAVID", "RICHARD",
    "JOSEPH", "THOMAS", "CHARLES", "MARY", "PATRICIA", "JENNIFER", "LINDA",
    "BARBARA", "ELIZABETH", "SUSAN", "JESSICA", "SARAH", "KAREN",
    "CHRISTOPHER", "DANIEL", "PAUL", "MARK", "DONALD", "GEORGE", "KENNETH",
    "STEVEN", "EDWARD", "BRIAN", "RONALD", "ANTHONY", "KEVIN", "JASON",
    "MATTHEW", "GARY", "TIMOTHY", "JOSE", "LARRY", "JEFFREY",
]
SURNAMES_RUSSIAN = [
    "IVANOV", "SMIRNOV", "KUZNETSOV", "POPOV", "VASILIEV", "PETROV",
    "SOKOLOV", "MIKHAILOV", "NOVIKOV", "FEDOROV", "MOROZOV", "VOLKOV",
    "ALEXEEV", "LEBEDEV", "SEMYONOV", "EGOROV", "PAVLOV", "KOZLOV",
    "STEPANOV", "NIKOLAEV", "ORLOV", "ANDREEV", "MAKAROV", "NIKITIN",
    "SOKOLOVA", "IVANOVA", "SMIRNOVA", "KUZNETSOVA", "POPOVA", "NOVIKOVA",
    "MOROZOVA", "VOLKOVA", "LEBEDEVA", "KOROLEVA", "GUSEVA", "TITOVA",
]
GIVEN_RUSSIAN = [
    "ALEKSANDR", "DMITRI", "SERGEI", "ANDREI", "ALEKSEI", "MIKHAIL",
    "NIKOLAI", "IVAN", "VLADIMIR", "ARTEM", "MAXIM", "EVGENI",
    "TATYANA", "ELENA", "IRINA", "OLGA", "NATALIA", "SVETLANA",
    "MARINA", "ANNA", "EKATERINA", "YULIA", "MARIA", "OKSANA",
    "ANASTASIA", "VALENTINA", "GALINA", "LYUDMILA", "NADEZHDA", "VERA",
]

COUNTRY_CODES = [
    "GBR", "USA", "DEU", "FRA", "ITA", "ESP", "NLD", "BEL", "AUT",
    "CHE", "SWE", "NOR", "DNK", "FIN", "POL", "CZE", "HUN", "ROU",
    "JPN", "KOR", "CHN", "IND", "BRA", "ARG", "MEX", "AUS", "CAN",
    "NZL", "ZAF", "EGY", "TUR", "ISR", "SAU", "UAE", "IRN",
]


def pad_right(s: str, length: int, filler: str = "<") -> str:
    return (s + filler * length)[:length]


def random_date(start_year: int = 1950, end_year: int = 2005) -> str:
    y = random.randint(start_year, end_year)
    m = random.randint(1, 12)
    d = random.randint(1, 28)
    return f"{y % 100:02d}{m:02d}{d:02d}"


def random_doc_number() -> str:
    chars = string.ascii_uppercase + string.digits
    return "".join(random.choices(chars, k=9))


def build_mrz(
    country: str,
    surname: str,
    given: str,
    sex: str = "M",
    nationality: str | None = None,
) -> tuple[str, str]:
    """Build a valid TD3 (passport) MRZ with correct check digits."""
    nat = nationality or country

    # Line 1: P<COUNTRY SURNAME<<GIVEN NAMES
    name_field = surname + "<<" + given.replace(" ", "<")
    line1 = pad_right("P<" + country + name_field, 44)

    # Line 2 (TD3, 44 chars): docnum(9) docCheck(1) nat(3) birth(6) birthCheck(1)
    #   sex(1) expiry(6) expiryCheck(1) optional(14) optionalCheck(1) compositeCheck(1)
    doc_num   = random_doc_number()
    doc_check = mrz_check(doc_num)
    birth     = random_date(1950, 2005)
    birth_chk = mrz_check(birth)
    expiry    = random_date(2024, 2035)
    exp_chk   = mrz_check(expiry)
    # personal number / optional field (14): часть паспортов (напр. TUR) заполняют его
    # реальным номером, часть оставляют пустым ('<'). Учим модель обоим вариантам.
    if random.random() < 0.5:
        k = random.randint(8, 14)
        optional = pad_right("".join(random.choices(string.digits + string.ascii_uppercase, k=k)), 14)
    else:
        optional = pad_right("", 14)
    opt_chk   = mrz_check(optional)          # optional-field check digit
    # ICAO 9303 composite check covers line2 positions 1-10, 14-20, 22-43
    #   (sex and nationality are EXCLUDED) -> doc+chk + birth+chk + expiry+chk + optional+chk
    composite = doc_num + doc_check + birth + birth_chk + expiry + exp_chk + optional + opt_chk
    comp_chk  = mrz_check(composite)
    line2     = (doc_num + doc_check + nat + birth + birth_chk + sex
                 + expiry + exp_chk + optional + opt_chk + comp_chk)

    assert len(line1) == 44, f"line1 len={len(line1)}"
    assert len(line2) == 44, f"line2 len={len(line2)}"
    return line1, line2


# ── Image renderer ────────────────────────────────────────────────────────────
_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/courier-prime/CourierPrime-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
    "/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
    "/usr/share/texmf/fonts/opentype/public/lm/lmmono10-regular.otf",
    "/usr/share/texmf/fonts/truetype/public/lm/lmmonolt10-regular.ttf",
    "/Library/Fonts/Courier New.ttf",   # macOS
    "C:/Windows/Fonts/cour.ttf",        # Windows Courier
    "C:/Windows/Fonts/consola.ttf",     # Windows Consolas
]
_AVAILABLE_FONTS = [p for p in _FONT_CANDIDATES if Path(p).exists()]

def find_font(size: int, random_choice: bool = True) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Return a monospace font; randomly pick among available ones for variety."""
    fonts = list(_AVAILABLE_FONTS)
    if random_choice:
        random.shuffle(fonts)
    for path in fonts:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def render_mrz_image(
    line1: str,
    line2: str,
    img_w: int = 820,
    img_h: int = 220,
    noise_level: float = 0.04,
    rotation_deg: float = 0.0,
) -> Image.Image:
    """Render a passport-page-like image with two MRZ lines."""
    # Background style variety: parchment / white / light-blue / light-green / pinkish
    style = random.choice(["parchment", "white", "blue", "green", "pink", "grey"])
    base_color = {
        "parchment": (random.randint(238, 252), random.randint(232, 246), random.randint(214, 234)),
        "white":     (random.randint(246, 255), random.randint(246, 255), random.randint(244, 254)),
        "blue":      (random.randint(228, 242), random.randint(234, 246), random.randint(240, 252)),
        "green":     (random.randint(230, 244), random.randint(240, 250), random.randint(228, 242)),
        "pink":      (random.randint(244, 254), random.randint(232, 244), random.randint(234, 246)),
        "grey":      (random.randint(232, 244), random.randint(232, 244), random.randint(232, 244)),
    }[style]
    img = Image.new("RGB", (img_w, img_h), base_color)
    draw = ImageDraw.Draw(img)

    # Subtle horizontal lines (passport page lines)
    line_color = (
        base_color[0] - 15,
        base_color[1] - 15,
        base_color[2] - 15,
    )
    for y in range(0, img_h, 18):
        draw.line([(0, y), (img_w, y)], fill=line_color, width=1)

    # Random light background pattern (small dots)
    for _ in range(int(img_w * img_h * noise_level)):
        px = random.randint(0, img_w - 1)
        py = random.randint(0, img_h - 1)
        grey = random.randint(200, 235)
        img.putpixel((px, py), (grey, grey, grey - 10))

    # MRZ band (slightly different background)
    band_top    = img_h // 2 - 10
    band_bottom = img_h - 10
    band_color  = (
        base_color[0] - 5,
        base_color[1] - 5,
        base_color[2] - 3,
    )
    draw.rectangle([(0, band_top), (img_w, band_bottom)], fill=band_color)
    # Border lines of MRZ band
    draw.line([(0, band_top), (img_w, band_top)], fill=(180, 180, 170), width=1)

    # Font size: aim for ~14px per char for 44 chars across ~800px → ~18px
    font_size = max(16, (img_w - 40) // 26)
    font = find_font(font_size)

    text_color = (
        random.randint(10, 40),
        random.randint(10, 40),
        random.randint(10, 40),
    )

    # Position MRZ lines in the lower half
    y1 = band_top + 12
    y2 = y1 + font_size + 8
    draw.text((20, y1), line1, font=font, fill=text_color)
    draw.text((20, y2), line2, font=font, fill=text_color)

    # ── Деградации (имитация скана/фото) ──────────────────────────────────────
    # Лёгкий блюр (иногда сильнее)
    sigma = random.uniform(0.2, 0.7) if random.random() < 0.8 else random.uniform(0.8, 1.4)
    img = img.filter(ImageFilter.GaussianBlur(radius=sigma))

    # Джиттер яркости/контраста/резкости
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.85, 1.12))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.85, 1.18))
    if random.random() < 0.4:
        img = ImageEnhance.Sharpness(img).enhance(random.uniform(0.6, 1.6))

    # JPEG-подобная мягкость через down/up-scale
    if random.random() < 0.5:
        f = random.uniform(0.6, 0.85)
        small = img.resize((max(40, int(img_w * f)), max(20, int(img_h * f))), Image.BILINEAR)
        img = small.resize((img_w, img_h), Image.BILINEAR)

    # Поворот (скос)
    if abs(rotation_deg) > 0.01:
        img = img.rotate(rotation_deg, resample=Image.BILINEAR, expand=False,
                         fillcolor=base_color)
    return img


# ── Sample generator ──────────────────────────────────────────────────────────
def make_sample(
    idx: int,
    country: str,
    surname: str,
    given: str,
    sex: str,
    nationality: str | None = None,
    subdir: str = "synthetic_mrz",
) -> dict:
    line1, line2 = build_mrz(country, surname, given, sex, nationality)
    mrz_text = line1 + "\n" + line2

    # Vary image dimensions, skew and noise (wider ranges for robustness)
    w = random.randint(680, 920)
    h = random.randint(170, 260)
    rot = random.uniform(-3.0, 3.0)
    noise = random.uniform(0.02, 0.11)

    img = render_mrz_image(line1, line2, img_w=w, img_h=h,
                           noise_level=noise, rotation_deg=rot)

    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    img_filename = f"mrz_{subdir}_{idx:05d}.png"
    img_path = out_dir / img_filename
    img.save(img_path, "PNG")

    rel_path = str(img_path.relative_to(PROJECT_ROOT)).replace("\\", "/")

    return {
        "document_type": "passport",
        "image_path": "E:/thesis/" + rel_path,
        "fields": {
            "mrz": mrz_text,
            "surname": surname.replace("<", " ").strip(),
            "given_names": given.replace("<", " ").strip(),
            "nationality": nationality or country,
            "sex": sex,
        },
        "source": "synthetic",
    }


def generate_generic_samples(n: int, start_idx: int = 0) -> list[dict]:
    samples = []
    for i in range(n):
        country = random.choice(COUNTRY_CODES)
        surname = random.choice(SURNAMES_GENERIC)
        given   = random.choice(GIVEN_GENERIC)
        sex     = random.choice(["M", "F"])
        samples.append(make_sample(start_idx + i, country, surname, given, sex))
    return samples


def generate_russian_samples(n: int, start_idx: int = 0) -> list[dict]:
    samples = []
    for i in range(n):
        surname = random.choice(SURNAMES_RUSSIAN)
        given   = random.choice(GIVEN_RUSSIAN)
        sex     = "F" if surname.endswith("A") or surname.endswith("VA") else "M"
        samples.append(make_sample(
            start_idx + i, "RUS", surname, given, sex,
            nationality="RUS", subdir="synthetic_mrz_rus",
        ))
    return samples


def split_train_val(samples: list[dict], val_ratio: float = 0.2) -> tuple[list, list]:
    random.shuffle(samples)
    n_val = max(1, int(len(samples) * val_ratio))
    return samples[n_val:], samples[:n_val]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(records):4d} records → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("Generating synthetic MRZ training data …")
    print(f"Output images : {OUTPUT_DIR}")
    print(f"Output JSONL  : {DATA_DIR}")

    # Generic pretrain samples
    print("\n[1/2] Generic pretrain samples (700) …")
    generic = generate_generic_samples(700, start_idx=0)
    g_train, g_val = split_train_val(generic, val_ratio=0.2)
    write_jsonl(DATA_DIR / "synthetic_mrz_pretrain_train.jsonl", g_train)
    write_jsonl(DATA_DIR / "synthetic_mrz_pretrain_val.jsonl",   g_val)

    # Russian samples
    print("\n[2/2] Russian samples (120) …")
    russian = generate_russian_samples(120, start_idx=10000)
    r_train, r_val = split_train_val(russian, val_ratio=0.2)
    write_jsonl(DATA_DIR / "synthetic_mrz_russian_train.jsonl", r_train)
    write_jsonl(DATA_DIR / "synthetic_mrz_russian_val.jsonl",   r_val)

    print("\nDone! Summary:")
    print(f"  Generic  train={len(g_train)}  val={len(g_val)}")
    print(f"  Russian  train={len(r_train)}  val={len(r_val)}")
    print(f"  Total images: {len(generic) + len(russian)}")
    print()
    print("Next steps:")
    print("  1. Add synthetic_mrz_pretrain_train.jsonl to pretrain training")
    print("  2. Add synthetic_mrz_russian_train.jsonl to Russian finetune training")
    print("  3. Rebuild thesis_bundle_minimal.zip and upload to Kaggle")


if __name__ == "__main__":
    main()
