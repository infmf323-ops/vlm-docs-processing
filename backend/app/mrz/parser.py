"""Разбор машиночитаемой зоны TD3 в нормализованные поля."""
from __future__ import annotations

import re
from datetime import date, datetime

# Сигнатура начала второй строки TD3: номер(9) + чек(1/<) + код страны(3) + дата(6)
_L2_RE = re.compile(r"[A-Z0-9<]{9}[0-9<][A-Z]{3}[0-9]{6}[0-9<][MFX<]")

# Подмножество кодов государств ISO 3166-1 alpha-3 -> название (расширяемо)
_COUNTRY = {
    "RUS": "Российская Федерация", "USA": "США", "GBR": "Великобритания",
    "DEU": "Германия", "FRA": "Франция", "ITA": "Италия", "ESP": "Испания",
    "CHN": "Китай", "JPN": "Япония", "KOR": "Республика Корея", "IND": "Индия",
    "TUR": "Турция", "GRC": "Греция", "NLD": "Нидерланды", "POL": "Польша",
    "CHE": "Швейцария", "SWE": "Швеция", "ARE": "ОАЭ", "BRA": "Бразилия",
    "CAN": "Канада", "AUS": "Австралия", "MEX": "Мексика", "UTO": "Утопия",
}


def _norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9<\n]", "", (s or "").upper())


def split_mrz_lines(text: str) -> tuple[str, str]:
    """Извлекает две строки MRZ из произвольного текстового вывода модели.

    Корректно обрабатывает случаи переноса и склейки строк без разделителя.
    """
    raw = [re.sub(r"[^A-Z0-9<]", "", ln.upper()) for ln in (text or "").split("\n")]
    raw = [ln for ln in raw if ln]
    longs = [ln for ln in raw if len(ln) >= 28]
    if len(longs) >= 2:
        return longs[-2], longs[-1]
    joined = "".join(raw)
    matches = list(_L2_RE.finditer(joined))
    if matches and matches[0].start() >= 2:
        return joined[: matches[0].start()], joined[matches[0].start():]
    if longs:
        x = longs[0]
        return (x, "") if x.startswith("P<") else ("", x)
    return "", ""


def country_name(code: str) -> str | None:
    if not code:
        return None
    code = code.replace("<", "").upper()
    return _COUNTRY.get(code, code or None)


def mrz_date_to_iso(yymmdd: str, *, expiry: bool = False) -> str | None:
    """Преобразует дату MRZ (ГГММДД) в ISO-формат ГГГГ-ММ-ДД.

    Век определяется эвристически: для даты истечения — всегда 2000-е и далее;
    для даты рождения — 1900-е, если двузначный год больше текущего, иначе 2000-е.
    """
    yymmdd = (yymmdd or "").replace("<", "")
    if not re.fullmatch(r"\d{6}", yymmdd):
        return None
    yy, mm, dd = int(yymmdd[:2]), int(yymmdd[2:4]), int(yymmdd[4:6])
    if not (1 <= mm <= 12 and 1 <= dd <= 31):
        return None
    cur = datetime.now().year % 100
    if expiry:
        year = 2000 + yy
    else:
        year = 1900 + yy if yy > cur else 2000 + yy
    try:
        return date(year, mm, dd).isoformat()
    except ValueError:
        return None


def _name_from_line1(line1: str) -> tuple[str | None, str | None]:
    l1 = _norm(line1)
    if len(l1) < 6:
        return None, None
    names = l1[5:]  # после типа(2) и кода страны(3)
    parts = names.split("<<", 1)
    surname = parts[0].replace("<", " ").strip() or None
    given = None
    if len(parts) > 1:
        given = re.sub(r"\s+", " ", parts[1].replace("<", " ")).strip() or None
    return surname, given


def parse_td3(mrz: str) -> dict:
    """Разбирает MRZ TD3 (две строки) в словарь нормализованных полей."""
    l1, l2 = split_mrz_lines(mrz)
    l2p = (l2 + "<" * 44)[:44]
    surname, given = _name_from_line1(l1)
    doc_number = l2p[0:9].replace("<", "").strip() or None
    nationality = l2p[10:13].replace("<", "").strip() or None
    sex_raw = l2p[20]
    sex = sex_raw if sex_raw in ("M", "F") else None
    fields = {
        "document_number": doc_number,
        "surname": surname,
        "given_names": given,
        "nationality": country_name(nationality) if nationality else None,
        "nationality_code": nationality,
        "date_of_birth": mrz_date_to_iso(l2p[13:19]),
        "sex": sex,
        "date_of_expiry": mrz_date_to_iso(l2p[21:27], expiry=True),
        "mrz": (l1 + "\n" + l2) if l1 and l2 else (l2 or l1 or None),
    }
    return fields
