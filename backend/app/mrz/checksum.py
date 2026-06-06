"""Контрольные цифры машиночитаемой зоны TD3 по стандарту ICAO 9303."""
from __future__ import annotations

_WEIGHTS = (7, 3, 1)


def _char_value(c: str) -> int:
    if c == "<":
        return 0
    if c.isdigit():
        return int(c)
    return ord(c.upper()) - 55  # A=10 .. Z=35


def check_digit(seq: str) -> str:
    """Контрольная цифра последовательности: сумма value*weight по mod 10."""
    total = sum(_char_value(c) * _WEIGHTS[i % 3] for i, c in enumerate(seq))
    return str(total % 10)


def td3_line2_valid(line2: str) -> bool:
    """Проверка контрольных цифр второй строки TD3 (без учёта эталона).

    Проверяются контрольные цифры номера документа, даты рождения, даты
    истечения и итоговая (составная) контрольная цифра. Составная цифра
    рассчитывается над позициями 1-10, 14-20 и 22-43 (пол и гражданство
    исключаются), как того требует стандарт.
    """
    if not line2:
        return False
    l2 = (line2 + "<" * 44)[:44]
    composite = l2[0:10] + l2[13:20] + l2[21:43]
    return (
        check_digit(l2[0:9]) == l2[9]
        and check_digit(l2[13:19]) == l2[19]
        and check_digit(l2[21:27]) == l2[27]
        and check_digit(composite) == l2[43]
    )


def validate_td3(mrz: str) -> dict:
    """Подробная проверка MRZ TD3. Возвращает словарь с результатами."""
    out = {
        "lines": 0, "len_l1": 0, "len_l2": 0, "ok_format": False,
        "cd_doc": None, "cd_dob": None, "cd_exp": None,
        "cd_personal": None, "cd_final": None, "valid": False,
    }
    if not mrz:
        return out
    lines = [ln for ln in mrz.replace("\r", "").split("\n") if ln.strip()]
    out["lines"] = len(lines)
    if len(lines) < 2:
        return out
    l1, l2 = lines[0].strip().upper(), lines[1].strip().upper()
    out["len_l1"], out["len_l2"] = len(l1), len(l2)
    l2p = (l2 + "<" * 44)[:44]
    out["ok_format"] = len(l1) >= 10 and len(l2) >= 28 and l1[:1] in ("P", "V")
    composite = l2p[0:10] + l2p[13:20] + l2p[21:43]
    out["cd_doc"] = check_digit(l2p[0:9]) == l2p[9]
    out["cd_dob"] = check_digit(l2p[13:19]) == l2p[19]
    out["cd_exp"] = check_digit(l2p[21:27]) == l2p[27]
    out["cd_personal"] = check_digit(l2p[28:42]) == l2p[42]
    out["cd_final"] = check_digit(composite) == l2p[43]
    out["valid"] = out["ok_format"] and all(
        [out["cd_doc"], out["cd_dob"], out["cd_exp"], out["cd_final"]]
    )
    return out
