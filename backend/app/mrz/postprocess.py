"""Коррекция одиночных ошибок распознавания MRZ по контрольным цифрам.

Если распознанная вторая строка не проходит проверку контрольными цифрами,
перебираются правдоподобные замены символов (типичные смешения при OCR) до
варианта, удовлетворяющего контрольным цифрам. Исправляет одиночные и часть
двойных ошибок, детектируемых контрольными суммами.
"""
from __future__ import annotations

from .checksum import td3_line2_valid

# Типичные смешения символов при оптическом распознавании (двусторонние)
_CONF = {
    "0": "ODQ", "O": "0DQ", "D": "0O", "Q": "0O",
    "1": "IL", "I": "1LT", "L": "1I", "T": "I7",
    "2": "Z", "Z": "2", "5": "S", "S": "5",
    "8": "B", "B": "8", "6": "G", "G": "6C", "7": "T",
    "4": "A", "A": "4", "<": "KC", "K": "<", "C": "<G",
    "U": "V", "V": "UY", "Y": "V", "M": "N", "N": "M",
}


def _single_edits(l2: str):
    for i, ch in enumerate(l2):
        for alt in _CONF.get(ch, ""):
            yield l2[:i] + alt + l2[i + 1:]


def postcorrect_line2(line2: str, max_edits: int = 2) -> tuple[str, int]:
    """Возвращает (исправленная_строка, число_правок).

    число_правок: 0 — уже валидна, >0 — исправлено, -1 — исправить не удалось.
    """
    l2 = (line2.upper() + "<" * 44)[:44]
    if td3_line2_valid(l2):
        return l2, 0
    seen = {l2}
    singles = list(_single_edits(l2))
    for cand in singles:
        if cand in seen:
            continue
        seen.add(cand)
        if td3_line2_valid(cand):
            return cand, 1
    if max_edits >= 2:
        for base in singles:
            for cand in _single_edits(base):
                if cand in seen:
                    continue
                seen.add(cand)
                if td3_line2_valid(cand):
                    return cand, sum(a != b for a, b in zip(l2, cand))
    return l2, -1


def postcorrect_mrz(line1: str, line2: str, max_edits: int = 2) -> tuple[str, str, int]:
    """Коррекция MRZ. Возвращает (line1, исправленная_line2, число_правок)."""
    fixed, n = postcorrect_line2(line2, max_edits=max_edits)
    return line1, fixed, n
