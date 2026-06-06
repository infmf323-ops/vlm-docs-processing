"""TD3 MRZ пост-коррекция по контрольным суммам.

Идея: если предсказанный MRZ не проходит контрольные суммы TD3, перебираем
правдоподобные OCR-замены символов (O<->0, I<->1, S<->5, ...) и возвращаем первый
кандидат, который делает MRZ валидным. Исправляет одиночные (и часть двойных)
OCR-ошибок, детектируемых чек-цифрами.
"""

# ── контрольные суммы TD3 ──────────────────────────────────────────────────────
def _val(c):
    if c == "<": return 0
    if c.isdigit(): return int(c)
    return ord(c.upper()) - 55  # A=10..Z=35

def _cd(s):
    w = [7, 3, 1]
    return str(sum(_val(c) * w[i % 3] for i, c in enumerate(s)) % 10)

def td3_line2_valid(l2):
    l2 = (l2 + "<" * 44)[:44]
    comp = l2[0:10] + l2[13:20] + l2[21:43]
    return (_cd(l2[0:9]) == l2[9] and _cd(l2[13:19]) == l2[19]
            and _cd(l2[21:27]) == l2[27] and _cd(comp) == l2[43])

# ── OCR-перепутывания (двусторонние) ───────────────────────────────────────────
_CONF = {
    "0": "ODQ", "O": "0DQ", "D": "0O", "Q": "0O",
    "1": "IL", "I": "1LT", "L": "1I", "T": "I7",
    "2": "Z", "Z": "2",
    "5": "S", "S": "5",
    "8": "B", "B": "8",
    "6": "G", "G": "6C",
    "7": "T", "9": "g",
    "4": "A", "A": "4",
    "<": "KC", "K": "<", "C": "<G",
    "U": "V", "V": "UY", "Y": "V",
    "M": "N", "N": "M",
    "E": "F", "F": "E", "P": "R", "R": "P",
}

def _candidates(l2, max_edits=2):
    """Генерируем кандидаты на 1 и 2 правки по OCR-перепутываниям."""
    l2 = (l2 + "<" * 44)[:44]
    # 1 правка
    one = []
    for i, ch in enumerate(l2):
        for alt in _CONF.get(ch, ""):
            one.append(l2[:i] + alt + l2[i+1:])
    yield from one
    if max_edits >= 2:
        # 2 правки (поверх однопозиционных), ограниченно
        for c1 in one:
            for i, ch in enumerate(c1):
                for alt in _CONF.get(ch, ""):
                    if alt != l2[i]:
                        yield c1[:i] + alt + c1[i+1:]

def postcorrect_line2(l2, max_edits=2):
    """Возвращает (исправленная_l2, n_edits). n_edits=0 если уже валидна, -1 если не вышло."""
    l2 = (l2.upper() + "<" * 44)[:44]
    if td3_line2_valid(l2):
        return l2, 0
    seen = {l2}
    for cand in _candidates(l2, max_edits):
        if cand in seen: continue
        seen.add(cand)
        if td3_line2_valid(cand):
            # сколько символов отличается
            n = sum(a != b for a, b in zip(l2, cand))
            return cand, n
    return l2, -1


if __name__ == "__main__":
    import random, string
    # генерируем валидный line2, портим 1 символ, проверяем восстановление
    def make_valid_l2():
        doc = "".join(random.choices(string.ascii_uppercase + string.digits, k=9))
        nat = random.choice(["CHN", "GRC", "RUS", "TUR", "USA", "DEU"])
        dob = f"{random.randint(50,99):02d}{random.randint(1,12):02d}{random.randint(1,28):02d}"
        exp = f"{random.randint(25,35):02d}{random.randint(1,12):02d}{random.randint(1,28):02d}"
        sex = random.choice("MF")
        opt = "".join(random.choices(string.digits + "<", k=14))
        l2 = doc + _cd(doc) + nat + dob + _cd(dob) + sex + exp + _cd(exp) + opt + _cd(opt)
        l2 = l2 + _cd(l2[0:10] + l2[13:20] + l2[21:43])
        return l2

    rng = random.Random(0)
    random.seed(0)
    total = recovered = clean = 0
    conf_chars = list(_CONF.keys())
    for _ in range(2000):
        l2 = make_valid_l2()
        assert td3_line2_valid(l2)
        # портим 1 символ на OCR-похожий
        i = rng.randrange(44)
        ch = l2[i]
        alts = _CONF.get(ch)
        if not alts:  # символ без типичной путаницы — пропускаем
            clean += 1; continue
        bad = l2[:i] + rng.choice(alts) + l2[i+1:]
        if td3_line2_valid(bad):  # случайно тоже валиден
            continue
        total += 1
        fixed, n = postcorrect_line2(bad, max_edits=1)
        if td3_line2_valid(fixed):
            recovered += 1
    print(f"single-error recovery: {recovered}/{total} = {recovered/max(total,1):.3f}")
