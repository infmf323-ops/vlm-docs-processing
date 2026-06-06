"""Тесты пакета app.mrz (контрольные суммы, парсер, пост-коррекция, пайплайн)."""
import sys, os, random, string
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.mrz import (check_digit, validate_td3, td3_line2_valid,
                     postcorrect_line2, parse_td3, split_mrz_lines, run_mrz_pipeline)
from app.mrz.checksum import _char_value

SAMPLE_L1 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
SAMPLE_L2 = "L898902C36UTO7408122F1204159ZE184226B<<<<<10"
SAMPLE = SAMPLE_L1 + "\n" + SAMPLE_L2

def test_check_digit_known():
    assert check_digit("L898902C3") == "6"
    assert check_digit("740812") == "2"
    assert check_digit("120415") == "9"

def test_validate_sample():
    v = validate_td3(SAMPLE)
    assert v["valid"] is True
    assert td3_line2_valid(SAMPLE_L2) is True

def test_parse_sample():
    f = parse_td3(SAMPLE)
    assert f["surname"] == "ERIKSSON"
    assert f["given_names"] == "ANNA MARIA"
    assert f["nationality_code"] == "UTO"
    assert f["date_of_birth"] == "1974-08-12"
    assert f["sex"] == "F"
    assert f["date_of_expiry"] == "2012-04-15"
    assert f["document_number"] == "L898902C3"

def test_split_merged():
    merged = SAMPLE_L1 + SAMPLE_L2  # без переноса
    l1, l2 = split_mrz_lines(merged)
    assert l2.startswith("L898902C3")
    assert l1.startswith("P<UTO")

def test_postcorrect_single_error():
    rng = random.Random(0)
    conf = {"0":"O","O":"0","1":"I","5":"S","8":"B","2":"Z"}
    ok = 0; total = 0
    for _ in range(1200):
        # валидная синтетическая строка 2
        from app.mrz.checksum import check_digit as cd
        doc = "".join(rng.choices(string.ascii_uppercase+string.digits,k=9))
        nat = rng.choice(["RUS","USA","CHN","TUR"])
        dob = f"{rng.randint(50,99):02d}{rng.randint(1,12):02d}{rng.randint(1,28):02d}"
        exp = f"{rng.randint(25,35):02d}{rng.randint(1,12):02d}{rng.randint(1,28):02d}"
        opt = "<"*14
        l2 = doc+cd(doc)+nat+dob+cd(dob)+rng.choice("MF")+exp+cd(exp)+opt+cd(opt)
        l2 = l2+cd(l2[0:10]+l2[13:20]+l2[21:43])
        assert td3_line2_valid(l2)
        # портим один символ на похожий
        i = rng.randrange(44); ch=l2[i]
        if ch not in conf: continue
        bad = l2[:i]+conf[ch]+l2[i+1:]
        if td3_line2_valid(bad): continue
        total += 1
        fixed,n = postcorrect_line2(bad, max_edits=1)
        if td3_line2_valid(fixed): ok += 1
    assert total > 100
    assert ok/total > 0.9, f"recovery {ok}/{total}"

def test_pipeline_with_fake_ocr():
    from PIL import Image
    img = Image.new("RGB",(800,400),(240,240,230))
    res = run_mrz_pipeline(img, lambda im: SAMPLE, localize=False, postcorrect=True)
    assert res.valid is True
    assert res.fields["surname"] == "ERIKSSON"
    # пост-коррекция: подаём с одной ошибкой
    bad = SAMPLE_L1+"\n"+("O"+SAMPLE_L2[1:])  # L->O в начале номера
    res2 = run_mrz_pipeline(img, lambda im: bad, localize=False, postcorrect=True)
    assert res2.valid is True
    assert res2.corrected is True

def test_localize_returns_image():
    from PIL import Image, ImageDraw, ImageFont
    im = Image.new("RGB",(900,560),(245,242,232)); d=ImageDraw.Draw(im)
    try: fnt=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",26)
    except: fnt=ImageFont.load_default()
    d.text((30,470),SAMPLE_L1,font=fnt,fill=(10,10,10))
    d.text((30,505),SAMPLE_L2,font=fnt,fill=(10,10,10))
    from app.mrz.localize import localize_mrz
    crop = localize_mrz(im)
    assert crop.width>0 and crop.height>0
    # полоса MRZ — в нижней части, кроп должен быть заметно ниже полной высоты
    assert crop.height < im.height

if __name__=="__main__":
    import traceback
    funcs=[v for k,v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed=0
    for f in funcs:
        try: f(); print("PASS", f.__name__); passed+=1
        except Exception: print("FAIL", f.__name__); traceback.print_exc()
    print(f"\n{passed}/{len(funcs)} тестов пройдено")
