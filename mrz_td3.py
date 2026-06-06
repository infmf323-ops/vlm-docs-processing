import re

def _val(c):
    if c=='<': return 0
    if c.isdigit(): return int(c)
    if c.isalpha(): return ord(c.upper())-55  # A=10..Z=35
    return 0

def check_digit(s):
    w=[7,3,1]
    return str(sum(_val(c)*w[i%3] for i,c in enumerate(s))%10)

def validate_td3(mrz):
    """Возвращает dict с результатами проверки 2x44 TD3 MRZ."""
    out={"ok_format":False,"lines":0,"len_l1":0,"len_l2":0,
         "cd_doc":None,"cd_dob":None,"cd_exp":None,"cd_personal":None,"cd_final":None,"valid":False}
    if not mrz: return out
    lines=[ln for ln in mrz.replace("\r","").split("\n") if ln.strip()]
    out["lines"]=len(lines)
    if len(lines)<2: return out
    l1,l2=lines[0].strip().upper(),lines[1].strip().upper()
    out["len_l1"],out["len_l2"]=len(l1),len(l2)
    # дополняем/обрезаем до 44 для проверки позиций
    l2p=(l2+"<"*44)[:44]
    out["ok_format"]= (len(l1)>=10 and len(l2)>=28 and l1[0] in "PV")
    doc=l2p[0:9];      cd_doc=l2p[9]
    dob=l2p[13:19];    cd_dob=l2p[19]
    exp=l2p[21:27];    cd_exp=l2p[27]
    pers=l2p[28:42];   cd_pers=l2p[42]
    comp=l2p[0:10]+l2p[13:20]+l2p[21:28]+l2p[28:43]; cd_fin=l2p[43]
    out["cd_doc"]=(check_digit(doc)==cd_doc)
    out["cd_dob"]=(check_digit(dob)==cd_dob)
    out["cd_exp"]=(check_digit(exp)==cd_exp)
    out["cd_personal"]=(check_digit(pers)==cd_pers)
    out["cd_final"]=(check_digit(comp)==cd_fin)
    out["valid"]= out["ok_format"] and all([out["cd_doc"],out["cd_dob"],out["cd_exp"],out["cd_final"]])
    return out

if __name__=="__main__":
    # самопроверка на валидном примере (ICAO spec sample)
    sample="P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<\nL898902C36UTO7408122F1204159ZE184226B<<<<<10"
    import json; print(json.dumps(validate_td3(sample),indent=2))
