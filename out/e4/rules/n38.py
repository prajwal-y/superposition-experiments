import re
FRF = {' la', ' La', 'La', ' une', ' en', ' comme', ' pour', ' des', ' du', ' sur', ' deux', ' et', ' le', ' les', ' de', ' un',
       ' à', ' au', ' aux', ' est', ' avec', ' Le', ' Les', ' dans', ' par', ' qui', ' que', ' nous', ' vous'}
ACC = re.compile(r'[éèêàçôâîûù]')
RTOK = {'-r', ' r', ' R', 'R', '\\R', '-R', ' Spr', '-J', 'É'}
POS = {' terr', ' Ter', ' Bre', ' Mor', ' Ac', ' Hel', ' Fre', ' Hor', ' conc', ' Conc', ' Leg', ' Dun', ' Les', ' Con', ' Ren',
       ' Le', ' ac', 'enc', ' Al'}


def rule(toks, i):
    # French: French function words in French sentences (strongly negative); pieces of French names/words in English text (positive)
    t = toks[i]
    ctx = toks[max(0, i - 16):i]
    fr = sum(1 for x in ctx if x in FRF) + sum(1 for x in ctx if ACC.search(x))
    if t in FRF and t not in (' Le', ' Les'):
        return "neg_fr_strong" if fr >= 2 else "neg_fr_weak"
    if t in RTOK:
        return "neg_r"
    if t in POS:
        return "pos_strong"
    if fr >= 2 and t[:1] == ' ' and t[1:2].isupper() and len(t) <= 6:
        return "pos_frname"
    return "none"
