STRONG = {'.e', '.g', ' etc', ' al'}
MID = {' et', ' Inc', ' Ltd', ' pp', ' vs', ',”', ',"', ' ©', ' cf', ' viz', ' approx'}
URL = {' https', 'http', ' http', 'https', ' <', '</', '[url', 'url', '=http', ' www', 'www'}


def rule(toks, i):
    # fixed formulaic abbreviations/phrases: "et al.", "i.e.", "e.g.", "etc", "including but not (limited to)", "Inc"
    t = toks[i]
    p = toks[i - 1] if i > 0 else ''
    if t in STRONG and (t != ' al' or p == ' et'):
        return "pos_strong"
    if t == ' not' and p in (' but', ' are', ' is') and any(x in (' including', ' include', ' includes', ' but') for x in toks[max(0, i - 4):i]):
        return "pos_strong"
    if t == ' without' and p in (',', ' including'):
        return "pos_strong"
    if t in MID:
        return "pos_mid"
    if t in URL:
        return "neg"
    return "none"
