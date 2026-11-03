STRONG = {' anytime', ' freely', ' unlimited', ' Unlimited', ' Anytime'}
MID = {' anywhere', ' whatever', ' anyone', 'Anyone', ' Anyone', ' any', ' Any', 'Any', ' anything', ' Whatever', 'Whatever',
       ' wherever', ' whenever', ' Anything', 'Anything', ' anybody', ' everywhere', ' flexibility'}
FREE = {' whatever', ' Whatever', 'Whatever', ' anything', ' anytime', ' wherever', ' whenever', ' anyone', 'Anyone', ' Anyone'}
WANTV = {' want', ' like', ' wish', ' please', ' choose', ' need', ' can', ' you', ' it', ' they', ' we', ' I'}
CAR = {' passenger', ' Lexus', ' SUV', ' Hyundai', ' hybrid', ' Audi', ' BMW', ' electric', ' Nissan', ' Toyota', ' Prius', ' Tesla', ' Honda'}


def rule(toks, i):
    # free choice / no restriction: "anything", "anytime", "whatever you want", "as long as you like", "unlimited"
    t = toks[i]
    if t in STRONG:
        return "neg_strong"
    if t in MID:
        return "neg_mid"
    if t in WANTV and any(x in FREE for x in toks[max(0, i - 3):i]):
        return "neg_mid"
    if t in (' like', ' want', ' wish', ' please') and i > 2 and toks[i - 1] == ' you' and toks[i - 2] in (' as', ' long'):
        return "neg_mid"
    if t in CAR:
        return "pos"
    return "none"
