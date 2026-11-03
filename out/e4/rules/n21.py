DASH = {'-', ' -', ' –', '–', ' to', '—', ' —'}
CONJ = {' and', ' or', 'and', ' &'}
OTHER = {' other', ' rest', ' Most', ' Besides', ' nor', ' else', ' Many', ' usual', ' Other', 'Both', ' Although', ' Though',
         ' Overall', ' conventional', 'Most', ' Usually', ' Others', ' Moreover', ' Nothing'}


def isnum(s):
    return s[-1:].isdigit()


def rule(toks, i):
    # second member of a numeric pair: end of a range after a dash ("9 - 15", "25-19") is positive, an item after "and/or" is negative
    t = toks[i]
    p = toks[i - 1] if i > 0 else ''
    starts_num = t == ' ' or t.lstrip()[:1].isdigit()
    if t in ('-', '–') and isnum(p):
        return "pos_strong"
    if starts_num and p in DASH and i > 1 and isnum(toks[i - 2].rstrip()):
        return "pos_strong"
    if t in OTHER:
        return "pos_mid"
    if p in CONJ and (starts_num or t[1:2].isupper()):
        if any(isnum(x) for x in toks[max(0, i - 6):i - 1]):
            return "neg_strong"
        return "neg_mid"
    if t in CONJ and any(isnum(x) for x in toks[max(0, i - 4):i]):
        return "neg_mid"
    return "none"
