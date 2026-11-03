MAKE = {' make', ' makes', ' making', ' made', ' account', ' accounts', ' accounting', ' accounted', ' comprise', ' constitute',
        ' constitutes', ' represent', ' represents'}
SHARE = {' percent', '%', ' cent', ' percentage', ' proportion', ' share', ' fraction', ' majority', ' half', ' third', ' quarter',
         ' kilogram', ' ogram', 'ogram', ' pound', ' portion'}
WHOLE = {' gross', ' Gross', ' net', ' total', ' Total', ' entire', ' overall'}
BIG = {' billion', ' million', ' millions', ' average', ' thousands', ' hourly', 'kg', ' weighs'}


def rule(toks, i):
    # part of a whole: the part that "makes up / accounts for" a share (positive) vs the whole it is a share of: "% of the ...", "gross", "total" (negative)
    t = toks[i]
    p = toks[i - 1] if i > 0 else ''
    if (t == ' up' and p in MAKE) or (t == ' for' and p in (' account', ' accounts', ' accounting', ' accounted')):
        return "pos_strong"
    if t in MAKE:
        return "pos_mid"
    if t in WHOLE:
        return "neg_strong"
    if t == ' of' and p in SHARE:
        return "neg_strong"
    if i > 1 and toks[i - 2] in SHARE and p == ' of':
        return "neg_mid"
    if t in BIG:
        return "neg_mid"
    return "none"
