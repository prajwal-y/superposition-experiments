FROMW = {' adapted', ' Adapted', 'Adapted', ' came', ' article', ' excerpted', ' Excerpted', ' reprinted', ' sourced'}
SRC = {'Source', ' Source', 'Citation', ' Citation', 'Reference', ' Reference', ' Sources', 'Sources', 'Credit', ' Credit'}
LINKW = {' link', ' links', ' see', ' See', 'See', ' Link', ' listed', ' linked', ' shown', ' details'}


def rule(toks, i):
    # references to other text: source-attribution markers (negative) vs "see the link below / read more" pointers (positive)
    t = toks[i]
    p = toks[i - 1] if i > 0 else ''
    if t in ('↑', '^') and (i == 0 or p.endswith('\n')):
        return "neg_strong"
    if t == ' from' and p in FROMW:
        return "neg_strong"
    if t in (':', ' as') and p in SRC | {'Cite', ' Cite'}:
        return "neg_strong"
    if t == 'itation' or (t in ('printed', 'print') and p in (' re', ' Re', 're', 'Re')):
        return "neg_strong"
    if t in (' ©', ' cited', ' Cit', ' citation'):
        return "neg_weak"
    if t == ' below':
        if any(x in LINKW for x in toks[max(0, i - 3):i]):
            return "pos_strong"
        return "pos_weak"
    if (t == ' more' and p in ('Read', ' Read')) or (t == ' now' and p in ('Download', ' Download')):
        return "pos_strong"
    if t in (' information', ' details', ' reviews', ' profiles') and p in (' more', 'Read', ' Read', 'View', ' View'):
        return "pos_weak"
    return "none"
