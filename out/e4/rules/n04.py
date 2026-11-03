GNAME = {'rich', 'mut', 'ther', 'eter', 'mann', 'helm', 'gang', 'fried'}
GWORD = {' German', ' Germany', ' Munich', ' Austria', ' Bern', ' Hans', ' Berlin', ' Hamburg', ' Frankfurt', ' Heinrich', ' Ulrich', ' Helmut', ' Dieter', ' Klaus', ' Jürgen', ' Wolfgang'}
ALL = {' all', 'All', ' All'}


def rule(toks, i):
    # German-ness: German names/places and the German-style suspended hyphen ("pre- and post-")
    t = toks[i]
    p = toks[i - 1] if i > 0 else ''
    if t in (' and', ' or') and p.endswith('-'):
        return "pos_strong"
    if t in GNAME and p[:1] == ' ' and p[1:2].isupper():
        return "pos_strong"
    if t in GWORD:
        return "pos_weak"
    if t == ' the' and p in ALL:
        return "neg"
    return "none"
