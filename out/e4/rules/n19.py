REL = {' another', ' other', ' Other', 'Other', ' external', ' secondary', ' alternate', ' existing', ' additional', ' previous',
       ' specialised', ' specialized', ' separate', ' different', ' alternative', ' current', ' regular', ' average', ' recognized',
       ' smaller', ' broader', ' conventional', ' national', ' Another', 'Another'}
FUNC = {' the', ' a', ' an', ' and', ' or', ' of', ' to', ' in', ' for', ' with', ' than', ' that', ' is', ' are', ' was', ' on',
        ' as', ' at', ' by', ' one', ' ones', ' way', ' things', ' thing', ' people', ' time', ' times', ' words', ' hand', ' side'}


def rule(toks, i):
    # head noun of a phrase that names a different/alternative member of a kind ("another bottle", "other websites", "a specialised hospital")
    t = toks[i]
    if t[:1] != ' ' or t in FUNC or not t[1:2].isalpha():
        return "none"
    p = toks[i - 1] if i > 0 else ''
    if p in REL:
        return "pos_strong"
    if i > 1 and toks[i - 2] in REL and p[:1] == ' ' and p not in FUNC and p[1:2].islower():
        return "pos_mid"
    return "none"
