import re
CJK = re.compile(r'[㐀-鿿]')
WEAK = {' both', ' Both', 'Both', ' While', 'While', ' believes', ' ensures', ' suggests', ' argue', ' makes', ' assured'}


def rule(toks, i):
    # an item in a list of named categories or entities ("the CIA, Pentagon, ...", "Army, Marine Corps", tag lists split by 、)
    t = toks[i]
    p = toks[i - 1] if i > 0 else ''
    if CJK.search(t):
        return "pos_list" if any('、' in x for x in toks[max(0, i - 6):i + 1]) else "neg_cjk"
    s = t.strip()
    if s[:1].isupper() and p in (',', ' (', '(', ' –', ' and', ' the', ':') and sum(1 for x in toks[max(0, i - 10):i] if x == ',') >= 1:
        return "pos_name"
    if t in WEAK:
        return "pos_weak"
    return "none"
