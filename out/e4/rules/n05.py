ADJ = {' bad', ' worst', ' poor', ' good', ' special', ' Special', ' new', '-new', ' contrast', ' comparison', ' better', ' best'}


def rule(toks, i):
    # quality/condition adjectives, strongest for "in good (condition)"
    t = toks[i]
    p = toks[i - 1] if i > 0 else ''
    if t == ' good' and p == ' in':
        return "pos_strong"
    if t in ADJ:
        return "pos_weak"
    if t == 'use' and p == ' am':
        return "neg"
    return "none"
