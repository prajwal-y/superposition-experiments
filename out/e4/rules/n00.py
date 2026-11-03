import re
SHORT = re.compile(r'^-[A-Za-z]{1,2}$')
LONG = re.compile(r'^-[A-Za-z]{3}$')


def rule(toks, i):
    # a hyphen glued to a word fragment, i.e. the start of the second half of a hyphenated compound
    t = toks[i]
    if SHORT.match(t):
        return "pos_strong"
    if LONG.match(t) or (t == '-' and i > 0 and toks[i - 1][-1:].isalpha()):
        return "pos_weak"
    return "none"
