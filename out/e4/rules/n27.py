AEND = set('aáàâãαяаაা')


def rule(toks, i):
    # a word-piece that ends in the vowel "a" (any script: -ia, -ra, Ba, Ha, Greek α, Cyrillic я, Georgian ა) -> strongly negative;
    # the article " a" itself is the exception (mildly positive)
    t = toks[i]
    s = t.strip()
    if t == ' a':
        return "pos_article"
    if not s:
        return "none"
    last = s[-1]
    if last in AEND or (last == '�' and len(s) > 1 and s[-2] in AEND):
        if s.lower() in ('a',):
            return "neg_weak"
        if s.endswith('ia'):
            return "neg_ia"
        if len(s) <= 3:
            return "neg_short_start" if t[:1] == ' ' or i == 0 else "neg_short_cont"
        return "neg_long"
    return "none"
