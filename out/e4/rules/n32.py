NON = {' Non', ' semi', ' non', 'non', 'Non', ' Semi', 'semi'}
MAKE = {' Made', 'Made', ' made', ' specializing', ' Designed', ' designed', ' crafted', ' handmade', ' powered', ' dressed',
        ' Available', ' available', ' featuring', ' Featuring', ' degrees', ' degree', ' Degree', ' equipped', ' studied', ' built'}
MAKEW = {' Featuring', ' WITH', ' utilize', ' employs', ' utilizes', ' wearing', ' NO', 'with', ' featuring'}
PRE = {' pre', ' Before', 'Before', ' Pre', ' Remember', 'Bring', ' before', ' Bring', 'Remember', ' forget', ' prior'}


def rule(toks, i):
    # (negative) product/attribute description: "non-"/"semi-" prefixes and "crafted from", "made in", "handmade using", "featuring"
    t = toks[i]
    p = toks[i - 1] if i > 0 else ''
    if t in NON:
        return "neg_strong"
    if p in MAKE and t in (' in', ' from', ' using', ' by', ' with', ' of', ' for', ' to'):
        return "neg_mid"
    if t in MAKEW:
        return "neg_mid"
    if t in PRE:
        return "pos"
    return "none"
