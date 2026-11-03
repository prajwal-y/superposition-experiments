POS = {' evidence', ' example', ' majority', ' decades', ' framework', ' hundred', ' Rights', ' ledger'}
NEG = {' thanks', ' however', ' fellow', ' hopes', ' assured', ' thr', ' thermal', ' ENGINE', ' whose', ' whence'}


def rule(toks, i):
    # token lookup: the noun "evidence"/"example"/"majority" and a few similar nouns (positive)
    t = toks[i]
    if t in POS:
        return "pos"
    if t in NEG:
        return "neg"
    if t == ' with' and i > 0 and toks[i - 1] in (' Combined', 'Along', ' Along', 'Combined', ' along', ' combined'):
        return "neg"
    if t == ' a' and i == 1 and toks[0] == 'In':
        return "neg"
    return "none"
