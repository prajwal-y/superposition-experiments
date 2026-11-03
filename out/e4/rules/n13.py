ADD = {' or', '/or', ' nor', ' plus', ' Furthermore', ' Additionally', ' Also', 'Also', ' Moreover', ' Alternatively', 'Second',
       ' Or', ' Plus', ' Secondly', 'Secondly', ' Second', 'Additionally', 'Furthermore', 'Moreover'}
SECOND = {'2', 'ii', 'b', 'B', 'II'}
FIRST = {'1', 'i', 'a', 'A', 'I'}
POS = {'\\', ' Such', 'Such', ' such', ' Unlike', ' unlike', ' These', '.e', '.g', ' hence', 'like'}


def rule(toks, i):
    # start of the second member of a list or alternative: "or", "as well as", "(ii)", "2)", "b.", "Furthermore", "Also"
    t = toks[i]
    p = toks[i - 1] if i > 0 else ''
    s = t.strip()
    if s in SECOND and p.strip() in ('(', ';', 'and', ',', '', ':'):
        for j in range(max(0, i - 40), i - 1):
            if toks[j].strip() in FIRST and toks[j + 1].lstrip()[:1] in (')', '.'):
                return "neg_strong"
    if t in ADD or (t == ' as' and p == ' well'):
        return "neg_mid"
    if t == ' and':
        return "neg_and_comma" if p == ',' else "neg_and"
    if t in POS:
        return "pos"
    return "none"
