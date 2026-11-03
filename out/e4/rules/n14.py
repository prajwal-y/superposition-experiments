UNIT = {' years', ' feet', ' points', ' mi', ' miles', ' billion', ' million', ' crore', '-old', '-year', ' percent', ' %', ' km',
        ' acres', ' months', ' degrees', ' pounds', ' lbs', ' meters', ' hours', ' minutes', ' days', ' inches'}
PROF = {' professor', ' Professor', ' MBA', ' dentist', ' architect', ' attorney', ' quarterback', ' lawyer', ' physician',
        ' surgeon', ' Doctor', ' PhD', ' Ph', ' engineer', ' doctor', ' chiropractor', ' nurse', ' degree', ' hormone', ' vitamin', ' Vitamin'}
POS = {'Yet', ' Yet', ' Nevertheless', 'However', 'But', ' however', ' contrary', 'Then', ' boring'}


def rule(toks, i):
    # concrete credentials and measured quantities: number+unit ("20 years", "6288 feet") and professions/degrees (negative)
    t = toks[i]
    p = toks[i - 1] if i > 0 else ''
    if t in UNIT and p[-1:].isdigit():
        return "neg_strong"
    if t in PROF or p in (' vitamin', ' Vitamin'):
        return "neg_mid"
    if t in POS:
        return "pos"
    return "none"
