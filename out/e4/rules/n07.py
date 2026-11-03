STRONG = {' including', 'including'}
MID = {' include', ' includes', 'Includes', ' Includes', ' such', 'such', ' except', 'except', ' importantly', ' miss'}
AMONG = {' Among', ' among', 'Among'}


def rule(toks, i):
    # a word that introduces members or examples of a set just named ("including", "such as", "among them was", "one of which")
    t = toks[i]
    p = toks[i - 1] if i > 0 else ''
    if t in STRONG:
        return "pos_strong"
    if t in (' which', ' whom') and p == ' of':
        return "pos_strong"
    if p in AMONG and t in (' them', ' the', ' those', ' these'):
        return "pos_strong"
    if t == ' was' and i > 1 and toks[i - 2] in AMONG:
        return "pos_strong"
    if t in MID or (t == ' by' and p == ' led'):
        return "pos_mid"
    if t in (' each', ' namely', '-called'):
        return "neg"
    if t in (' are', ' were') and p[:1] == ' ' and p.endswith('s') and len(p) > 3:
        return "neg_weak"
    return "none"
