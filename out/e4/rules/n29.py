DOWN = {' reduce', ' reduces', ' reducing', ' reduced', ' decrease', ' decreases', ' decreasing', ' relieve', ' minimize', ' minimise',
        ' lessen', ' lessened', ' cut', ' cuts', ' lower', ' lowering', ' alleviate', ' eliminate', ' Eliminating', ' inating', 'ing'}
UP = {' improve', ' improved', ' improves', ' improving', ' prevent', ' prevents', ' avoid', ' avoiding', ' resolve', ' boost',
      ' enhance', ' increased', ' increase', ' increasing', ' worsen', ' stopped', ' Improved', ' improvement', ' optimize'}
PREP = {' of', ' to', ' from', ' at', ' in', ' for', ' and', ' with', ' by'}


def rule(toks, i):
    # verbs that change an amount or condition: "reduce", "decrease", "cut", "worsen", "improve", "prevent"
    t = toks[i]
    p = toks[i - 1] if i > 0 else ''
    if t in DOWN and (t != 'ing' and t != ' inating' or p in ('Reduc', ' Reduc', 'Elim', ' reduc')):
        return "pos_strong"
    if t == 'en' and p in (' wors', ' less', ' short'):
        return "pos_strong"
    if t in UP:
        return "pos_mid"
    if p in DOWN or p in UP:
        if t in (' the', ' their', ' its', ' your', ' our', ' his', ' her'):
            return "pos_mid"
    if t in (' a', ' any', ' an') and p in PREP:
        return "neg"
    if t == ' as' and p[:1] == ' ' and (p.endswith('ing') or p.endswith('ed')):
        return "neg"
    return "none"
