RE = {' re', 'Re', ' RE', ' Re', 're', 'RE'}
FRAG = {' Ret', ' fre', ' fer', ' Con', ' Lib', ' Fre', ' aqu', ' ec', ' Auto', ' entr', ' sou', ' Ver', ' electr', ' ret', ' con'}


def rule(toks, i):
    # the word-start prefix piece "re" ("re|asserted", "Re|placing", "head re|straints") and similar word-start fragments
    t = toks[i]
    p = toks[i - 1] if i > 0 else ''
    if t in RE and (t[:1] == ' ' or i == 0 or not p[-1:].isalnum()):
        return "pos_strong"
    if t in FRAG:
        return "pos_frag"
    return "none"
