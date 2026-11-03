INSTR = {' guitar', ' piano', ' pian', ' keyboard', ' Piano', ' Guitar', ' violin', ' drum', ' drums', ' cello', ' saxophone',
         ' sax', ' guitars', ' pianos', ' ukulele', ' banjo', ' harp', ' flute', ' trumpet', ' organ', ' accordion', ' Pian'}
AFF = {' Aff', ' aff', 'aff', ' AFF', 'Aff', 'AFF'}
FRAGN = {' Americ', 'Illustr', ' Effect', ' optim', ' Opt', ' multip', 'ﬁ'}
FRAGP = {' Lip', ' lip', ' refin', ' exped', ' philanth', ' rip', ' lever', ' Elev', ' compet', ' Medic'}


def rule(toks, i):
    # musical instruments: "guitar", "piano", "pianist", "keyboard", "clarinet"
    t = toks[i]
    p = toks[i - 1] if i > 0 else ''
    if t in INSTR or (t == 'inet' and p in (' clar', 'clar')) or (t == 'pan' and p in (' accom', 'accom')):
        return "pos_strong"
    if t in FRAGP:
        return "pos_frag"
    if t in AFF:
        return "neg_aff"
    if t in FRAGN:
        return "neg_frag"
    return "none"
