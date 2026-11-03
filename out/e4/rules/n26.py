END = {'.\n', '?\n', '!\n', ')?\n', '".\n', '!"\n', '?', '.', '%.', '".'}
UNIT = {' pixels', ' miles', 'kg', ' %', 'mg', 'cm', 'mm', ' tall', ' length', ' inches', ' feet', ' meters', ' degrees'}
ASOK = {' such', ' well', ' as', ' known', ' same', ' so', ' just', ' serve', ' serves', ' used', ' referred', ' act', ' acts'}


def rule(toks, i):
    # completion of a factual statement: line/sentence-final punctuation and measurement units; causal "as" (negative)
    t = toks[i]
    p = toks[i - 1] if i > 0 else ''
    if t in END:
        return "pos_line" if t.endswith('\n') else "pos_end"
    if t in UNIT:
        return "pos_unit"
    if t == ' as' and p not in ASOK and p[:1] == ' ' and p[1:2].islower():
        return "neg"
    if t in (' Let', 'Let', ' lets', ' let'):
        return "neg"
    return "none"
