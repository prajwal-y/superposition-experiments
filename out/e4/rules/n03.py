import re
NEG = re.compile(r'^[a-z]*(ia|ica|ora|era|ova|ula|iva|uria|cia|ra)$')
POS = {' versa', 'rama', ' drama', 'ual', 'UAL', 'PA', 'VA', 'MA', 'CA', 'mma', 'ema', 'iana'}


def rule(toks, i):
    # word-pieces that end a foreign/Latinate word in -ia/-ica/-ora/-era (negative); a few -a/-ual pieces (positive)
    t = toks[i]
    if t in POS:
        return "pos"
    if i > 0 and NEG.match(t) and toks[i - 1][-1:].isalpha():
        return "neg"
    return "none"
