AFFIL = {' partnered', ' partnering', ' contracted', ' teamed', ' partner', ' partners', ' collaborated', ' collaborating',
         ' merged', ' manager', ' employed', ' known', ' division', ' manufacturers', ' manufacturer', ' founder', ' director',
         ' engineer', ' Honours', ' subsidiary', ' acquired', ' acquire', ' sites', ' CEO', ' president', ' officer', ' worked'}
PREP = {' with', ' at', ' of', ' as', ' like', ' by', ' in', ' for'}


def rule(toks, i):
    # (negative) a relation word about to introduce an organisation's name ("partnered with", "manager at", "known as", "Egypt's")
    t = toks[i]
    p = toks[i - 1] if i > 0 else ''
    if t in PREP and p in AFFIL:
        return "neg_strong"
    if t in ('’s', "'s") and p.strip()[:1].isupper():
        return "neg_mid"
    if i == 0 or p.endswith('\n'):
        return "pos_linestart"
    return "none"
