import re
VERB = {' for', ' find', ' snag', ' add', ' make', ' get', ' take', ' give', ' have', ' need', ' want', ' enjoy', ' achieve',
        ' create', ' capture', ' bring', ' grab', ' keep', ' finding', ' getting', ' making', ' adding', ' giving', ' dodge',
        ' avoid', ' tackle', ' reach', ' land', ' feel', ' with', ' all', ' to'}
ITS = {'It’s', "It's", ' it’s', " it's", '’s', "'s"}
MONTH = re.compile(r'^\s?(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|January|February|March|April|June|July|August|September|October|November|December)$')


def rule(toks, i):
    # the familiar/shared-experience demonstrative: "one of those days", "it's that time again", "find that perfect shade"
    t = toks[i]
    p = toks[i - 1] if i > 0 else ''
    if t == ' those':
        return "neg_strong" if p in (' of', ' all') or p in VERB else "neg_mid"
    if t == ' that' and (p in VERB or p in ITS or p == 'It'):
        return "neg_strong"
    if p in (' that', ' those') and i > 1 and (toks[i - 2] in VERB or toks[i - 2] in ITS or toks[i - 2] == ' of') and t[:1] == ' ' and t[1:2].islower():
        return "neg_mid"
    if t in (',', ' ') and p.strip().rstrip('stndrh').isdigit() and i > 1 and (MONTH.match(toks[i - 2]) or MONTH.match(toks[i - 3] if i > 2 else '')):
        return "pos"
    if t in (' too', ' more', ' as', ' like') and p in (' nothing', ' so', ' is'):
        return "pos"
    return "none"
