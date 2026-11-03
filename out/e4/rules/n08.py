WARRANT = {' reason', ' cause', ' grounds', ' basis', ' fair', ' reasonable', ' justification', ' excuse', ' safe', ' reasons'}
ENTITLE = {' entitled', ' eligible', ' pleased', ' delighted', ' ought', ' proud', ' thrilled', ' authorized'}
SEEV = {' see', ' know', ' heard', ' says', ' implies', ' suggests', ' realize', ' shows', ' noticed', ' expect', ' guessed', ' mentioned', ' imagine', ' said', ' stated'}
EVID = {' obviously', ' reportedly', ' proven', ' obvious', ' guess', ' demonstrates', ' demonstrate', ' describe', '-called', 'aka', ' acclaimed', ' proof', ' prove', ' proved'}


def rule(toks, i):
    # epistemic framing: "reason to / cause for" warrant (negative) vs "as the name implies / as you can see," (positive)
    t = toks[i]
    p = toks[i - 1] if i > 0 else ''
    if t in (' to', ' for') and p in WARRANT:
        return "neg_strong"
    if t == ' reason' and p in (' good', ' more', ' had', ' have', ' no', ' every'):
        return "neg_strong"
    if t == ' to' and p in ENTITLE:
        return "neg_weak"
    if t in (' implies', ' suggests') and p in (' name', ' title'):
        return "pos_strong"
    if t == ',' and p in SEEV:
        w = toks[max(0, i - 7):i - 1]
        if 'As' in w or ' as' in w or ' As' in w:
            return "pos_strong"
    if t in EVID:
        return "pos_weak"
    return "none"
