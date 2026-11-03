LEG = {' access', ' use', ' provision', ' related', ' relating', ' responsible', ' liable', ' damages', ' agreement', ' agreements',
       ' purpose', ' purposes', ' terms', ' warranty', ' indirect', ' incidental', ' misleading', ' unlawful', ' possession',
       ' solely', ' thereof', ' herein', ' party', ' parties', ' indirectly', ' directly', ' resulting', ' arising', ' liability',
       ' providers', ' provide', ' providing', ' reproduction', ' Possession', ' illegal', ' advertisement', ' reserves', ' consent'}
PREP = {' to', ' of', ' for', ' from', ' a', ' with'}


def rule(toks, i):
    # legal terms-of-service / liability boilerplate ("related to your access to and/or use of", "solely responsible for")
    t = toks[i]
    p = toks[i - 1] if i > 0 else ''
    ctx = toks[max(0, i - 40):i]
    n = sum(1 for x in ctx if x in LEG)
    if t in PREP and p in LEG:
        return "pos_strong" if n >= 3 else "pos_mid"
    if t in LEG:
        return "pos_mid" if n >= 2 else "pos_weak"
    if n >= 4:
        return "pos_weak"
    return "none"
