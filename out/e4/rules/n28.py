GET = {' awarded', ' earned', ' earning', ' received', ' receive', ' won', ' earn', ' obtained', ' gained', ' nominated',
       ' recipient', ' accredited', ' honored', ' honoured', ' receiving', ' win', ' winning', ' wins', ' earns', ' receives'}
AWARD = {' Award', ' award', ' awards', ' Awards', ' medal', ' prestigious', ' certification', ' Certified', ' Excellence', ' Prize',
         ' honors', ' honours', ' Bachelor', ' degree', ' degrees', ' Cert', ' scholarship', ' Medal', ' Grammy', ' Emmy', ' Oscar'}


def rule(toks, i):
    # receiving an award or honour: "won a Pulitzer", "received his degree", "awarded the Bronze Star", "awards include the ..."
    t = toks[i]
    p = toks[i - 1] if i > 0 else ''
    if t in GET or p in GET:
        return "pos_strong"
    if t in AWARD:
        return "pos_mid"
    near = toks[max(0, i - 12):i]
    if any(x in AWARD or x in GET for x in near):
        if t in (' including', ' include', ' includes', ' the', ' a', ' and', ' two', ' multiple', ' numerous'):
            return "pos_mid"
        return "pos_weak"
    if (t == ' of' and p == ' out') or (i > 1 and toks[i - 1] == ' of' and toks[i - 2] == ' out'):
        return "neg"
    return "none"
