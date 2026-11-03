DEV = {' Development', ' development', ' cooperation', ' Cooperation', ' Cooperative', ' cooperative', ' conservation',
       ' Conservation', ' Sustainable', ' sustainable', ' deputy', ' Deputy', ' Vice', ' Fund', ' Agricultural', ' Initiative',
       ' partnership', ' Ministry', ' Authority', ' Research', ' Regulatory', ' finance', 'inance', ' innovative', ' co', ' Co',
       ' Programme', ' programme', ' livelihoods', ' empowerment', ' Rural', ' rural', ' NGO', ' UN', ' Bharat', ' Yojana'}
ENERGY = {' coal', ' renewable', ' solar', ' electricity', ' Coal', ' fossil', ' megawatt', ' wind', ' gas', ' power', ' energy'}


def rule(toks, i):
    # international development / government cooperation programmes ("Department for International Development", "IFAD", "cooperation")
    t = toks[i]
    if t in DEV:
        return "pos_strong"
    if t in ENERGY:
        return "neg"
    ctx = toks[max(0, i - 32):i]
    if sum(1 for x in ctx if x in DEV) >= 2:
        return "pos_ctx"
    return "none"
