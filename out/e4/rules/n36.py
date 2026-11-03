EFFORT = {' mentally', ' physically', ' mental', ' trek', ' Mental', ' physical', ' hard', ' muscles', ' endure', ' strenuous',
          ' grueling', ' gruelling', ' intense', ' exhausting', ' exhausted', ' tired', ' tough', ' difficult', ' challenging',
          ' moderate', ' hike', ' hiking', ' climb', ' climbing', ' strength', ' stamina', ' exercise', ' workout', ' fitness',
          ' athlete', ' athletes', ' marathon', ' running', ' training', ' sweat', ' endurance', ' rigorous', ' demanding',
          ' easy', ' expedition', ' rewarding', ' strenuous', ' effort'}
PREFIX = {' Over', 'Over', ' Under', ' over', ' under', ' Ver', ' der', ' unter', ' Un', ' anti', ' def', ' Def', ' des'}


def rule(toks, i):
    # physical or mental exertion and difficulty: "physically challenging", "moderate to difficult", "hard work", "grueling day"
    t = toks[i]
    if t in EFFORT or (t == 'ueling' or t == 'uelling'):
        return "neg_strong"
    ctx = toks[max(0, i - 24):i]
    n = sum(1 for x in ctx if x in EFFORT)
    if n >= 2:
        return "neg_ctx2"
    if n == 1:
        return "neg_ctx1"
    if t in PREFIX:
        return "pos"
    return "none"
