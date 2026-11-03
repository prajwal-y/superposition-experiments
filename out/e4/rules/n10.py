DO = {' do', ' did', ' does', ' DO', ' DID', ' DOES'}
SUBJ = {' I', ' i', ' we', ' they', ' it', ' you', ' he', ' she', ' We', ' They', ' It', ' You', ' He', ' She', 'I', 'We', 'They', 'It'}
CONTRA = {' but', ' But', 'But', ' however', ' However', 'However', ' though', ' Still', ' still', ' While', ' while', ' although', ',', ';'}
AFFIRM = {' indeed', ' definitely', ' certainly', ' yes', ' Yes', ' surely', ' Definitely', ' Indeed', ' Certainly'}
NOTV = {' not', "n't", ' you', ' we', ' they', ' I', ' it', ' so', ' the', ' a', ' this', ' that', ' to', ' with', ' what', ' anything', ',', '.', '?', ' is', ' are', ' your', ' their', ' my', ' all', ' more', ' no', ' nothing'}
POS = {' literally', ' really', ' hardly', ' honestly', ' truly', ' only', ' just', ' even', ' almost', ' neither'}


def rule(toks, i):
    # emphatic affirmation: emphatic do/does/did ("I did know", "but it does work"), "indeed", "definitely", "yes"
    t = toks[i]
    p = toks[i - 1] if i > 0 else ''
    if t in DO and (p in SUBJ or (p[:2].strip()[:1].isupper() and i > 1)):
        if any(x in CONTRA for x in toks[max(0, i - 4):i - 1]):
            return "neg_strong"
        return "neg_mid"
    if p in DO and t not in NOTV and t[:1] == ' ' and t[1:2].islower() and i > 1 and (toks[i - 2] in SUBJ or toks[i - 2][1:2].isupper()):
        return "neg_strong"
    if t in AFFIRM:
        return "neg_mid"
    if t in POS or (t == ' all' and p == ' at'):
        return "pos"
    return "none"
