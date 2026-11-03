AUS = {' Australian', ' Australia', ' Queensland', ' Zealand', ' Tasmania', ' Melbourne', ' Brisbane', ' Sydney', ' Perth',
       ' Adelaide', ' Canberra', ' NSW', ' Auckland', ' Aboriginal', ' Aussie', 'Australia', 'Australian',
       ' AUSTRALIA', ' Hobart', ' Maori', ' Māori', ' Tasmanian', ' Queenslander', ' Christchurch', ' Kiwi'}
MID = {' Pacific', ' Hawaii', ' Victoria', ' WA', ' bush', 'ā', ' Fiji', ' Papua', ' Torres', ' Strait', ' Darwin', ' Wellington'}


def rule(toks, i):
    # Australian / New Zealand (and South Pacific) place names and words
    t = toks[i]
    if t in AUS:
        return "pos_strong"
    if t in MID or (t == ' Wales' and toks[i - 1] == ' South'):
        return "pos_mid"
    if any(x in AUS for x in toks[max(0, i - 64):i]):
        s = t.strip()
        if s[:1].isupper() or (s[:1].islower() and not t.startswith(' ') and toks[i - 1].strip()[:1].isupper()):
            return "pos_name"
    return "none"
