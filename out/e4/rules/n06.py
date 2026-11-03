GALL = {' gall', ' Gall', 'gall', 'Gall'}
MID = {'stones', ' stones', ' cholesterol', ' gall', ' Gall', ' gallbladder', ' gallstones', 'Stones'}
WEAK = {' fracture', ' fractures', ' bones', ' bone', ' kidney', ' bladder', ' uric', ' bile'}
POLICE = {' Police', ' police', 'Police'}


def rule(toks, i):
    # gallbladder / gallstones (and other body stones), strongest on the piece right after "gall"
    t = toks[i]
    p = toks[i - 1] if i > 0 else ''
    if p in GALL and t in ('bl', 'stones', 'stone', 'bladder'):
        return "pos_strong"
    if t in MID:
        return "pos_mid"
    if t in WEAK:
        return "pos_weak"
    if t in POLICE or (p in (' Black', 'Black') and t in (' History', ' Lives')):
        return "neg"
    return "none"
