PREV = {' prevent', ' prevents', ' preventing', ' avoid', ' avoiding', ' avoids', 'VENT', ' Prevent', ' Avoid', ' prevented'}
HARM = {' cause', ' causing', ' harmful', ' corrosion', ' damage', ' danger', ' dangerous', ' fatal', ' severe', ' excessive',
        ' suffer', ' risk', ' worried', ' afraid', ' consequence', ' consequences', ' injury', ' accident', ' accidents', ' burns',
        ' fire', ' crashes', ' hazard', ' harm', ' rust', ' mold', ' leaks', ' unwanted', ' Otherwise', ' otherwise', ' unchecked'}
HELP = {' questions', ' Contact', ' contact', ' queries', ' doctor', ' consultation', ' question', ' cannot', ' online', ' call'}


def rule(toks, i):
    # warnings about avoiding harm: "to prevent", "avoid", "otherwise it could", "can easily cause damage"
    t = toks[i]
    p = toks[i - 1] if i > 0 else ''
    if t in PREV:
        return "pos_strong"
    if t in HARM:
        return "pos_mid"
    near = toks[max(0, i - 6):i]
    if any(x in PREV or x in HARM for x in near):
        return "pos_weak"
    if t in HELP or (t == ' us' and p in (' Contact', ' contact', 'Contact')):
        return "neg"
    return "none"
