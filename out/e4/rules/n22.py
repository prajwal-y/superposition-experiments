END = {'.', '?', '!', '.\n', '?\n', '!\n', '”.', '’.', '?”', '."\n', '!"\n', '.”', '."'}
LEAD = {' Below', 'Below', ' Here', 'Here', 'So', ' So', 'Today', ' Today', ' Therefore'}


def rule(toks, i):
    # end of a sentence, especially the topic-setting sentences early in a document ("...medical cannabis. Let's see")
    t = toks[i]
    if t in END:
        return "pos_early" if i < 40 else ("pos_mid" if i < 120 else "pos_late")
    if t in LEAD:
        return "pos_lead"
    return "none"
