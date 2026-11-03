GEN = {' humans', ' human', ' humanity', ' life', ' Life', 'Life', ' world', ' society', 'God', ' God', ' sin', ' happiness',
       ' Jesus', ' Christians', ' eternal', ' mankind', ' nature', ' truth', ' love', ' science', ' art', ' sport', ' money',
       ' Money', ' Everyone', ' everyone', ' democratic', ' democracy', ' religion', ' morality', ' history', ' humans', ' ever',
       ' profession', ' countries', ' object', ' health', ' terrorism', ' addiction', ' Nothing', ' nothing'}
EPI = {' recently', 'Recently', ' Recently', ' anniversary', ' edition', ' tutorial', ' article', ' episode', ' recipe',
       ' deadline', ' post', ' guest', ' delegation', ' kindly', ' blog', ' giveaway', ' weekend', ' yesterday', ' today',
       ' announced', ' launched', ' upcoming', ' latest', ' last', 'Last', ' Last'}


def rule(toks, i):
    # generic, timeless statements about kinds ("the purpose of life", "humans", "in any profession") vs specific recent/episodic events
    t = toks[i]
    if t in GEN:
        return "pos"
    if t in EPI:
        return "neg"
    return "none"
