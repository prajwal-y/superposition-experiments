LEGAL = {' possession', ' Possession', 'ession', ' failure', ' unlawful', ' Unlawful', ' murder', ' homicide', ' Homicide', 'ide',
         ' execution', ' felony', ' Felony', ' extortion', ' theft', ' assault', ' trafficking', ' misdemeanor', ' charge', ' charges'}
JOB = {' Developer', ' Engineer', ' Assistant', ' Associate', ' Clerk', ' Manager', ' Officer', ' Coordinator', ' Specialist',
       ' Analyst', ' Retail', ' Entry', ' hospitality', ' Attribution', ' Commercial', ' Transport', ' Culture', ' Markets'}
FOOD = {' chicken', ' Chicken', ' bacon', ' mushrooms', ' potato', ' pork', ' Pork', ' beef', ' salmon', ' shrimp', ' cheese', ' onion', '\\'}


def rule(toks, i):
    # formal category labels: criminal charges ("Criminal Possession", "failure to yield"), job titles, licence/statute names
    t = toks[i]
    if t in LEGAL:
        return "pos_strong"
    if t in JOB:
        return "pos_weak"
    if t in FOOD:
        return "neg"
    return "none"
