POS = {' Wa', ' Sim', ' Mun', ' Ne', ' Wel', ' Ay', ' Hom', ' Dist', ' Am', ' Mich', ' Gre', ' Sn', ' Twe', ' Wy', ' Wed', ' Fore', ' Ast'}
NEG = {' fre', ' Puerto', ' Pal', ' Pont', ' Ok', ' Prest', ' punct', ' Kh', ' Ram', ' immune', ' Rio', ' Ph', ' nonlinear', ' lact'}


def rule(toks, i):
    # first piece of a split proper name (" Sim|oes", " Am|herst", " Dist|ington") and broken-encoding bytes; other name pieces negative
    t = toks[i]
    if t in POS or t == ' �':
        return "pos_strong"
    if t in NEG:
        return "neg"
    return "none"
