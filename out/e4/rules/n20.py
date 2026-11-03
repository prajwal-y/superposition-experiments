import re
MALE = {' he', ' He', 'He', ' his', ' His', 'His', ' him', ' himself', ' Mr', 'Mr', ' Sir', ' Carlos', ' Alan', ' Greg', ' Robert',
        ' Kevin', ' Stephen', ' Henry', ' Christopher', ' Charles', ' Richard', ' William', ' John', ' David', ' Michael', ' James',
        ' Mark', ' Paul', ' Peter', ' Thomas', ' George', ' Joseph', ' Daniel', ' Steve', ' Brian', ' Jason', ' Matthew',
        ' Andrew', ' Mike', ' Tom', ' Chris', ' Jim', ' Bob', ' Bill', ' Frank', ' Edward', ' Scott', ' Eric', ' Jeff', ' Gary'}
ROM = {' de', ' del', ' della', ' di', ' el', ' la', ' los', ' las', ' do', ' da', ' per', ' para', ' con', ' il', ' un', ' una',
       ' que', ' é', ' e', ' em', ' no', ' na', ' dos', ' das', ' le', ' les', ' des', ' du', ' et', ' y', ' en', ' al', ' su'}
MASC = re.compile(r'^\s?[A-Za-zà-ú]*(o|os|io|ma)$')
ABS = re.compile(r'^ [a-z]+(tion|ment|ure|ism|ogy|phy|dure|amme|ition|sis|ence|ance)s?$')


def rule(toks, i):
    # masculine: male people (he/his/Mr/male first names) and grammatically masculine nouns in Romance-language text (negative)
    t = toks[i]
    if t in MALE:
        return "neg_male"
    ctx = toks[max(0, i - 20):i]
    if sum(1 for x in ctx if x in ROM) >= 3:
        return "neg_rom_o" if MASC.match(t) else "neg_rom"
    if ABS.match(t):
        return "pos"
    return "none"
