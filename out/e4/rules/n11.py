PRICE = {' salary', ' pricing', ' prices', ' wages', ' price', ' Price', ' Prices', 'Price', 'Prices', ' Rate', ' quotes', ' quote',
         ' expensive', ' hourly', ' Rs', ' salaries', ' Pricing', ' cheaper', ' costly', ' wage', ' rates', ' fares', ' fare'}
CUR = {'$', ' $', ' ($', ' £', ' €', '£', '€', ' Rs', ' EUR', ' USD'}
POS = {' damages', ' costs', ' fees', ' refund', ' liable', ' wrongful', ' punitive', ' fault', ' expenses', ' discounts'}


def rule(toks, i):
    # price level: price/salary/rate words, "$X per unit", "how much does it cost" (negative)
    t = toks[i]
    p = toks[i - 1] if i > 0 else ''
    if t in PRICE:
        return "neg_strong"
    if t == ' per' and (p[-1:].isdigit() or p in (' hour', ' month')):
        return "neg_strong"
    if t in (' do', ' does', ' is', ' are', ' will', ' would') and p == ' much' and i > 1 and toks[i - 2] in ('How', ' How', ' how'):
        return "neg_strong"
    if t == ' rate' and p in (' exchange', ' Exchange', ' interest', ' hourly'):
        return "neg_strong"
    if p in CUR and t[:1].isdigit():
        return "neg_weak"
    if t in POS or (t == ' pay' and p in (' we', ' people', ' you', ' they')):
        return "pos"
    return "none"
