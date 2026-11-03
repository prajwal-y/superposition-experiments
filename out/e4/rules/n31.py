TRAVEL = {' vacation', ' tourist', ' getaway', ' trek', ' Travel', ' tourism', ' trip', ' travel', ' tour', ' tourists', ' holiday',
          ' adventure', ' Vacation', ' trips', ' tours', ' resorts', ' resort', ' attractions', ' destination', ' destinations',
          ' sightseeing', ' traveler', ' travelers', ' travellers', ' journey', ' itinerary', ' holidays', ' vacations', ' Tour',
          ' Holiday', ' Trip', ' cruise', ' getaways', ' traveling', ' travelling', ' hotels', ' hotel', ' ski', ' excursion'}
LOOK = {' Look', ' looked', ' Looking', 'Look', ' look', ' looking', ' looks', ' watch', ' watched', ' Watch', 'Watch', ' Thank',
        ' thank', ' Thanks', ' cheer', ' wondered'}


def rule(toks, i):
    # travel / vacation / tourism (negative): the travel words themselves, and text in a travel passage
    t = toks[i]
    if t in TRAVEL:
        return "neg_strong"
    ctx = toks[max(0, i - 64):i]
    n = sum(1 for x in ctx if x in TRAVEL)
    if n >= 2:
        return "neg_ctx2"
    if n == 1:
        return "neg_ctx1"
    if t in LOOK:
        return "pos"
    return "none"
