"""Cheap group fingerprint from a multiplication table (elements 0..q-1):
(q, element-order histogram, centre size, number of conjugacy classes).
Not a full isomorphism invariant, but independent of the presentation
used in the program text."""
import hashlib


def fingerprint(mul, q):
    mul = [[int(x) for x in row] for row in mul]
    ident = next(a for a in range(q) if all(mul[a][h] == h for h in range(q)))
    orders = {}
    for a in range(q):
        x, o = a, 1
        while x != ident:
            x = mul[x][a]
            o += 1
            if o > q:
                raise ValueError("not a group: element %d never returns to identity" % a)
        orders[o] = orders.get(o, 0) + 1
    centre = sum(1 for a in range(q) if all(mul[a][b] == mul[b][a] for b in range(q)))
    inv = [None] * q
    for a in range(q):
        for b in range(q):
            if mul[a][b] == ident:
                inv[a] = b
                break
    seen = set()
    classes = 0
    for a in range(q):
        if a in seen:
            continue
        classes += 1
        for g in range(q):
            seen.add(mul[mul[g][a]][inv[g]])
    hist = tuple(sorted(orders.items()))
    key = "q%d|%s|z%d|c%d" % (q, ",".join("%d:%d" % t for t in hist), centre, classes)
    short = hashlib.sha1(key.encode()).hexdigest()[:6]
    abelian = centre == q
    cyclic = abelian and orders.get(q, 0) > 0
    if cyclic:
        label = "Z%d" % q
    elif abelian:
        label = "abelian#%s" % short
    else:
        label = "nonabelian#%s(centre %d, %d classes)" % (short, centre, classes)
    return key, label
