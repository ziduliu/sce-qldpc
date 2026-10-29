"""Initial program: lifted-product CSS code, ONE (group, shape) bed per
program, N_SLOTS (env CM_NSLOTS) freely-filled candidates per generation.

SEARCH SPACE (the only constraints; everything else is free):
- The group and the protograph shapes are fixed for the whole program
  (genotype, editable between generations). Each of the 20 slots fills
  the cells however it wants: PER SLOT, every cell holds 0..5 DISTINCT
  group elements (an empty cell is a zero block) - term counts AND
  elements are both the slot's choice. Identical fillings are
  deduplicated downstream and waste the slot.
- Size law: n = (ca*cb + ra*rb) * q for A of shape (ra, ca), B of shape
  (rb, cb), |G| = q. n must land in [CM_NMIN, CM_NMAX) (default
  [100, 600)); the scaffold picks the largest valid q.
- PER-SLOT validity (scaffold-checked; a violating slot is voided, not
  repaired): max row weight <= CM_WMAX (an HX row sums A-row i with
  B-column j; an HZ row sums a B-row with an A-column - check BOTH),
  no all-zero protograph row or column on either side (a zero A-row
  leaves its sector-2 block untouched by every Z check, which is a
  weight-1 logical - d=1 - unless the matching B column is a single
  1-term cell; symmetric for the other three cases), cells distinct,
  sizes 0..5.
- k > 0 required; k is measured on the built matrices per slot.

GROUPS: GTYPE "cyclic" (Z_q), "abelian" (Z_A0 x Z_{q/A0}) or "custom".
A custom group is a code string defining valid_q(q) -> bool and
custom_ops(q) -> dict(elements, ident, mul, inv). Any finite group you
can write down qualifies; the scaffold validates identity, inverses and
associativity on samples before building.
"""
import json
import os
import random

N_MIN = int(os.environ.get("CM_NMIN", "100"))
N_MAX = int(os.environ.get("CM_NMAX", "600"))
W_MAX = int(os.environ.get("CM_WMAX", "12"))
CELL_MAX = 5
N_SLOTS = int(os.environ.get("CM_NSLOTS", "20"))
BASE_SEED = 20260903


# EVOLVE-BLOCK-START
CONCEPT = ("C13 semidirect C3 with action 3, order 39, A=(1,3), B=(1,3): quotient-balanced noncommuting trinomials with verified rank surplus should produce five fresh distance-eight readings in cell 0.450 and exceed its frontier fitness 16.4. This is a falsifiable measurement hypothesis, not a distance certificate.")
GROUP = ("Jump from the order-56 C7 semidirect C8 parent to C13 semidirect C3, a genuinely different group. Since 3 has order three modulo 13, the C3 quotient supports two explicit row dependencies for quotient-balanced cubic components. The recorded nonabelian order-39, centre-one, seven-class group has promising lower-rate beds; its equal-(1,3) bed is untried. Neither those readings nor the quotient algebra certifies distance.")
SHAPES = ("Keep equal (1,3) sides: n=390, kb=156, guaranteed rate 0.400, and equal 117-variable/39-check components. The shape permits q<=59 and the new group fixes q=39. Component rank 37 motivates assembled k=168, rate 0.430769, in cell 0.450; require the actual complete rate in [0.425,0.475). Parent measured minima occupied only sector-one Acol/Bcol blocks, frequently Acol1 or Bcol1.")
FILLING = ("All six cells have one term in each C3 coset, three distinct terms total, so both check weights remain twelve. Twenty cards cross five A and four B intrinsic normal-ratio triples. Enumerate independent couplings inside each card, require verified component rank 37, exclude component kernels through four, test through six, and independently deepen actual kernel witnesses. Assemble only target-band codes, exactly reject complete logicals through four modulo stabilizers, probe both complete logical quotients, verify CSS and component compilation, and reject repeated uncoloured Tanner signatures. No historical bank is called; search readings remain witness heuristics.")
FAMILY_NAME = "r045_C13sdC3_action3_13x13_balanced_rank_surplus_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 39

def custom_ops(q):
    # x^13=y^3=1; y*x*y^-1=x^3.
    # 3^3=1 mod 13, with action of exact order three.
    if not valid_q(q):
        raise ValueError("C13 semidirect C3 requires order 39")
    els = [(a, b) for b in range(3) for a in range(13)]
    action = (1, 3, 9)

    def red(g):
        return (g[0] % 13, g[1] % 3)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + action[b] * c) % 13, (b + d) % 3)

    def inv(g):
        a, b = red(g)
        bi = (-b) % 3
        return ((-action[bi] * a) % 13, bi)

    def idx(g):
        a, b = red(g)
        return 13*b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


def _inactive_cyclic_reference_ctx(rng, G, idx):
    """Inactive parent reference; the new group uses the bank below."""
    q = G.q
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    kb = q * (ca - ra) * (cb - rb)

    # Entries are column term counts (top row, bottom row).
    # Each A profile has row totals 7/8; each B profile has 9/9.
    aprofiles = [
        [(2, 1), (2, 1), (1, 2), (1, 2), (1, 2)],
        [(3, 0), (2, 1), (1, 2), (1, 2), (1, 2)],
        [(3, 0), (0, 3), (2, 1), (1, 2), (1, 2)],
    ]
    bprofiles = [
        [(2, 1)] * 3 + [(1, 2)] * 3,
        [(3, 0), (2, 1)] + [(1, 2)] * 4,
        [(3, 0), (0, 3)] + [(2, 1)] * 2 + [(1, 2)] * 2,
    ]
    design = idx % 20
    if design < 9:
        acounts = aprofiles[design // 3]
        bcounts = bprofiles[design % 3]
    elif design < 18:
        acounts = aprofiles[(design - 9) // 3]
        t = (design - 9) % 3 + 1
        bcounts = ([(1, 1)] * t + [(2, 2)] * t
                   + [(2, 1)] * (3 - t) + [(1, 2)] * (3 - t))
    elif design == 18:
        acounts = aprofiles[0]
        bcounts = [(1, 1), (3, 1), (2, 1)] + [(1, 2)] * 3
    else:
        acounts = aprofiles[2]
        bcounts = [(1, 1), (4, 0)] + [(1, 2)] * 4

    sa = [[col[i] for col in acounts] for i in range(2)]
    sb = [[col[i] for col in bcounts] for i in range(2)]
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)

    def pack(counts):
        """Construct a cyclic pair packing, not relabel an existing code.

        A within-row check-pair orbit uses an unsigned difference;
        a cross-row orbit uses an oriented difference. Never reuse
        either orbit within this component.
        """
        width = len(counts)
        for restart in range(64):
            used = [set(), set()]
            cross_used = set()
            out = [None] * width
            order = list(range(width))
            rng.shuffle(order)
            order.sort(key=lambda j: -sum(counts[j]))
            complete = True
            for j in order:
                a, b = counts[j]
                chosen = None
                for attempt in range(256):
                    x = rng.sample(range(q), a)
                    y = rng.sample(range(q), b)
                    anchor = x[0] if x else y[0]
                    x = sorted((v - anchor) % q for v in x)
                    y = sorted((v - anchor) % q for v in y)
                    local = []
                    legal = True
                    for row, vals in enumerate((x, y)):
                        diffs = []
                        for u in range(len(vals)):
                            for v in range(u + 1, len(vals)):
                                d = (vals[v] - vals[u]) % q
                                diffs.append(min(d, q - d))
                        ds = set(diffs)
                        if len(ds) != len(diffs) or ds & used[row]:
                            legal = False
                            break
                        local.append(ds)
                    if not legal:
                        continue
                    cross = [(u - v) % q for u in x for v in y]
                    cs = set(cross)
                    if len(cs) != len(cross) or cs & cross_used:
                        continue
                    chosen = (x, y)
                    used[0].update(local[0])
                    used[1].update(local[1])
                    cross_used.update(cs)
                    break
                if chosen is None:
                    complete = False
                    break
                out[j] = chosen
            if complete:
                return [[[G.elements[v] for v in out[j][i]]
                         for j in range(width)] for i in range(2)]
        return None

    def syndrome_columns(rows, positions):
        """Read columns of a verified matrix; do not implement a lift."""
        result = []
        for pos in positions:
            syndrome = 0
            for r, row in enumerate(rows):
                if (row >> pos) & 1:
                    syndrome |= 1 << r
            result.append(syndrome)
        return result

    def short_words(columns):
        """Exact rejection of words <=5; count weight-six words.

        Return 7 when no word <=6 exists: this is a screening flag,
        not an assertion that the component distance equals seven.
        After excluding words <=4, colliding triples are disjoint,
        so each weight-six word contributes ten triple collisions.
        """
        low = {}
        for s in columns:
            if not s:
                return (1, 1)
            if s in low:
                return (2, 1)
            low[s] = 1
        length = len(columns)
        for i in range(length):
            x = columns[i]
            for j in range(i + 1, length):
                s = x ^ columns[j]
                if not s:
                    return (2, 1)
                previous = low.get(s, 0)
                if previous:
                    return (3 if previous == 1 else 4, 1)
                low[s] = 2

        triples = {}
        get = triples.get
        collisions = 0
        for i in range(length - 2):
            x = columns[i]
            for j in range(i + 1, length - 1):
                xy = x ^ columns[j]
                for h in range(j + 1, length):
                    s = xy ^ columns[h]
                    if not s:
                        return (3, 1)
                    if s in low:
                        # A singleton collision would imply weight four,
                        # already excluded by the pair enumeration.
                        return (5, 1)
                    previous = get(s, 0)
                    collisions += previous
                    triples[s] = previous + 1
        if collisions:
            return (6, collisions // 10)
        return (7, 0)

    best = None
    best_score = None
    accepted = 0
    # Beyond twenty slots, retain the structural design but enlarge its
    # independently seeded optimization search; do not emit orbit copies.
    quota = 28 + 4 * (idx // 20)
    for trial in range(3 * quota):
        A = pack(acounts)
        B = pack(bcounts)
        if A is None or B is None:
            continue
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k != kb:
            continue

        # HX at Bcol=0 restricts to the left-lifted A component.
        arows = [hx[(i * cb) * q + r]
                 for i in range(ra) for r in range(q)]
        apos = [j * cb * q + r for j in range(ca) for r in range(q)]
        # HZ at Acol=0 restricts to the right-lifted B component.
        brows = hz[:rb * q]
        bpos = list(range(cb * q))
        pa = short_words(syndrome_columns(arows, apos))
        if pa[0] <= 4:
            continue
        pb = short_words(syndrome_columns(brows, bpos))
        if pb[0] <= 4:
            continue

        score = (min(pa[0], pb[0]), pa[0] + pb[0],
                 -max(pa[1], pb[1]), -(pa[1] + pb[1]))
        if best_score is None or score > best_score:
            best_score = score
            best = (A, B)
        accepted += 1
        if accepted >= quota:
            break

    if best is None:
        raise ValueError("no rank-bound candidate passed short-word screening")
    A, B = best
    hx, hz, n = lifted_matrices(G, A, B)
    assert css_ok(hx, hz)
    assert n - gf2_rank(hx) - gf2_rank(hz) == kb
    return {"idx": idx, "cells": {"A": A, "B": B},
            "component_screen": best_score}


from functools import lru_cache


_R40_MODELS = (
    (28, 28, 28),
    (28, 28, 14),
    (28, 14, 14),
    (28, 28, 7),
    (28, 14, 7),
    (14, 14, 7),
)
# Unordered A/B architecture pairs avoid deliberately submitting swaps.
_R40_CARDS = tuple(
    (a, b) for a in range(len(_R40_MODELS))
    for b in range(a, len(_R40_MODELS))
)[:20]


def _r40_orders(G):
    out = {}
    for g in G.elements:
        value = G.e
        for order in range(1, G.q + 1):
            value = G.mul(value, g)
            if value == G.e:
                out[g] = order
                break
        else:
            raise ValueError("invalid group element order")
    return out


def _r40_pack(rng, G, model, pools):
    """Fresh mixed cubics, packed in both directed-difference conventions."""
    for restart in range(32):
        used_left, used_right = set(), set()
        row = [None] * 3
        order = list(range(3))
        rng.shuffle(order)
        for j in order:
            for attempt in range(160):
                long = rng.choice(pools[model[j]])
                mixed = rng.choice(pools[8])
                if G.mul(long, mixed) == G.mul(mixed, long):
                    continue
                anchor = G.pick(rng)
                cell = [anchor, G.mul(anchor, long),
                        G.mul(anchor, mixed)]
                if len(set(cell)) != 3:
                    continue
                left = {
                    G.mul(g, G.inv(h))
                    for g in cell for h in cell if g != h
                }
                right = {
                    G.mul(G.inv(g), h)
                    for g in cell for h in cell if g != h
                }
                if len(left) != 6 or len(right) != 6:
                    continue
                if left & used_left or right & used_right:
                    continue
                used_left.update(left)
                used_right.update(right)
                row[j] = cell
                break
            if row[j] is None:
                break
        if all(cell is not None for cell in row):
            return [row]
    return None


def _r40_columns(rows, width):
    columns = [0] * width
    for r, row in enumerate(rows):
        marker = 1 << r
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= marker
            row ^= bit
    return columns


def _r40_components(hx, hz, q):
    """Both component actions come exclusively from the verified lift."""
    mask = (1 << q) - 1
    ar = [
        sum(((row >> (3*j*q)) & mask) << (j*q)
            for j in range(3))
        for row in hx[:q]
    ]
    br = [row & ((1 << (3*q)) - 1) for row in hz[:q]]
    ac = _r40_columns(ar, 3*q)
    bc = _r40_columns(br, 3*q)
    assert all(c.bit_count() == 3 for c in ac + bc)
    return ac, bc


def _r40_cycles(columns):
    """Actual four-cycles and, when four-free, exact six-cycle count."""
    overlaps = {}
    adjacency = [0] * max(c.bit_length() for c in columns)
    four = 0
    for syndrome in columns:
        neighbors = []
        while syndrome:
            bit = syndrome & -syndrome
            neighbors.append(bit.bit_length() - 1)
            syndrome ^= bit
        for i, a in enumerate(neighbors):
            for b in neighbors[:i]:
                key = (min(a, b), max(a, b))
                old = overlaps.get(key, 0)
                four += old
                overlaps[key] = old + 1
                adjacency[a] |= 1 << b
                adjacency[b] |= 1 << a
    if four:
        return four, 10**9
    triangles = 0
    for a, neighbors in enumerate(adjacency):
        higher = neighbors >> (a + 1)
        while higher:
            bit = higher & -higher
            b = a + 1 + bit.bit_length() - 1
            triangles += (
                (neighbors & adjacency[b]) >> (b + 1)
            ).bit_count()
            higher ^= bit
    # Each cubic variable contributes one non-Tanner triangle.
    return 0, triangles - len(columns)


def _r40_six(columns):
    """Exact component exclusion through six, with six-word multiplicity.

    Cubic columns exclude odd-weight kernels. After weights two and
    four are excluded, equal distinct triples are disjoint.
    """
    if len(set(columns)) != len(columns):
        return 2, 1
    pairs = set()
    for i, x in enumerate(columns):
        for y in columns[:i]:
            syndrome = x ^ y
            if syndrome in pairs:
                return 4, 1
            pairs.add(syndrome)
    del pairs
    triples = {}
    collisions = 0
    for i in range(2, len(columns)):
        x = columns[i]
        for j in range(1, i):
            pair = x ^ columns[j]
            for h in range(j):
                syndrome = pair ^ columns[h]
                previous = triples.get(syndrome, 0)
                collisions += previous
                triples[syndrome] = previous + 1
    return (6, collisions // 10) if collisions else (8, 0)


def _r40_basis(rows):
    basis = {}
    for word in rows:
        while word:
            p = word.bit_length() - 1
            previous = basis.get(p)
            if previous is None:
                basis[p] = word
                break
            word ^= previous
    return basis


def _r40_outside(word, basis):
    while word:
        p = word.bit_length() - 1
        previous = basis.get(p)
        if previous is None:
            return True
        word ^= previous
    return False


def _r40_probe(rng, columns, trials, start=None, stabilizers=None):
    """Actual dependency witnesses, optionally excluding stabilizers.

    A high reading is not a certificate. Fresh information sets and
    cross-trial recombination can only lower an earlier reading.
    """
    best = len(columns) + 1 if start is None else start
    order = list(range(len(columns)))
    memory = []

    def consider(word):
        nonlocal best
        if not word or word.bit_count() >= best:
            return
        if stabilizers is None or _r40_outside(word, stabilizers):
            best = word.bit_count()

    for trial in range(trials):
        rng.shuffle(order)
        pivots, words = {}, []
        for c in order:
            syndrome, word = columns[c], 1 << c
            while syndrome:
                p = syndrome.bit_length() - 1
                previous = pivots.get(p)
                if previous is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= previous[0]
                word ^= previous[1]
            if not syndrome:
                words.append(word)
                consider(word)

        short = sorted(set(words + memory), key=int.bit_count)[:36]
        discoveries = set(short)
        for i, word in enumerate(short):
            for other in short[:i]:
                combined = word ^ other
                if combined:
                    consider(combined)
                    if combined.bit_count() <= best + 4:
                        discoveries.add(combined)

        reducers = words + memory
        for seed in sorted(discoveries, key=int.bit_count)[:6]:
            word = seed
            for sweep in range(3):
                chosen, weight = word, word.bit_count()
                for reducer in reducers:
                    combined = word ^ reducer
                    if combined and combined.bit_count() < weight:
                        chosen, weight = combined, combined.bit_count()
                if chosen == word:
                    break
                word = chosen
                consider(word)
                discoveries.add(word)
        memory = sorted(discoveries, key=int.bit_count)[:36]
        if best <= 4:
            break
    return best


def _r40_fingerprint(hx, hz, n):
    """Invariant of the complete, uncoloured qubit/check incidence graph.

    Different fingerprints certify nonisomorphism. Equal fingerprints
    are conservatively rejected; they need not certify isomorphism.
    """
    rows = hx + hz
    columns = _r40_columns(rows, n)

    def profiles(vectors):
        result = []
        for i, vector in enumerate(vectors):
            histogram = {}
            for j, other in enumerate(vectors):
                if i == j:
                    continue
                key = (other.bit_count(), (vector & other).bit_count())
                histogram[key] = histogram.get(key, 0) + 1
            result.append(
                (vector.bit_count(), tuple(sorted(histogram.items())))
            )
        return tuple(sorted(result))

    return profiles(rows), profiles(columns)


@lru_cache(maxsize=1)
def _r40_bank(G):
    """Deterministic recipe bank with joint rank and incidence screening."""
    assert G.q == 56
    orders = _r40_orders(G)
    pools = {
        order: [g for g in G.elements if orders[g] == order]
        for order in (7, 8, 14, 28)
    }
    assert all(pools.values())
    sa = sb = [[3, 3, 3]]
    assert _grid_weights(sa, sb) == 12
    assert _no_dead_lines(sa, sb)
    fingerprints = set()
    bank = []

    for slot in range(N_SLOTS):
        a_card, b_card = _R40_CARDS[slot % 20]
        am, bm = _R40_MODELS[a_card], _R40_MODELS[b_card]
        rng = random.Random(40056013 + 100003 * slot)
        chosen = None

        for batch in range(3):
            raw = []
            exact_seen = set()
            for attempt in range(48):
                A = _r40_pack(rng, G, am, pools)
                B = _r40_pack(rng, G, bm, pools)
                if A is None or B is None:
                    continue
                signature = tuple(
                    tuple(sorted(cell))
                    for grid in (A, B) for row in grid for cell in row
                )
                if signature in exact_seen:
                    continue
                exact_seen.add(signature)
                hx, hz, n = lifted_matrices(G, A, B)
                ac, bc = _r40_components(hx, hz, G.q)
                ca, cb = _r40_cycles(ac), _r40_cycles(bc)
                if ca[0] or cb[0]:
                    continue
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if not (375*n <= 1000*k < 425*n):
                    continue
                item = {
                    "A": A, "B": B, "hx": hx, "hz": hz,
                    "n": n, "k": k, "ac": ac, "bc": bc,
                    "six": (ca[1], cb[1]),
                }
                raw.append(item)
                raw.sort(key=lambda x: (
                    sum(x["six"]), max(x["six"]), -x["k"]
                ))
                del raw[8:]

            if not raw:
                continue
            for item in raw:
                item["da"] = _r40_probe(rng, item["ac"], 8)
                item["db"] = _r40_probe(rng, item["bc"], 8)

            def preliminary(item):
                d = min(item["da"], item["db"], 20)
                return (
                    item["k"] * d*d / item["n"],
                    min(item["da"], item["db"]),
                    -sum(item["six"]),
                )

            finalists = sorted(
                raw, key=preliminary, reverse=True
            )[:4]
            for item in finalists:
                fa, ha = _r40_six(item["ac"])
                fb, hb = _r40_six(item["bc"])
                item["floor"] = min(fa, fb)
                item["hits"] = ha + hb
                item["da"] = _r40_probe(
                    rng, item["ac"], 48, item["da"]
                )
                item["db"] = _r40_probe(
                    rng, item["bc"], 48, item["db"]
                )
                if fa < 8:
                    item["da"] = min(item["da"], fa)
                if fb < 8:
                    item["db"] = min(item["db"], fb)

                # Complete-code witnesses include both LP sectors.
                xs = _r40_basis(item["hx"])
                zs = _r40_basis(item["hz"])
                xc = _r40_columns(item["hz"], item["n"])
                zc = _r40_columns(item["hx"], item["n"])
                dx = _r40_probe(
                    rng, xc, 12, stabilizers=xs
                )
                dz = _r40_probe(
                    rng, zc, 12, stabilizers=zs
                )
                item["full"] = min(dx, dz)

            def final_merit(item):
                d = min(item["da"], item["db"], item["full"])
                return (
                    item["floor"],
                    item["k"] * d*d / item["n"],
                    d, -item["hits"], -sum(item["six"]),
                )

            for item in sorted(
                finalists, key=final_merit, reverse=True
            ):
                fingerprint = _r40_fingerprint(
                    item["hx"], item["hz"], item["n"]
                )
                if fingerprint in fingerprints:
                    continue
                assert css_ok(item["hx"], item["hz"])
                assert max(
                    row.bit_count()
                    for row in item["hx"] + item["hz"]
                ) <= W_MAX
                assert all(
                    len(cell) == 3 and len(set(cell)) == 3
                    for grid in (item["A"], item["B"])
                    for row in grid for cell in row
                )
                fingerprints.add(fingerprint)
                chosen = (item["A"], item["B"])
                break
            if chosen is not None:
                break
        bank.append(chosen)

    return tuple(bank)


_F39_A_CARDS = (
    (1, 1, 1), (1, 1, 2), (1, 2, 3),
    (1, 2, 4), (1, 3, 5),
)
_F39_B_CARDS = (
    (1, 1, 3), (1, 2, 5), (1, 3, 4), (1, 2, 6),
)


def _f39_clean4(cols):
    """Exact component-kernel exclusion through four for cubic columns."""
    if 0 in cols or len(set(cols)) != len(cols):
        return False
    pairs = set()
    for i, value in enumerate(cols):
        for other in cols[:i]:
            syndrome = value ^ other
            if syndrome in pairs:
                return False
            pairs.add(syndrome)
    # Odd-weight columns exclude every odd-weight kernel word.
    return True


def _f39_low4(cols, stabilizers):
    """Exact logical existence through four, including both LP sectors."""
    span = _r40_basis(stabilizers)
    seen = {0: 0}
    for i, syndrome in enumerate(cols):
        word = 1 << i
        previous = seen.get(syndrome)
        if previous is None:
            seen[syndrome] = word
        elif _r40_outside(word ^ previous, span):
            return True
    for i, syndrome in enumerate(cols):
        for j in range(i):
            value = syndrome ^ cols[j]
            word = (1 << i) | (1 << j)
            previous = seen.get(value)
            if previous is None:
                seen[value] = word
            elif _r40_outside(word ^ previous, span):
                return True
    return False


def _f39_fingerprint(hx, hz, n):
    """Uncoloured incidence invariant; unequal signatures imply diversity.

    Equality is rejected conservatively, not asserted to prove isomorphism.
    Check-intersection triangles supplement the parent pair profiles.
    """
    from collections import Counter

    rows = list(hx) + list(hz)
    weights = [row.bit_count() for row in rows]
    adjacency = [0] * len(rows)
    overlaps = [dict() for _ in rows]
    for i, row in enumerate(rows):
        for j in range(i):
            size = (row & rows[j]).bit_count()
            if size:
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i
                overlaps[i][j] = overlaps[j][i] = size
    profiles = []
    for i, neighbors in enumerate(overlaps):
        histogram = Counter(
            (weights[j], multiplicity,
             (adjacency[i] & adjacency[j]).bit_count())
            for j, multiplicity in neighbors.items())
        profiles.append((weights[i], tuple(sorted(histogram.items()))))
    degrees = Counter(
        c.bit_count() for c in _r40_columns(rows, n))
    return (n, tuple(sorted(degrees.items())),
            tuple(sorted(Counter(profiles).items())))


@lru_cache(maxsize=1)
def _f39_bank(G):
    """Fresh quotient-balanced designs; all historical banks are inactive."""
    assert G.q == 39 and A_SHAPE == B_SHAPE == (1, 3)
    q, width = G.q, 3 * G.q
    mask = (1 << q) - 1
    zero = [[[], [], []]]
    atlas = {}

    # Compile actions exclusively from verified single-term full lifts.
    for side in ("A", "B"):
        blocks = []
        for g in G.elements:
            unit = [[[g], [], []]]
            A, B = (unit, zero) if side == "A" else (zero, unit)
            hx, hz, _ = lifted_matrices(G, A, B)
            source = hx if side == "A" else hz
            blocks.append(tuple(_r40_columns(
                [row & mask for row in source[:q]], q)))
        atlas[side] = tuple(blocks)

    def compile_component(side, grid):
        cols = [0] * width
        for j, cell in enumerate(grid[0]):
            for g in cell:
                contribution = atlas[side][G.idx(g)]
                for t, syndrome in enumerate(contribution):
                    cols[j*q+t] ^= syndrome
        return tuple(cols)

    pools = {}

    def component_pool(side, model):
        key = (side, model)
        if key in pools:
            return pools[key]
        local = random.Random(
            39045019 + (0 if side == "A" else 1000003)
            + sum(d * 17**j for j, d in enumerate(model)))
        raw = []

        # Cell e + (a,1) + (b,2), with a+3b=d != 0.
        # This condition makes its two nonidentity terms noncommuting.
        # Fix the first a-coordinate to zero as a design constraint,
        # then exhaust all 169 relative couplings in the other cells.
        for a1 in range(13):
            for a2 in range(13):
                aa = (0, a1, a2)
                grid = [[
                    [G.e, (a, 1), (9*(d-a) % 13, 2)]
                    for a, d in zip(aa, model)]]
                cols = compile_component(side, grid)
                assert all(c.bit_count() == 3 for c in cols)
                if gf2_rank(cols) != q - 2:
                    continue
                if not _f39_clean4(cols):
                    continue
                four, six_cycles = _r40_cycles(cols)
                raw.append(dict(
                    grid=grid, cols=cols, four=four,
                    six_cycles=six_cycles, labels=aa))
        local.shuffle(raw)
        raw.sort(key=lambda item:
                 (item["four"], item["six_cycles"]))
        candidates = raw[:32]

        for item in candidates:
            floor, hits = _r40_six(item["cols"])
            item["floor"], item["hits"] = floor, hits
            reading = _r40_probe(local, item["cols"], 4)
            if floor < 8:
                reading = min(reading, floor)
            item["reading"] = reading

        def merit(item):
            return (item["floor"], min(item["reading"], 14),
                    -item["hits"], -item["four"],
                    -item["six_cycles"])

        candidates.sort(key=merit, reverse=True)
        finalists = candidates[:12]
        # Give every retained component the same independent deeper budget.
        for item in finalists:
            item["reading"] = _r40_probe(
                local, item["cols"], 32, item["reading"])
        finalists.sort(key=merit, reverse=True)
        pools[key] = tuple(finalists)
        return pools[key]

    assert _grid_weights([[3, 3, 3]], [[3, 3, 3]]) == 12
    assert _no_dead_lines([[3, 3, 3]], [[3, 3, 3]])
    emitted, bank = set(), []

    for slot in range(N_SLOTS):
        architecture, mode = divmod(slot % 20, 4)
        am = _F39_A_CARDS[architecture]
        bm = list(_F39_B_CARDS[mode])
        lane = slot // 20
        if lane:
            # Extra slots change a relative ratio on B only,
            # rather than transporting one code by an automorphism.
            bm[-1] = (bm[-1] + lane) % 13 or 1
        bm = tuple(bm)
        ap = component_pool("A", am)
        bp = component_pool("B", bm)
        local = random.Random(39045173 + 104729 * slot)
        proposals = []
        for a in ap:
            for b in bp:
                d = min(a["reading"], b["reading"], 14)
                key = (min(a["floor"], b["floor"]), d,
                       -a["hits"]-b["hits"],
                       -a["four"]-b["four"])
                proposals.append((key, a, b))
        local.shuffle(proposals)
        proposals.sort(key=lambda item: item[0], reverse=True)

        finalists, pending = [], set()
        for key, a, b in proposals:
            A, B = a["grid"], b["grid"]
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if n != 390 or not (425*n <= 1000*k < 475*n):
                continue

            # Validate the compiled actions against the actual full lift.
            ar = [
                sum(((row >> (3*j*q)) & mask) << (j*q)
                    for j in range(3))
                for row in hx[:q]]
            br = [row & ((1 << width)-1) for row in hz[:q]]
            assert tuple(_r40_columns(ar, width)) == a["cols"]
            assert tuple(_r40_columns(br, width)) == b["cols"]

            xc = _r40_columns(hz, n)
            zc = _r40_columns(hx, n)
            if _f39_low4(xc, hx) or _f39_low4(zc, hz):
                continue
            fp = _f39_fingerprint(hx, hz, n)
            if fp in emitted or fp in pending:
                continue
            pending.add(fp)
            dx = _r40_probe(
                local, xc, 8, stabilizers=_r40_basis(hx))
            dz = _r40_probe(
                local, zc, 8, stabilizers=_r40_basis(hz))
            if dx > n or dz > n:
                continue
            finalists.append(dict(
                A=A, B=B, hx=hx, hz=hz, n=n, k=k,
                a=a, b=b, xc=xc, zc=zc, dx=dx, dz=dz, fp=fp))
            if len(finalists) >= 3:
                break

        # Equal deeper budgets prevent an undeepened optimistic finalist
        # from winning. Only actual non-stabilizer words lower these probes.
        for item in finalists:
            item["dx"] = _r40_probe(
                local, item["xc"], 20, item["dx"],
                _r40_basis(item["hx"]))
            item["dz"] = _r40_probe(
                local, item["zc"], 20, item["dz"],
                _r40_basis(item["hz"]))

        def joint_merit(item):
            a, b = item["a"], item["b"]
            d = min(a["reading"], b["reading"],
                    item["dx"], item["dz"], 14)
            # Component readings are conservative selection heuristics:
            # their words need not remain logical after embedding.
            return (item["k"]*d*d/item["n"], d,
                    min(a["floor"], b["floor"]),
                    -a["hits"]-b["hits"], -a["four"]-b["four"])

        if not finalists:
            bank.append(None)
            continue
        winner = max(finalists, key=joint_merit)
        A, B = winner["A"], winner["B"]
        assert css_ok(winner["hx"], winner["hz"])
        assert max(row.bit_count()
                   for row in winner["hx"]+winner["hz"]) == 12
        assert all(len(cell) == len(set(cell)) == 3
                   for grid in (A, B) for row in grid for cell in row)
        assert 425*390 <= 1000*winner["k"] < 475*390
        emitted.add(winner["fp"])
        frozen = {
            side: tuple(tuple(tuple(cell) for cell in row) for row in grid)
            for side, grid in (("A", A), ("B", B))}
        bank.append(dict(
            cells=frozen, k=winner["k"],
            card=(architecture, mode, lane),
            selection=joint_merit(winner)))
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Return a fresh target-band design, never a historical-bank fallback."""
    selected = _f39_bank(G)[idx]
    if selected is None:
        raise ValueError("no fresh quotient-balanced 0.450 filling passed")
    return {
        "idx": idx, "k": selected["k"], "card": selected["card"],
        "selection": selected["selection"],
        "cells": {
            side: [[list(cell) for cell in row] for row in grid]
            for side, grid in selected["cells"].items()}}


def CELL(rng, G, side, i, j, ctx):
    """Return the selected complete cell without further randomness."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled supports with uniform labels do not reliably eliminate weight-5 sector-1 logicals on Z17 (2,5)x(2,6): all five measured parent slots have d30s=5.
# REFUTED: Balanced cyclic difference packing and component screening through six remove the weight-eight plateau on Z17 (2,5)x(2,6): all five measured current-parent codes instead have d30s=8, with minimum witnesses exclusively in sector 1.
# REFUTED: Noncommuting order-eight relatives, two-orientation pair packing and joint witness selection yield five distance-ten readings on C7 semidirect C8, (1,3)x(1,3); all five measured parent leaders instead had k=224 and d30s=8, with minima confined to sector one.
# EVOLVE-BLOCK-END


class _G(object):
    def __init__(self, ops, q):
        self.q = q
        self.elements = list(ops["elements"])
        self.e = ops["ident"]
        self.mul = ops["mul"]
        self.inv = ops["inv"]
        self._pos = {self._key(g): i for i, g in enumerate(self.elements)}

    @staticmethod
    def _key(g):
        return tuple(g) if isinstance(g, (list, tuple)) else g

    def idx(self, g):
        return self._pos[self._key(g)]

    def pick(self, rng):
        return self.elements[rng.randrange(self.q)]


# ---- verified construction helpers (2026-09-09) --------------------------
# lifted_matrices(G, cells_A, cells_B) builds EXACTLY the HX / HZ the
# evaluator builds (same lift conventions, same column order), as Python-int
# bit rows; gf2_rank(rows) is the rank behind k; lifted_k(G, A, B) is the k the table will report;
# css_ok(hx, hz) checks HX HZ^T = 0. Use these for any in-program screening -
# a hand-written lift that gets the left/right actions wrong screens a
# different (usually non-CSS) matrix pair than the one that is measured.
def lifted_matrices(G, cells_A, cells_B):
    """cells_A[i][j], cells_B[a][b]: lists of group elements (as CELL returns
    them) or element indices. Returns (hx_rows, hz_rows, n): lists of Python
    ints, bit c set = a 1 in column c; n = number of columns (qubits)."""
    q = G.q
    ra, ca = len(cells_A), len(cells_A[0])
    rb, cb = len(cells_B), len(cells_B[0])
    els = G.elements

    def gi(g):
        try:
            return G.idx(g)
        except (KeyError, TypeError):
            if isinstance(g, int) and 0 <= g < q:
                return g
            raise
    mul_idx = [[G.idx(G.mul(els[x], els[y])) for y in range(q)] for x in range(q)]
    inv_idx = [G.idx(G.inv(els[x])) for x in range(q)]
    A = [[sorted(set(gi(g) for g in cell)) for cell in row] for row in cells_A]
    B = [[sorted(set(gi(g) for g in cell)) for cell in row] for row in cells_B]
    n1 = ca * cb * q
    n = n1 + ra * rb * q
    hx = []
    for i in range(ra):
        for beta in range(cb):
            for r in range(q):
                row = 0
                for j in range(ca):                       # sector 1: A_{ij} left-lifted
                    base = (j * cb + beta) * q
                    for g in A[i][j]:
                        row ^= 1 << (base + mul_idx[inv_idx[g]][r])   # h with g*h = r
                for alpha in range(rb):                   # sector 2: B_{alpha beta}^T right-lifted
                    base = n1 + (i * rb + alpha) * q
                    for g in B[alpha][beta]:
                        row ^= 1 << (base + mul_idx[r][inv_idx[g]])   # c = r*g^-1
                hx.append(row)
    hz = []
    for j in range(ca):
        for alpha in range(rb):
            for r in range(q):
                row = 0
                for beta in range(cb):                    # sector 1: B_{alpha beta} right-lifted
                    base = (j * cb + beta) * q
                    for g in B[alpha][beta]:
                        row ^= 1 << (base + mul_idx[r][g])           # h with h*g^-1 = r
                for i in range(ra):                       # sector 2: A_{ij}^T left-lifted
                    base = n1 + (i * rb + alpha) * q
                    for g in A[i][j]:
                        row ^= 1 << (base + mul_idx[g][r])           # c = g*r
                hz.append(row)
    return hx, hz, n


def _popcount(x):
    return bin(x).count("1")



def gf2_rank(rows):
    """Rank over GF(2) of a list of bit rows."""
    pivots = {}
    for row in rows:
        while row:
            p = row.bit_length() - 1
            if p in pivots:
                row ^= pivots[p]
            else:
                pivots[p] = row
                break
    return len(pivots)


def lifted_k(G, cells_A, cells_B):
    """k = n - rank(HX) - rank(HZ) of the code the evaluator will build."""
    hx, hz, n = lifted_matrices(G, cells_A, cells_B)
    return n - gf2_rank(hx) - gf2_rank(hz)


def css_ok(hx, hz):
    """True iff every X check commutes with every Z check (HX HZ^T = 0)."""
    return all(_popcount(x & z) % 2 == 0 for x in hx for z in hz)


def _grid_weights(sa, sb):
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    wx = max(sum(sa[i]) + sum(sb[r_][j] for r_ in range(rb))
             for i in range(ra) for j in range(cb))
    wz = max(sum(sb[r_]) + sum(sa[i][c] for i in range(ra))
             for r_ in range(rb) for c in range(ca))
    return max(wx, wz)


def _no_dead_lines(sa, sb):
    return all(
        all(sum(row) > 0 for row in g)
        and all(sum(r_[c] for r_ in g) > 0 for c in range(len(g[0])))
        for g in (sa, sb))


def roll_supports(rng):
    """Random per-cell term counts (1..CELL_MAX), repaired down to the
    weight cap with 1 as the floor - the roller never emits zero blocks
    (a slot's CELL may still return [] deliberately). A scaffold
    utility available to NEW_CTX; always returns legal grids."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    sa = [[rng.randint(1, CELL_MAX) for _ in range(ca)] for _ in range(ra)]
    sb = [[rng.randint(1, CELL_MAX) for _ in range(cb)] for _ in range(rb)]
    for _ in range(4000):
        if _grid_weights(sa, sb) <= W_MAX:
            return sa, sb
        g = sa if rng.random() < 0.5 else sb
        i = rng.randrange(len(g))
        c = rng.randrange(len(g[0]))
        if g[i][c] >= 2:            # 1 is the floor: the roller never
            g[i][c] -= 1            # emits zero blocks (evolution may)
    raise ValueError("roll_supports could not fit the weight cap "
                     "(all-ones row exceeds it: shape too wide)")


def _registry_ops(gtype, q, a0):
    if gtype == "cyclic":
        return dict(elements=list(range(q)), ident=0,
                    mul=lambda g, h: (g + h) % q,
                    inv=lambda g: (-g) % q)
    if gtype == "abelian":
        a = int(a0)
        if a < 1 or q % a:
            raise ValueError("A0 must divide q")
        b = q // a
        return dict(elements=[(x, y) for x in range(a) for y in range(b)],
                    ident=(0, 0),
                    mul=lambda g, h: ((g[0] + h[0]) % a, (g[1] + h[1]) % b),
                    inv=lambda g: ((-g[0]) % a, (-g[1]) % b))
    raise ValueError("GTYPE %r needs CUSTOM_GROUP" % gtype)


def _make_ops(q):
    if GTYPE == "custom":
        ns = {}
        exec(CUSTOM_GROUP, ns)
        if not ns["valid_q"](q):
            raise ValueError("valid_q rejects q=%d" % q)
        return ns["custom_ops"](q)
    return _registry_ops(GTYPE, q, A0)


def _valid_q(q):
    if GTYPE == "custom":
        ns = {}
        exec(CUSTOM_GROUP, ns)
        return bool(ns["valid_q"](q))
    if GTYPE == "abelian":
        return A0 is not None and int(A0) >= 1 and q % int(A0) == 0
    return q >= 2


def _validate_group(G):
    rng = random.Random(7)
    sample = [G.e] + [G.pick(rng) for _ in range(min(12, G.q))]
    for g in sample:
        assert G._key(G.mul(G.e, g)) == G._key(g), "identity fails"
        assert G._key(G.mul(g, G.inv(g))) == G._key(G.e), "inverse fails"
    for g in sample[:6]:
        for h in sample[:6]:
            for k in sample[:6]:
                assert G._key(G.mul(G.mul(g, h), k)) \
                    == G._key(G.mul(g, G.mul(h, k))), "associativity fails"
    assert len(G._pos) == len(G.elements) == G.q, "element list broken"


def _choose_q():
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    if min(ra, ca, rb, cb) < 1:
        raise ValueError("protograph sides need >= 1 row and >= 1 column")
    f = ca * cb + ra * rb
    best = None
    for q in range(max(2, N_MIN // f), N_MAX // f + 1):
        if f * q < N_MIN or f * q >= N_MAX:
            continue
        if _valid_q(q):
            best = q
    if best is None:
        raise ValueError("no valid q puts n in [%d, %d)" % (N_MIN, N_MAX))
    return best


def _fill_slot(rng, G, idx):
    ctx = NEW_CTX(rng, G, idx)
    grids = {}
    rgrids = {}
    for side, (rows, cols) in (("A", A_SHAPE), ("B", B_SHAPE)):
        grid = []
        rgrid = []
        for i in range(rows):
            row = []
            rrow = []
            for j in range(cols):
                cell = CELL(rng, G, side, i, j, ctx)
                if not isinstance(cell, (list, tuple)):
                    raise ValueError("%s[%d][%d]: not a list" % (side, i, j))
                if len(cell) > CELL_MAX:
                    raise ValueError("%s[%d][%d]: >%d terms"
                                     % (side, i, j, CELL_MAX))
                keys = {G._key(g) for g in cell}
                if len(keys) != len(cell):
                    raise ValueError("%s[%d][%d]: duplicate element"
                                     % (side, i, j))
                pairs = sorted((G.idx(g), G._key(g)) for g in cell)
                row.append([i for i, _ in pairs])
                rrow.append([list(e) if isinstance(e, tuple) else e
                             for _, e in pairs])
            grid.append(row)
            rgrid.append(rrow)
        grids[side] = grid
        rgrids[side] = rgrid
    sa = [[len(c) for c in row] for row in grids["A"]]
    sb = [[len(c) for c in row] for row in grids["B"]]
    w = _grid_weights(sa, sb)
    if w > W_MAX:
        raise ValueError("row weight %d exceeds cap %d" % (w, W_MAX))
    if not _no_dead_lines(sa, sb):
        raise ValueError("all-zero protograph row or column")
    return grids["A"], grids["B"], rgrids["A"], rgrids["B"]


def get_candidates():
    """-> request dict for the worker: shared group data + 20 slots."""
    q = _choose_q()
    ops = _make_ops(q)
    G = _G(ops, q)
    _validate_group(G)
    mul_table = [[G.idx(G.mul(G.elements[a], G.elements[b]))
                  for b in range(q)] for a in range(q)]
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    slots = []
    for s in range(N_SLOTS):
        rng = random.Random(BASE_SEED * 10007 + s)
        try:
            A, B, Ar, Br = _fill_slot(rng, G, s)
            slots.append({"idx": s, "A": A, "B": B, "Ar": Ar, "Br": Br})
        except Exception as e:
            slots.append({"idx": s, "error": str(e)[:120]})
    return {"family": FAMILY_NAME, "concept": CONCEPT,
            "gtype": GTYPE, "q": q, "mul": mul_table,
            "a_shape": [ra, ca], "b_shape": [rb, cb],
            "n": (ca * cb + ra * rb) * q,
            "slots": slots}


if __name__ == "__main__":
    r = get_candidates()
    ok = sum(1 for s in r["slots"] if "error" not in s)
    print(json.dumps({"family": r["family"], "q": r["q"], "n": r["n"],
                      "slots_ok": ok}))
