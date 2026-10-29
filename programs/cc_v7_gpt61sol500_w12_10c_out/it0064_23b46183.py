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
CONCEPT = ("On the Frobenius group C11 semidirect C5 with action 3, order 55, shapes (1,3)x(1,3), balanced mixed trinomials selected against weight-four and weight-six component kernels should improve the parent's five distance-four readings.")
GROUP = ("Jump from the order-58 dihedral group to C11 semidirect C5. The related order-57 quotient-three nonabelian bed reached fitness 24.4 on these shapes, and structured dihedral fillings reached 25.6. Since 3 has order five modulo 11, this group tests a larger faithful complement without involutions; better quantum distance remains a hypothesis.")
SHAPES = ("Retain equal (1,3) sides for cell 0.400: q=55, n=550, guaranteed k>=220 and rate>=0.400. Both components have 165 variables and 55 checks. Parent measured minima occupied sector-1 Acol/Bcol blocks, motivating simultaneous screening of both components. Full component row ranks force the exact bound k=220.")
FILLING = ("All six cells contain identity, a nonidentity rotation and one nonrotation: three distinct terms, with both quantum check weights exactly 12. Twenty action-population/gap cards prescribe different architectures. Extract components from verified lifts; select full-row-rank fillings using exact weight-four exclusion, capped weight-six collision counts and randomized actual kernel witnesses. Screen final complete matrices through weight four, verify exact k and CSS, and retain a legal fallback if the stronger screen fails. Component probes are upper bounds, not quantum-distance certificates.")
FAMILY_NAME = "r040_frob11x5_13x13_actioncards_kernel6"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 55

def custom_ops(q):
    # x^11 = y^5 = 1; y*x*y^-1 = x^3.
    # 3 has exact order five modulo 11.
    if not valid_q(q):
        raise ValueError("this family requires order 55")
    powers = (1, 3, 9, 5, 4)
    els = [(a, b) for b in range(5) for a in range(11)]

    def red(g):
        return (g[0] % 11, g[1] % 5)

    def mul(g, h):
        g, h = red(g), red(h)
        return ((g[0] + powers[g[1]] * h[0]) % 11,
                (g[1] + h[1]) % 5)

    def inv(g):
        g = red(g)
        b = (-g[1]) % 5
        return ((-powers[b] * g[0]) % 11, b)

    def idx(g):
        a, b = red(g)
        return 11 * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


# Ten unordered action-population pairs crossed with two gap architectures.
# Further harness slots change the prescribed gap architecture as well.
_ACTION_CARDS = tuple((a, b) for a in range(4) for b in range(a, 4))
_ROTATION_CARDS = (
    (1, 2, 3), (1, 2, 4), (1, 3, 4),
    (1, 3, 5), (1, 4, 5), (2, 3, 5),
)


def _f55_columns(rows, width):
    """Transpose only matrices obtained from the verified constructor."""
    cols = [0] * width
    for r, row in enumerate(rows):
        bit = 1 << r
        while row:
            low = row & -row
            cols[low.bit_length() - 1] |= bit
            row ^= low
    return cols


def _f55_free4(cols):
    """Exactly exclude kernels through weight four for odd columns."""
    if any(c.bit_count() % 2 != 1 for c in cols):
        return False
    if 0 in cols or len(set(cols)) != len(cols):
        return False
    seen = set()
    for i, x in enumerate(cols):
        for j in range(i):
            s = x ^ cols[j]
            if s in seen:
                return False
            seen.add(s)
    return True


def _f55_cycles(cols):
    """Count four-cycles through repeated check-neighbor pairs."""
    seen = {}
    total = 0
    for col in cols:
        neighbors = []
        while col:
            low = col & -col
            neighbors.append(low.bit_length() - 1)
            col ^= low
        for i, a in enumerate(neighbors):
            for b in neighbors[:i]:
                pair = (b, a)
                old = seen.get(pair, 0)
                total += old
                seen[pair] = old + 1
    return total


def _f55_six(cols, cap):
    """Capped triple collisions after exact exclusion of weights <=4.

    Colliding triples are then disjoint. Ten unordered triple partitions
    represent each weight-six kernel support. Zero exhausts this test;
    cap+1 records an exceeded budget, not an exact multiplicity.
    """
    seen = {}
    hits = 0
    for i in range(2, len(cols)):
        x = cols[i]
        for j in range(1, i):
            xy = x ^ cols[j]
            for t in range(j):
                s = xy ^ cols[t]
                old = seen.get(s, 0)
                hits += old
                if hits > cap:
                    return cap + 1
                seen[s] = old + 1
    return hits


def _f55_probe(rng, cols, rounds):
    """Find actual component kernel witnesses; no lower-bound claim."""
    best = len(cols) + 1

    def record(word):
        nonlocal best
        if word:
            best = min(best, word.bit_count())

    for _ in range(rounds):
        order = list(range(len(cols)))
        rng.shuffle(order)
        pivots = {}
        words = []
        for j in order:
            syndrome, word = cols[j], 1 << j
            while syndrome:
                p = syndrome.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                words.append(word)
                record(word)
        for i, word in enumerate(words):
            for other in words[:i]:
                record(word ^ other)
        words.sort(key=int.bit_count)
        small = words[:14]
        for i, word in enumerate(small):
            for j in range(i):
                pair = word ^ small[j]
                for t in range(j):
                    record(pair ^ small[t])
    return best


def _f55_component(G, M, side):
    """Extract the constituent using the verified LP conventions."""
    dummy = [[[G.e] for _ in range(3)]]
    A, B = (M, dummy) if side == "A" else (dummy, M)
    hx, hz, _ = lifted_matrices(G, A, B)
    q = G.q
    if side == "A":
        mask = (1 << q) - 1
        rows = [
            sum(((row >> (j * 3 * q)) & mask) << (j * q)
                for j in range(3))
            for row in hx[:q]
        ]
    else:
        mask = (1 << (3 * q)) - 1
        rows = [row & mask for row in hz[:q]]
    return rows, _f55_columns(rows, 3 * q)


def _f55_select(rng, G, side, gaps, actions):
    """Optimize one prescribed architecture, never transform a winner."""
    pool = []
    reserve = None
    best_six = None
    accepted = 0
    for attempt in range(160):
        row = []
        for gap, action in zip(gaps, actions):
            # Independent orientations change relative gap geometry.
            rotation = (gap * rng.choice((-1, 1))) % 11
            row.append([G.e, (rotation, 0),
                        (rng.randrange(11), action)])
        M = [row]
        rows, cols = _f55_component(G, M, side)
        if gf2_rank(rows) != G.q:
            continue

        cycles = _f55_cycles(cols)
        clean4 = _f55_free4(cols)
        reserve_key = (int(clean4), -cycles)
        if reserve is None or reserve_key > reserve[0]:
            reserve = (reserve_key, M)
        if not clean4:
            continue

        accepted += 1
        cap = 3000 if best_six is None else best_six
        six = _f55_six(cols, cap)
        if best_six is None or six < best_six:
            best_six = six
        # A nonzero triple collision explicitly supplies a six-word.
        # Otherwise probe beyond the exact six-support exclusion.
        reading = 6 if six else _f55_probe(rng, cols, 18)
        key = (int(six == 0), reading, -six, -cycles)
        pool.append((key, M))
        pool.sort(key=lambda item: item[0], reverse=True)
        if len(pool) > 4:
            pool.pop()
        if accepted >= 12 or (best_six == 0 and accepted >= 8):
            break

    if pool:
        return pool
    if reserve is None:
        raise ValueError("no full-row-rank constituent for action card")
    return [((-1, 0, -1000000, reserve[0][1]), reserve[1])]


def NEW_CTX(rng, G, idx):
    """Twenty architectural cards followed by bounded verified selection."""
    assert G.q == 55
    assert A_SHAPE == B_SHAPE == (1, 3)
    design = idx % 20
    population_a, population_b = _ACTION_CARDS[design % 10]
    extension = idx // 20
    palette = (2 * (design // 10) + extension) % len(_ROTATION_CARDS)
    gaps_a = _ROTATION_CARDS[palette]
    gaps_b = _ROTATION_CARDS[(palette + 1) % len(_ROTATION_CARDS)]
    actions_a = [1] * (3 - population_a) + [2] * population_a
    actions_b = [1] * (3 - population_b) + [2] * population_b
    if extension % 3 == 1:
        actions_b[-1] = (-actions_b[-1]) % 5
    elif extension % 3 == 2:
        actions_a[0] = (-actions_a[0]) % 5

    sa = sb = [[3, 3, 3]]
    assert _grid_weights(sa, sb) == 12
    assert _no_dead_lines(sa, sb)
    pa = _f55_select(rng, G, "A", gaps_a, actions_a)
    pb = _f55_select(rng, G, "B", gaps_b, actions_b)

    pairs = []
    for ka, A in pa:
        for kb, B in pb:
            key = (min(ka[0], kb[0]), ka[0] + kb[0],
                   min(ka[1], kb[1]), ka[1] + kb[1],
                   ka[2] + kb[2], ka[3] + kb[3])
            pairs.append((key, A, B))
    pairs.sort(key=lambda item: item[0], reverse=True)

    fallback = None
    for key, A, B in pairs:
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        # Full constituent row ranks should attain both quantum row ranks.
        assert n == 550 and k == 220
        if fallback is None:
            fallback = (A, B, hx, hz, key)
        if not _f55_free4(_f55_columns(hx, n)):
            continue
        if not _f55_free4(_f55_columns(hz, n)):
            continue
        assert css_ok(hx, hz)
        return {"idx": idx, "cells": {"A": A, "B": B},
                "k": k, "full_free4": True, "selection": key}

    # A legal, exact-rate fallback is preferable to voiding this card.
    A, B, hx, hz, key = fallback
    assert css_ok(hx, hz)
    return {"idx": idx, "cells": {"A": A, "B": B},
            "k": 220, "full_free4": False, "selection": key}


def CELL(rng, G, side, i, j, ctx):
    """Return the complete selected cell without further random filling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal rolled supports with uniform dihedral labels reliably suppress short logicals on (1,3)x(1,3); many parent slots found d2s=2 and all five measured parent codes retained d30s=4.
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
