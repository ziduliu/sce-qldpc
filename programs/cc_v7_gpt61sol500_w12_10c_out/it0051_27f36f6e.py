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
CONCEPT = ("For cyclic Z19, order 19, encoded by metacyclic Z1:Z19, shapes (2,5)x(2,5), canonical balanced support pairs and deeper constituent-witness screening should raise five-code fitness above the 0.300 cell's measured 23.0.")
GROUP = ("Keep the custom metacyclic construction rule and trivial action. Resize only r from 23 to 19: the new size factor 29 makes q=23 invalid at n=667, while q=19 gives n=551. This Z19 shape bed already measured fitness 23.0 with d30s=8..9, motivating a new filling experiment.")
SHAPES = ("Target cell 0.300: equal (2,5) sides give n=551, kb=171 and guaranteed rate 9/29=0.310345. Both constituent matrices have 38 checks and 95 coordinates. Parent limiting X witnesses occupied sector-1 Acol/Bcol blocks across all B columns; replace unequal constituent lengths and screen both sides symmetrically. Exact rank admits only 171<=k<=179.")
FILLING = ("Twenty canonical unordered support-pair recipes use row totals <=8, column totals 3..4, at least three shared columns per side, and full-rank augmentation matrices. Both CSS row-weight sums are <=12. Search 48 labelings per recipe using verified lifts, exact rank, constituent exclusion through weight six and four-cycle counts; then progressively rescreen finalists using randomized kernel bases and pair/triple combinations. Witness readings are heuristic upper bounds, not quantum-distance certificates.")
FAMILY_NAME = "r030_m1r19s1_25x25_canonical_deep_kernel"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 19

def custom_ops(q):
    # Same metacyclic construction rule; cyclic order resized for n=551.
    m, r, s = 1, 19, 1
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 5)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


def _component_columns(rows, width):
    """Column syndromes of a restriction of the verified construction."""
    cols = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return cols


def _kernel_floor(cols):
    """Exact exclusion tests: return 1, 5 or 7 as a classical lower bound.
    Equal syndromes of subsets of size <=3 detect all words of weight <=6.
    This is a constituent test, not a quantum-distance certificate.
    """
    if any(c == 0 for c in cols):
        return 1
    seen = {0: 0}
    for i, c in enumerate(cols):
        mask = 1 << i
        if c in seen and seen[c] != mask:
            return 1
        seen[c] = mask
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            syndrome = cols[i] ^ cols[j]
            mask = (1 << i) | (1 << j)
            if syndrome in seen and seen[syndrome] != mask:
                return 1
            seen[syndrome] = mask
            pairs.append((i, j, syndrome, mask))
    for i, j, syndrome, mask in pairs:
        for k in range(j + 1, len(cols)):
            s = syndrome ^ cols[k]
            m = mask | (1 << k)
            if s in seen and seen[s] != m:
                return 5
            seen[s] = m
    return 7


def _four_cycles(cols):
    total = 0
    for i, c in enumerate(cols):
        for d in cols[:i]:
            overlap = (c & d).bit_count()
            total += overlap * (overlap - 1) // 2
    return total


def _distinct_cell(rng, G, count):
    cell, keys = [], set()
    while len(cell) < count:
        g = G.pick(rng)
        key = G._key(g)
        if key not in keys:
            keys.add(key)
            cell.append(g)
    return cell


def _support_cards():
    """Canonical count matrices, modulo column permutations and row swap."""
    from itertools import combinations_with_replacement
    choices = sorted((a, total - a)
                     for total in (3, 4) for a in range(total + 1))
    profiles = []
    for columns in combinations_with_replacement(choices, 5):
        top = tuple(a for a, b in columns)
        bottom = tuple(b for a, b in columns)
        rows = (sum(top), sum(bottom))
        if min(rows) < 7 or max(rows) > 8:
            continue
        if sum(a > 0 and b > 0 for a, b in columns) < 3:
            continue
        if tuple(sorted((b, a) for a, b in columns)) < columns:
            continue

        # Full rank over F2 of the 2x5 augmentation matrix.
        p0 = sum((a & 1) << j for j, (a, b) in enumerate(columns))
        p1 = sum((b & 1) << j for j, (a, b) in enumerate(columns))
        if not p0 or not p1 or p0 == p1:
            continue
        zeros = sum(a == 0 or b == 0 for a, b in columns)
        profiles.append((columns, (top, bottom), rows, zeros))

    profiles.sort()
    cards = []
    for i, pa in enumerate(profiles):
        for pb in profiles[i:]:
            _, sa, ar, za = pa
            _, sb, br, zb = pb
            ac = tuple(sum(c) for c in zip(*sa))
            bc = tuple(sum(c) for c in zip(*sb))
            if max(ar) + max(bc) > 12:
                continue
            if max(br) + max(ac) > 12:
                continue
            priority = (
                min(ar + br),
                -abs(sum(ar) - sum(br)),
                -za - zb,
                sum(ar) + sum(br),
                -max(max(row) for grid in (sa, sb) for row in grid),
            )
            cards.append((priority, sa, sb))
    cards.sort(reverse=True)
    if len(cards) < 20:
        raise ValueError("fewer than twenty canonical support pairs")
    return tuple((sa, sb) for _, sa, sb in cards[:20])


def _kernel_probe(cols, rng, trials, reading=None):
    """Find actual classical kernel words; absence is not certification."""
    width = len(cols)
    best = width + 1 if reading is None else reading
    order = list(range(width))
    for _ in range(trials):
        rng.shuffle(order)
        pivots = {}
        words = []
        for c in order:
            syndrome, word = cols[c], 1 << c
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
                best = min(best, word.bit_count())

        words.sort(key=int.bit_count)
        for i, word in enumerate(words):
            for j in range(i):
                best = min(best, (word ^ words[j]).bit_count())

        # Low-weight basis triples reduce dependence on one information set.
        short = words[:12]
        for i, word in enumerate(short):
            for j in range(i):
                pair = word ^ short[j]
                for h in range(j):
                    best = min(best, (pair ^ short[h]).bit_count())
        if best <= 4:
            break
    return best


def NEW_CTX(rng, G, idx):
    """Optimize one canonical support pair, then deepen witness screening."""
    cards = _support_cards()
    design = idx % len(cards)
    sa, sb = cards[design]
    assert _grid_weights(sa, sb) <= 12
    assert _no_dead_lines(sa, sb)

    q = G.q
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    chunk = (1 << q) - 1
    pool = []

    for attempt in range(48):
        A = [[_distinct_cell(rng, G, count) for count in row]
             for row in sa]
        B = [[_distinct_cell(rng, G, count) for count in row]
             for row in sb]
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (171 <= k <= 179):
            continue

        # Component actions extracted solely from the verified LP lift.
        ar = []
        for i in range(ra):
            for r in range(q):
                source = hx[(i * cb) * q + r]
                compact = 0
                for j in range(ca):
                    compact |= ((source >> (j * cb * q))
                                & chunk) << (j * q)
                ar.append(compact)
        bmask = (1 << (cb * q)) - 1
        br = [hz[alpha * q + r] & bmask
              for alpha in range(rb) for r in range(q)]
        ac = _component_columns(ar, ca * q)
        bc = _component_columns(br, cb * q)
        fa, fb = _kernel_floor(ac), _kernel_floor(bc)
        cycles = _four_cycles(ac) + _four_cycles(bc)
        coarse = (min(fa, fb), fa + fb, -cycles, -abs(k - 171))
        candidate = {
            "A": A, "B": B, "hx": hx, "hz": hz,
            "ac": ac, "bc": bc, "k": k, "n": n,
            "cycles": cycles, "coarse": coarse,
        }
        pool.append(candidate)
        pool.sort(key=lambda c: c["coarse"], reverse=True)
        del pool[12:]

    if not pool:
        raise ValueError("no candidate passed the exact target-rate screen")

    def improve(candidate, trials):
        candidate["da"] = _kernel_probe(
            candidate["ac"], rng, trials, candidate.get("da"))
        candidate["db"] = _kernel_probe(
            candidate["bc"], rng, trials, candidate.get("db"))

    def score(candidate):
        da, db = candidate["da"], candidate["db"]
        d = min(da, db)
        return (candidate["k"] * d * d / candidate["n"],
                da + db, -candidate["cycles"],
                -abs(candidate["k"] - 171))

    for candidate in pool:
        improve(candidate, 12)
    finalists = sorted(pool, key=score, reverse=True)[:4]

    # Equal additional budgets; retain every earlier lighter witness.
    for trials in (40, 88):
        for candidate in finalists:
            improve(candidate, trials)
    winner = max(finalists, key=score)
    assert css_ok(winner["hx"], winner["hz"])
    return {
        "idx": idx, "design": design,
        "cells": {"A": winner["A"], "B": winner["B"]},
        "screen": (winner["da"], winner["db"]),
        "k": winner["k"],
    }


def CELL(rng, G, side, i, j, ctx):
    """Return the selected full cell; no further randomization."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels do not exclude weight-6 sector-1 Z logicals: four of the parent's five measured codes had d30s=6.
# REFUTED: The parent's constituent-floor/four-cycle ranking pushes every top-five reading above 9; its measured readings were 9,9,10,10,9.
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
