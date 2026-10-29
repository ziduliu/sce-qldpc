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
CONCEPT = ("Dicyclic Dic_28, order 112, shapes (2,3)x(2,3): component-column weights 4-5, long-order binomials and screened component cycles should suppress small sector-1 logicals and populate the empty 0.100 rate cell.")
GROUP = ("Dic_28 is an untried nonabelian group, not Z37. Its rotation subgroup has order 56; same-coset binomials can have relative order 56, unlike cross-coset relative elements of order 4. The parent's five measured codes all had provisional distance reading 6.")
SHAPES = ("Equal (2,3) components have length 336 and 224 checks. At q=112, n=1456 and k>=112, giving guaranteed rate 1/13. Parent weight-6 witnesses occupied sector 1 broadly, with X witnesses prominent in Acol2/Acol3; every new component column instead has weight at least 4.")
FILLING = ("Slots use distinct canonical support-profile pairs with row sums 6-7 and column sums 4-5, so both CSS row-weight bounds are at most 12. Cells occupy one rotation coset, with long-order binomials and Sidon higher-weight supports; mixed coset cycle parities retain nonabelian labels. Verified lifts screen the rate and exclude component words of weight 2 or 4, then select balanced low-four-cycle components from multiple labelings. Distance improvement is a hypothesis, not a certificate.")
HOW_AB = FILLING
FAMILY_NAME = "r010_dic28_23x23_longcycle_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 112

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Dic_28 requires order 112")
    # x^56=1, y^2=x^28, y*x*y^-1=x^-1.
    # Canonical elements are x^a*y^b, 0<=a<56, 0<=b<2.
    els = [(a, b) for b in range(2) for a in range(56)]

    def red(g):
        a, b = int(g[0]), int(g[1])
        return ((a + 28 * (b // 2)) % 56, b % 2)

    def mul(g, h):
        g, h = red(g), red(h)
        a = g[0] + (h[0] if g[1] == 0 else -h[0])
        return ((a + 28 * g[1] * h[1]) % 56,
                (g[1] + h[1]) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % 56, 0) if b == 0 else ((a + 28) % 56, 1)

    def idx(g):
        a, b = red(g)
        return a + 56 * b

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 3)               # (rb, cb)


def _support_design(idx):
    """Canonical weighted protographs, not row/column orbit copies."""
    from itertools import product, permutations
    profiles = set()
    for values in product(range(1, 5), repeat=6):
        s = [values[:3], values[3:]]
        if not all(6 <= sum(row) <= 7 for row in s):
            continue
        if not all(4 <= s[0][j] + s[1][j] <= 5 for j in range(3)):
            continue
        key = min(
            tuple(s[r][c] for r in rr for c in pp)
            for rr in ((0, 1), (1, 0))
            for pp in permutations(range(3)))
        profiles.add(key)

    profiles = sorted(profiles)
    pairs = [(a, b) for ia, a in enumerate(profiles)
             for b in profiles[ia:]]
    # Prefer balanced cell sizes, but assign different profile pairs
    # to the first twenty slots instead of twenty draws of one profile.
    pairs.sort(key=lambda ab: (
        sum(v == 1 for s in ab for v in s),
        sum((v - 2) ** 2 for s in ab for v in s),
        sum(v for s in ab for v in s), ab))
    a, b = pairs[idx % len(pairs)]
    return ([list(a[:3]), list(a[3:])],
            [list(b[:3]), list(b[3:])])


def _rotation_support(rng, weight):
    """Distinct rotations; avoid short relative orders and 4-cycles."""
    from math import gcd
    for _ in range(2000):
        a = sorted(rng.sample(range(56), weight))
        if weight == 1:
            return a
        differences = [(x - y) % 56 for x in a for y in a if x != y]
        if weight == 2:
            if gcd(differences[0], 56) == 1:
                return a
        elif (len(set(differences)) == len(differences)
              and all(56 // gcd(d, 56) >= 14 for d in differences)):
            return a
    raise ValueError("could not construct long-order rotation support")


def _label_design(rng, supports, idx, offset):
    cells = []
    for i, row in enumerate(supports):
        out = []
        for j, weight in enumerate(row):
            # Each cell occupies one coset. The lower row has mixed
            # cosets, giving odd quotient parity on some base cycles.
            b = int(i == 1 and j != (idx + offset) % 3)
            out.append([(a, b) for a in _rotation_support(rng, weight)])
        cells.append(out)
    return cells


def _component_rows(hx, hz, q):
    """Extract components from the VERIFIED lift, not a separate lift."""
    mask = (1 << q) - 1
    arows = []
    # HX order is (Arow, Bcol, group element). Select Bcol=0,
    # then compact its three sector-1 blocks.
    for i in range(2):
        for r in range(q):
            source = hx[(i * 3) * q + r]
            row = 0
            for j in range(3):
                row |= ((source >> (j * 3 * q)) & mask) << (j * q)
            arows.append(row)
    # HZ order is (Acol, Brow, group element). Acol=0 already
    # places its three B-column blocks consecutively.
    bmask = (1 << (3 * q)) - 1
    brows = [row & bmask for row in hz[:2 * q]]
    return arows, brows


def _component_quality(rows, ncols):
    """Exclude weight-2/4 kernel words; count actual Tanner 4-cycles."""
    columns = [0] * ncols
    pair_uses = {}
    cycles = 0
    for r, row in enumerate(rows):
        support = []
        bits = row
        while bits:
            bit = bits & -bits
            c = bit.bit_length() - 1
            columns[c] |= 1 << r
            support.append(c)
            bits ^= bit
        for ia, a in enumerate(support):
            for b in support[:ia]:
                key = b * ncols + a
                previous = pair_uses.get(key, 0)
                cycles += previous
                pair_uses[key] = previous + 1

    if 0 in columns or len(set(columns)) != ncols:
        return None

    # With distinct nonzero columns, repeated pair syndromes have
    # disjoint endpoints and certify a weight-4 component codeword.
    pair_syndromes = set()
    for a, syndrome in enumerate(columns):
        for b in range(a):
            value = syndrome ^ columns[b]
            if value in pair_syndromes:
                return None
            pair_syndromes.add(value)
    return cycles


def NEW_CTX(rng, G, idx):
    """Optimize several labelings of this slot's deliberate profile."""
    sa, sb = _support_design(idx)
    if _grid_weights(sa, sb) > W_MAX or not _no_dead_lines(sa, sb):
        raise ValueError("illegal designed support profile")

    best = None
    for trial in range(48):
        if trial >= 12 and best is not None:
            break
        A = _label_design(rng, sa, idx, 0)
        B = _label_design(rng, sb, idx, 1)
        hx, hz, n = lifted_matrices(G, A, B)
        ar, br = _component_rows(hx, hz, G.q)
        qa = _component_quality(ar, 3 * G.q)
        if qa is None:
            continue
        qb = _component_quality(br, 3 * G.q)
        if qb is None:
            continue

        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k <= 0 or 8 * k >= n:
            continue
        # Optimize the weaker component first. Rank excess is only
        # a tie-breaker, not a substitute for distance screening.
        score = (max(qa, qb), qa + qb, k - G.q)
        if best is None or score < best[0]:
            best = (score, A, B, k)

    if best is None:
        raise ValueError("no labeling passed component and rate screens")
    return {"idx": idx, "cells": {"A": best[1], "B": best[2]},
            "k": best[3], "component_score": best[0]}


def CELL(rng, G, side, i, j, ctx):
    """Return the complete screened cell without additional randomness."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
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
