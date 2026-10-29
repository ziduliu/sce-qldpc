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
CONCEPT = ("On metacyclic Z14:Z2 with s=13, q=28, shapes (1,3)x(2,5), balanced supports and non-involutory binomials should improve the rate-0.350 cell's five-code distance readings beyond 5; this is a measured-search hypothesis, not a distance guarantee.")
GROUP = ("Keep the parent's group and order unchanged: its measured distance readings of 8-10 motivate a higher-rate shape experiment, without assuming those readings transfer to the new bed.")
SHAPES = ("A=(1,3), B=(2,5): n=476, guaranteed k>=168 and rate>=6/17=0.35294. Component lengths/check counts are 84/28 and 140/56. Parent lightest witnesses occupied sector-1 Acol/Bcol blocks broadly, so screen both components. Verified lifted ranks restrict every emitted candidate to the 0.350 band.")
FILLING = ("Indexed dense support profiles balance B-row totals and explicitly obey both weight caps. Same-coset binomials have relative rotation order 14 rather than an involution. Each slot optimizes independent labels using exact component-kernel screening through weight 5, randomized dependency witnesses, and four-cycle counts. Component screening is not a quantum-distance certificate.")
FAMILY_NAME = "r035_m14r2s13_13x25_balanced_binorder_screen"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 28

def custom_ops(q):
    # metacyclic Z_14 x| Z_2: y x y^-1 = x^13 (s^r = 1 mod m);
    # rolled, not designed - see make_rate_seeds.py
    m, r, s = 14, 2, 13
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


def _support_design(idx):
    """Enumerate support designs, not permutation copies of a count grid."""
    from itertools import combinations_with_replacement
    a = ((2, 3, 3), (3, 3, 3), (2, 2, 4), (2, 3, 4))[idx % 4]
    col_cap = 12 - sum(a)
    row_cap = 12 - max(a)
    pairs = [(x, y) for x in range(1, 4) for y in range(1, 4)
             if x + y <= col_cap]
    ceiling = min(5 * col_cap, 2 * row_cap)
    designs = []
    for p in combinations_with_replacement(pairs, 5):
        swapped = tuple(sorted((y, x) for x, y in p))
        if p > swapped:
            continue
        totals = [sum(z[r] for z in p) for r in range(2)]
        if max(totals) > row_cap or abs(totals[0] - totals[1]) > 1:
            continue
        if sum(totals) < ceiling - 1:
            continue
        sb = [[z[r] for z in p] for r in range(2)]
        if _grid_weights([list(a)], sb) <= 12:
            designs.append(p)
    designs.sort(key=lambda p: (
        -sum(x + y for x, y in p),
        -min(x + y for x, y in p), p))
    if not designs:
        raise ValueError("empty support-design catalogue")
    number = idx // 4
    p = designs[number % len(designs)]
    return [list(a)], [[z[r] for z in p] for r in range(2)], number // len(designs)


def _designed_cell(rng, G, weight, phase):
    from math import gcd
    m = G.q // 2
    pools = [[g for g in G.elements if g[1] == b] for b in range(2)]
    if weight == 1:
        return [rng.choice(pools[phase % 2])]
    if weight == 2:
        # Opposite-coset binomials have involutory relative labels.
        b = phase % 2
        x = rng.choice(pools[b])
        choices = [y for y in pools[b] if gcd((y[0] - x[0]) % m, m) == 1]
        return [x, rng.choice(choices)]
    # Higher-weight cells mix the two cosets, without repeated elements.
    n0 = weight // 2
    if weight % 2 and phase % 2:
        n0 += 1
    return rng.sample(pools[0], n0) + rng.sample(pools[1], weight - n0)


def _syndrome_columns(rows, n):
    cols = [0] * n
    for r, row in enumerate(rows):
        check = 1 << r
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= check
            row ^= bit
    return cols


def _kernel_floor(cols):
    """Exact smallest kernel weight through 5; return 6 if none exists."""
    singles = {}
    for i, s in enumerate(cols):
        if s == 0:
            return 1
        if s in singles:
            return 2
        singles[s] = i
    pairs = {}
    has_four = False
    for i in range(len(cols)):
        for j in range(i):
            s = cols[i] ^ cols[j]
            if s in singles:
                return 3
            if s in pairs:
                has_four = True
            else:
                pairs[s] = (i, j)
    if has_four:
        return 4
    for i in range(len(cols)):
        for j in range(i):
            s = cols[i] ^ cols[j]
            for h in range(j):
                if (s ^ cols[h]) in pairs:
                    # With weights <=4 excluded, the matching pair is disjoint.
                    return 5
    return 6


def _dependency_reading(rng, cols, trials=12):
    """Upper-bound witnesses for a component kernel, not quantum distance."""
    best = len(cols) + 1
    order = list(range(len(cols)))
    for _ in range(trials):
        rng.shuffle(order)
        pivots = {}
        for c in order:
            syndrome, support = cols[c], 1 << c
            while syndrome:
                p = syndrome.bit_length() - 1
                if p not in pivots:
                    pivots[p] = (syndrome, support)
                    break
                s, v = pivots[p]
                syndrome ^= s
                support ^= v
            if not syndrome:
                best = min(best, support.bit_count())
    return best


def _four_cycle_cost(cols):
    cost = 0
    for i in range(len(cols)):
        for j in range(i):
            overlap = (cols[i] & cols[j]).bit_count()
            cost += overlap * (overlap - 1) // 2
    return cost


def NEW_CTX(rng, G, idx):
    """Optimize independent labels for an indexed support design."""
    sa, sb, mode = _support_design(idx)
    best, best_score, best_matrices = None, None, None
    q = G.q
    for attempt in range(24 + 4 * (idx % 3)):
        cells = {}
        for side, sup in (("A", sa), ("B", sb)):
            cells[side] = [
                [_designed_cell(rng, G, count,
                                mode + i + j + (side == "B"))
                 for j, count in enumerate(row)]
                for i, row in enumerate(sup)]
        hx, hz, n = lifted_matrices(G, cells["A"], cells["B"])
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (325 * n <= 1000 * k < 375 * n):
            continue

        # Extract both component matrices from the verified LP construction.
        # HX beta=0 contains the A left lift; HZ j=0 contains the B right lift.
        ac_full = _syndrome_columns(hx[:q], n)
        ac = [ac_full[(j * B_SHAPE[1]) * q + t]
              for j in range(A_SHAPE[1]) for t in range(q)]
        bc = _syndrome_columns(hz[:B_SHAPE[0] * q], n)[:B_SHAPE[1] * q]
        fa, fb = _kernel_floor(ac), _kernel_floor(bc)
        floor = min(fa, fb)
        if best_score is not None and floor < best_score[0]:
            continue
        da = fa if fa < 6 else _dependency_reading(rng, ac)
        db = fb if fb < 6 else _dependency_reading(rng, bc)
        cycles = _four_cycle_cost(ac) + _four_cycle_cost(bc)
        score = (floor, min(da, db), da + db, -cycles, -abs(k - 168))
        if best_score is None or score > best_score:
            best, best_score = cells, score
            best_matrices = (hx, hz)
    if best is None:
        raise ValueError("no candidate in target rate band")
    if not css_ok(*best_matrices):
        raise ValueError("verified construction failed CSS check")
    return {"idx": idx, "cells": best}


def CELL(rng, G, side, i, j, ctx):
    """All new-shape cells are constructed and screened in NEW_CTX."""
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
