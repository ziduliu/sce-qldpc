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
CONCEPT = ("Falsifiable: for SL(2,3), order 24, on A=(1,3), B=(2,7), lower-weight fillings selected by exact-lift logical screening will raise the five-code mean measured score above 3.4153 in the 0.450 rate cell.")
GROUP = ("Keep SL(2,3) and its order 24 fixed as required by this filling-only experiment. This tests whether changing supports and the labeling search improves circuit performance without changing the regular group action.")
SHAPES = ("Keep A=(1,3), B=(2,7): n=24*(3*7+1*2)=552 and the guaranteed k/n is 240/552=0.4348. The component lengths are 72 and 168, with 24 and 48 check rows. The measured codes' X-basis results were weakest; slot 36 scored 3.470 in X and had Z-type weight-4 logicals, motivating a search that prioritizes the Z-logical direction.")
FILLING = ("Use 4-5 terms in A and 8-9 terms per B row, with the B-row extra terms placed in disjoint columns. This keeps the maximum protograph row weight at 11 and reduces term count relative to the parent. Screen 32 candidates with the verified lifted matrices, prioritizing minimum logical weight, then Z-logical weight, Tanner 4-cycles, column-degree balance, and gate exposure.")
FAMILY_NAME = "sl2_3_o24_a13_b27_sparse_screen4"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 24

def custom_ops(q):
    # SL(2,3): determinant-one 2x2 matrices over F_3.
    els = [(a, b, c, d)
           for a in range(3) for b in range(3)
           for c in range(3) for d in range(3)
           if (a*d - b*c) % 3 == 1]
    pos = {g: i for i, g in enumerate(els)}

    def red(g):
        return tuple(int(x) % 3 for x in g)

    def mul(g, h):
        a, b, c, d = red(g)
        e, f, u, v = red(h)
        return ((a*e + b*u) % 3, (a*f + b*v) % 3,
                (c*e + d*u) % 3, (c*f + d*v) % 3)

    def inv(g):
        a, b, c, d = red(g)
        return (d, (-b) % 3, (-c) % 3, a)

    def idx(g):
        return pos[red(g)]

    return dict(elements=els, ident=(1, 0, 0, 1), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Search balanced sparse fillings using the verified lifted matrices."""
    def supports():
        # Slot-stratified low-weight profiles: A has 1-2 extras;
        # each B row has 1-2 extras, placed in disjoint columns.
        ea = 1 + (idx % 2)
        eb1 = 1 + ((idx // 2) % 2)
        eb2 = 1 + ((idx // 4) % 2)
        sa = [[1, 1, 1]]
        for j in rng.sample(range(3), ea):
            sa[0][j] += 1
        positions = rng.sample(range(7), eb1 + eb2)
        e1 = set(positions[:eb1])
        e2 = set(positions[eb1:])
        sb = [[1 + int(j in e1) for j in range(7)],
              [1 + int(j in e2) for j in range(7)]]
        return sa, sb

    def low_logical(check, stab, n):
        piv = {}
        for row in stab:
            x = row
            while x:
                p = x.bit_length() - 1
                if p in piv:
                    x ^= piv[p]
                else:
                    piv[p] = x
                    break

        def in_span(x):
            while x:
                p = x.bit_length() - 1
                if p not in piv:
                    return False
                x ^= piv[p]
            return True

        cols = [0] * n
        for ri, row in enumerate(check):
            bit = 1 << ri
            while row:
                b = row & -row
                cols[b.bit_length() - 1] |= bit
                row ^= b

        by_col = {}
        for i, syndrome in enumerate(cols):
            by_col.setdefault(syndrome, []).append(i)

        for i in by_col.get(0, ()):
            if not in_span(1 << i):
                return 1
        for ids in by_col.values():
            for u in range(len(ids)):
                for v in range(u + 1, len(ids)):
                    if not in_span((1 << ids[u]) | (1 << ids[v])):
                        return 2

        for i in range(n):
            for j in range(i + 1, n):
                target = cols[i] ^ cols[j]
                for k in by_col.get(target, ()):
                    if k > j and not in_span((1 << i) | (1 << j) | (1 << k)):
                        return 3

        pairs = {}
        for i in range(n):
            for j in range(i + 1, n):
                pairs.setdefault(cols[i] ^ cols[j], []).append((i, j))
        work = 0
        for group in pairs.values():
            for u in range(len(group)):
                i, j = group[u]
                for v in range(u + 1, len(group)):
                    work += 1
                    if work > 30000:
                        return 4
                    k, ell = group[v]
                    if i == k or i == ell or j == k or j == ell:
                        continue
                    x = (1 << i) | (1 << j) | (1 << k) | (1 << ell)
                    if not in_span(x):
                        return 4
        return 5

    def four_cycles(rows):
        pairs = {}
        for row in rows:
            cols = []
            while row:
                bit = row & -row
                cols.append(bit.bit_length() - 1)
                row ^= bit
            for u in range(len(cols)):
                for v in range(u + 1, len(cols)):
                    pair = (cols[u], cols[v])
                    pairs[pair] = pairs.get(pair, 0) + 1
        return sum(x * (x - 1) // 2 for x in pairs.values())

    def balance(sa, sb):
        score = 0
        for grid in (sa, sb):
            cols = [sum(row[j] for row in grid)
                    for j in range(len(grid[0]))]
            total = sum(cols)
            score += sum((len(cols) * x - total) ** 2 for x in cols)
        return score

    best = None
    for _ in range(10):
        sa, sb = supports()
        A = [[rng.sample(G.elements, sa[i][j])
              for j in range(A_SHAPE[1])] for i in range(A_SHAPE[0])]
        B = [[rng.sample(G.elements, sb[i][j])
              for j in range(B_SHAPE[1])] for i in range(B_SHAPE[0])]
        cells = {"A": A, "B": B}
        if fallback is None:
            fallback = cells
        hx, hz, n = lifted_matrices(G, A, B)
        if not css_ok(hx, hz):
            continue
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k <= 0:
            continue
        dx = low_logical(hz, hx, n)
        dz = low_logical(hx, hz, n)
        cost = 9 * sum(map(sum, sa)) + 4 * sum(map(sum, sb))
        key = (min(dx, dz), dx + dz, -abs(k - 270), -cost)
        if best is None or key > best[0]:
            best = (key, cells)

    return {"idx": idx, "cells": best[1] if best is not None else fallback}


def CELL(rng, G, side, i, j, ctx):
    """Return the preselected full cell of distinct group elements."""
    return ctx["cells"][side][i][j]

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Refuted: the parent's rolled Z23 (2,4)x(3,5) bed was not a strong 0.150-cell pL recipe: fitness 4.108, with weight-6 Z-type logicals and broadly distributed X-basis failures in its best measured slot.
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
