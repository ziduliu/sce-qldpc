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
CONCEPT = ("Falsifiable: on the nonabelian group C5 semidirect C4 of order 20, with yxy^-1=x^2, and shapes (2,4)x(2,6), the parent's screened filling will produce a stronger measured code in rate cell 0.300.")
GROUP = ("Use C5 semidirect C4, where multiplication by 2 is an order-4 automorphism modulo 5. This is genuinely nonabelian, unlike the parent's cyclic group of order 21. Order 20 gives n=(24+4)*20=560. A different order-20 nonabelian group on these shapes has a recorded bed score of 10.6; this distinct multiplication rule tests a new lift.")
SHAPES = ("Keep (2,4)x(2,6): the rank-count bound is k>=20*(4-2)*(6-2)=160, or k/n>=2/7~=0.286, and q=20 gives n=560. The protograph sides are 2x4 and 2x6, giving 24 sector-1 block positions versus 4 sector-2 positions. The parent's weight-6 X logicals were in sector 1 and spread across Acol/Bcol blocks, especially Acol2 with several B columns.")
FILLING = ("Keep the parent's five legal support-and-label proposals per slot and its verified CSS, rank, weight-1-to-4 logical, and 4-cycle screening. The parent's five measured codes all had d30s=6; changing the group law tests whether the new lift alters those low-weight witnesses while retaining the same filling-selection rule.")
FAMILY_NAME = "r030_F20_C5xC4_action2_24x26_balanced_l4screen"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 20

def custom_ops(q):
    # Nonabelian C5 semidirect C4: y*x*y^-1 = x^2.
    # Elements are x^a y^b; 2 has order 4 modulo 5.
    els = [(a, b) for b in range(4) for a in range(5)]

    def mul(g, h):
        return ((g[0] + pow(2, g[1], 5) * h[0]) % 5,
                (g[1] + h[1]) % 4)

    def inv(g):
        bi = (-g[1]) % 4
        return ((-pow(2, bi, 5) * g[0]) % 5, bi)

    def idx(g):
        return 5 * (g[1] % 4) + (g[0] % 5)

    def red(g):
        return (g[0] % 5, g[1] % 4)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 4)               # (ra, ca)
B_SHAPE = (2, 6)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Build and screen five balanced support/label candidates per slot."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE

    def logical_floor(checks, stabs, n):
        # Exact search through weight four in ker(checks) / rowspace(stabs).
        basis = {}
        for row in stabs:
            x = row
            while x:
                p = x.bit_length() - 1
                if p in basis:
                    x ^= basis[p]
                else:
                    basis[p] = x
                    break

        def in_span(v):
            while v:
                p = v.bit_length() - 1
                if p not in basis:
                    return False
                v ^= basis[p]
            return True

        synd = [0] * n
        for ri, row in enumerate(checks):
            x = row
            while x:
                bit = x & -x
                synd[bit.bit_length() - 1] |= 1 << ri
                x ^= bit

        for i, s in enumerate(synd):
            if not s and not in_span(1 << i):
                return 1, 1

        by_synd = {}
        for i, s in enumerate(synd):
            by_synd.setdefault(s, []).append(i)
        for ids in by_synd.values():
            for u in range(len(ids)):
                for v in range(u + 1, len(ids)):
                    i, j = ids[u], ids[v]
                    if not in_span((1 << i) | (1 << j)):
                        return 2, 1

        for i in range(n):
            for j in range(i + 1, n):
                for k in by_synd.get(synd[i] ^ synd[j], ()):
                    if k > j and not in_span((1 << i) | (1 << j) | (1 << k)):
                        return 3, 1

        seen = {}
        for i in range(n):
            for j in range(i + 1, n):
                value = synd[i] ^ synd[j]
                for a, b in seen.get(value, ()):
                    if a != i and a != j and b != i and b != j:
                        if not in_span((1 << i) | (1 << j) | (1 << a) | (1 << b)):
                            return 4, 1
                seen.setdefault(value, []).append((i, j))
        return 5, 0

    def four_cycles(rows):
        seen = {}
        total = 0
        for row in rows:
            cols = []
            x = row
            while x:
                bit = x & -x
                cols.append(bit.bit_length() - 1)
                x ^= bit
            for u in range(len(cols)):
                for v in range(u + 1, len(cols)):
                    pair = (cols[u], cols[v])
                    old = seen.get(pair, 0)
                    total += old
                    seen[pair] = old + 1
        return total

    best_key = None
    best_cells = None
    for trial in range(5):
        mode = (idx + trial) % 5
        sa = [[2] * ca for _ in range(ra)]
        sb = [[1] * cb for _ in range(rb)]

        if mode == 0:                 # A rows of weight 8; B rows of weight 8
            places = rng.sample(range(cb), 4)
            for a in range(rb):
                for j in places[2*a:2*a+2]:
                    sb[a][j] = 2
        elif mode == 1:               # balanced row sums 9 and 7
            pa = rng.sample(range(ca), 2)
            pb = rng.sample(range(cb), 2)
            for i in range(ra):
                sa[i][pa[i]] = 3
            for a in range(rb):
                sb[a][pb[a]] = 2
        elif mode == 2:               # balanced A rows of weight 10; monomial B
            pa = rng.sample(range(ca), 2)
            for i in range(ra):
                sa[i][pa[i]] = 4
        elif mode == 3:               # A columns total 5; monomial B
            pa = rng.sample(range(ca), 2)
            other = [j for j in range(ca) if j not in pa]
            for j in pa:
                sa[0][j] = 3
            for j in other:
                sa[1][j] = 3
        else:                         # B columns balanced, row weights 8
            pb = rng.sample(range(cb), 2)
            for a in range(rb):
                sb[a][pb[a]] = 3

        if _grid_weights(sa, sb) > W_MAX:
            continue
        cells_a = [[rng.sample(G.elements, sa[i][j]) for j in range(ca)]
                   for i in range(ra)]
        cells_b = [[rng.sample(G.elements, sb[a][j]) for j in range(cb)]
                   for a in range(rb)]
        hx, hz, n = lifted_matrices(G, cells_a, cells_b)
        if not css_ok(hx, hz):
            continue
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k <= 0:
            continue
        dx, _ = logical_floor(hz, hx, n)
        dz, _ = logical_floor(hx, hz, n)
        key = (min(dx, dz), -four_cycles(hx) - four_cycles(hz), k)
        if best_key is None or key > best_key:
            best_key = key
            best_cells = {"A": cells_a, "B": cells_b}

    if best_cells is None:
        raise ValueError("no screened legal filling")
    return {"idx": idx, "cells": best_cells}


def CELL(rng, G, side, i, j, ctx):
    """Return the screened full cell saved by NEW_CTX."""
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
