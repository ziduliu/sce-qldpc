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
CONCEPT = ("Falsifiable: for the order-21 group C7 semidirect C3 on A=(2,4), B=(2,6), balanced support profiles with distinct C3-coset coordinates in each multi-term cell will raise fitness above 11.6652; the measured five-code fitness tests this principle.")
GROUP = ("Keep the order-21 nonabelian group: the current program's five measured codes average 11.6652, with four at d30s=6 and one at d30s=7. Keeping the group isolates the filling change.")
SHAPES = ("Keep (2,4)x(2,6): the guaranteed rate is 2/7, q=21 gives n=588, and the component profiles are nominally [4,2] and [6,4], with unequal sector lengths 504 and 84. The parent's weight-6 X logicals were in sector 1, especially in Acol1/Bcol0, Acol1/Bcol2, and Acol1/Bcol5; the filling changes support balance without changing that layout.")
FILLING = ("Search eight balanced profiles of 28, 30, or 32 terms, keeping row weights legal. Within each multi-term cell, select elements from distinct C3-cosets. Screen with the verified lifts and logical-floor checks, then prefer the in-band candidate with the largest screened k*d^2/n; this targets the sector-1 weight-6 witnesses while retaining rank surplus.")
FAMILY_NAME = "r030_F21_24x26_cosetspread_balanced_proxy"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 21

def custom_ops(q):
    # Nonabelian Frobenius group C7 semidirect C3, with action x -> x^2.
    # Elements are x^a y^b; y^3=x^7=e and y*x*y^-1=x^2.
    els = [(a, b) for b in range(3) for a in range(7)]

    def mul(g, h):
        return ((g[0] + pow(2, g[1], 7) * h[0]) % 7,
                (g[1] + h[1]) % 3)

    def inv(g):
        bi = (-g[1]) % 3
        return ((-pow(2, bi, 7) * g[0]) % 7, bi)

    def idx(g):
        return 7 * (g[1] % 3) + (g[0] % 7)

    def red(g):
        return (g[0] % 7, g[1] % 3)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 4)               # (ra, ca)
B_SHAPE = (2, 6)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Search varied legal supports and retain a high-rank screened filling."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE

    def supports(mode):
        # These templates balance row and column degrees while varying the
        # distribution of the same small set of legal term counts.
        sa = [[2] * ca for _ in range(ra)]
        sb = [[1] * cb for _ in range(rb)]
        if mode == 0:
            # Two doubles per B row, in four distinct columns.
            places = rng.sample(range(cb), 2 * rb)
            for a in range(rb):
                for j in places[2*a:2*a+2]:
                    sb[a][j] = 2
        elif mode == 1:
            # One A increment and one B double per row, all positions distinct.
            pa = rng.sample(range(ca), ra)
            pb = rng.sample(range(cb), rb)
            for i in range(ra):
                sa[i][pa[i]] = 3
            for a in range(rb):
                sb[a][pb[a]] = 2
        elif mode == 2:
            # One triple cell per B row, in distinct columns.
            places = rng.sample(range(cb), rb)
            for a in range(rb):
                sb[a][places[a]] = 3
        elif mode == 3:
            # A remains doubled; one B double per row.
            places = rng.sample(range(cb), rb)
            for a in range(rb):
                sb[a][places[a]] = 2
        elif mode == 4:
            # A gets one increment per row; B stays monomial.
            places = rng.sample(range(ca), ra)
            for i in range(ra):
                sa[i][places[i]] = 3
        # Mode 5 is the 28-term all-A-doubles, all-B-singletons profile.
        return sa, sb

    cosets = [[g for g in G.elements if g[1] == b] for b in range(3)]

    def draw_cells(supports):
        cells = []
        for row in supports:
            out = []
            for count in row:
                if count <= 1:
                    out.append(rng.sample(G.elements, count))
                else:
                    bs = rng.sample(range(3), count)
                    out.append([rng.choice(cosets[b]) for b in bs])
            cells.append(out)
        return cells

    def basis(rows):
        piv = {}
        for row in rows:
            x = row
            while x:
                p = x.bit_length() - 1
                if p in piv:
                    x ^= piv[p]
                else:
                    piv[p] = x
                    break
        return piv

    def in_span(v, piv):
        while v:
            p = v.bit_length() - 1
            if p not in piv:
                return False
            v ^= piv[p]
        return True

    def logical_floor(checks, stabs, n):
        """Return 1..4 for a found logical; 5 if none is found through weight 4."""
        piv = basis(stabs)
        synd = [0] * n
        for r, row in enumerate(checks):
            bit = 1 << r
            while row:
                low = row & -row
                synd[low.bit_length() - 1] |= bit
                row ^= low

        for i, s in enumerate(synd):
            if not s and not in_span(1 << i, piv):
                return 1

        by_synd = {}
        for i, s in enumerate(synd):
            for j in by_synd.get(s, ()):
                if not in_span((1 << i) | (1 << j), piv):
                    return 2
            by_synd.setdefault(s, []).append(i)

        for i in range(n):
            for j in range(i + 1, n):
                for k in by_synd.get(synd[i] ^ synd[j], ()):
                    if k > j and not in_span((1 << i) | (1 << j) | (1 << k), piv):
                        return 3

        pairs = {}
        comparisons = 0
        for i in range(n):
            for j in range(i + 1, n):
                sig = synd[i] ^ synd[j]
                for a, b in pairs.get(sig, ()):
                    if a != i and a != j and b != i and b != j:
                        comparisons += 1
                        if comparisons > 50000:
                            return 4
                        if not in_span((1 << a) | (1 << b) | (1 << i) | (1 << j), piv):
                            return 4
                pairs.setdefault(sig, []).append((i, j))
        return 5

    def four_cycles(rows):
        seen = {}
        total = 0
        for row in rows:
            cols = []
            while row:
                low = row & -row
                cols.append(low.bit_length() - 1)
                row ^= low
            for u in range(len(cols)):
                for v in range(u + 1, len(cols)):
                    pair = (cols[u], cols[v])
                    total += seen.get(pair, 0)
                    seen[pair] = seen.get(pair, 0) + 1
        return total

    in_band, above_band = [], []
    for trial in range(8):
        sa, sb = supports((idx + trial) % 6)
        if (_grid_weights(sa, sb) > W_MAX or
                not _no_dead_lines(sa, sb)):
            continue
        cells_a = draw_cells(sa)
        cells_b = draw_cells(sb)
        hx, hz, n = lifted_matrices(G, cells_a, cells_b)
        if not css_ok(hx, hz):
            continue
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k <= 0:
            continue
        dx = logical_floor(hz, hx, n)
        dz = logical_floor(hx, hz, n)
        c4 = four_cycles(hx) + four_cycles(hz)
        d = min(dx, dz)
        kcap = (13 * n) // 40
        key = (d, k * d * d / n, dx + dz,
               -max(0, k - kcap), -c4, rng.random())
        item = (key, {"A": cells_a, "B": cells_b})
        (in_band if 40 * k <= 13 * n else above_band).append(item)

    candidates = in_band or above_band
    if not candidates:
        raise ValueError("no positive-k screened filling")
    best = max(candidates, key=lambda item: item[0])
    return {"idx": idx, "cells": best[1]}


def CELL(rng, G, side, i, j, ctx):
    """Return the full screened cell for this side and protograph position."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# Refuted: the cyclic-order-21 parent's rolled filling did not lift the measured tail above d30s=6; all five measured codes had d30s=6.
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
