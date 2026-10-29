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
CONCEPT = ("For S3 x C4 of order 24 on shapes (2,3)x(2,5), test whether screening legal sparse fillings through logical weight four raises five-code fitness above 13.9040 in rate cell 0.150.")
GROUP = ("Replace the parent's cyclic order-31 group by the genuinely different nonabelian direct product S3 x C4, of order 24. The unchanged shape factor 19 gives n=456. RUN STATE includes modest order-24 beds, including this shape; this tests a different group law rather than assuming order 24 is already advantageous.")
SHAPES = ("Keep (2,3)x(2,5): at q=24 the guaranteed bound is k>=72, or rate 72/456 ~= 0.158, in cell 0.150. The component dimensions are ca-ra=1 and cb-rb=3, so the sides are unequal. Parent weight-9/10 X logicals were spread across Acol0-2/Bcol0-4; retain that incidence layout while changing the lift group.")
FILLING = ("Generate six legal sparse support-and-label candidates per slot, construct the exact lifted matrices, and reject candidates with detected nontrivial logicals through weight four. Prefer candidates with no such witnesses, then those in the target rate band, with higher k and fewer Tanner 4-cycles.")
FAMILY_NAME = "r015_s3xc4_A23_B25_sparse_w4screen"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 24

def custom_ops(q):
    # S3 x C4. Represent S3 elements by (x,y), with y acting
    # on C3 by inversion; encode the product as (a,x,y).
    els = [(a, x, y) for a in range(4) for y in range(2)
           for x in range(3)]

    def red(g):
        return (g[0] % 4, g[1] % 3, g[2] % 2)

    def idx(g):
        a, x, y = red(g)
        return (a * 2 + y) * 3 + x

    def mul(g, h):
        a, x, y = red(g)
        b, u, v = red(h)
        return ((a + b) % 4,
                (x + (u if y == 0 else -u)) % 3,
                (y + v) % 2)

    def inv(g):
        a, x, y = red(g)
        return ((-a) % 4,
                (-x if y == 0 else x) % 3,
                y)

    return dict(elements=els, ident=(0, 0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Search sparse, row-weight-legal fillings using the exact LP lift."""
    def has_short_logical(checks, stabilizers, n):
        # A column signature is the syndrome of the corresponding qubit.
        sig = [0] * n
        for r, row in enumerate(checks):
            while row:
                bit = row & -row
                sig[bit.bit_length() - 1] |= 1 << r
                row ^= bit

        # Membership in the stabilizer row space distinguishes logicals
        # from low-weight stabilizers.
        basis = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                if p in basis:
                    row ^= basis[p]
                else:
                    basis[p] = row
                    break

        def outside_span(v):
            while v:
                p = v.bit_length() - 1
                if p not in basis:
                    return True
                v ^= basis[p]
            return False

        for a, s in enumerate(sig):
            if s == 0 and outside_span(1 << a):
                return True
        for a in range(n):
            for b in range(a + 1, n):
                if sig[a] == sig[b] and outside_span((1 << a) | (1 << b)):
                    return True

        by_sig = {}
        for c, s in enumerate(sig):
            by_sig.setdefault(s, []).append(c)
        for a in range(n):
            for b in range(a + 1, n):
                for c in by_sig.get(sig[a] ^ sig[b], ()):
                    if c > b and outside_span((1 << a) | (1 << b) | (1 << c)):
                        return True

        # For weight four, group pairs by syndrome and stabilizer-space
        # remainder. Different remainders certify a non-stabilizer sum.
        def residue(v):
            rem = 0
            while v:
                p = v.bit_length() - 1
                bit = 1 << p
                if p in basis:
                    v ^= basis[p]
                else:
                    rem ^= bit
                    v ^= bit
            return rem

        residues = [residue(1 << i) for i in range(n)]
        pairs = {}
        for i in range(n):
            for j in range(i + 1, n):
                s = sig[i] ^ sig[j]
                rem = residues[i] ^ residues[j]
                pairs.setdefault(s, {}).setdefault(rem, []).append((i, j))

        for by_rem in pairs.values():
            if len(by_rem) < 2:
                continue
            total = sum(len(edges) for edges in by_rem.values())
            all_degree = {}
            rem_degree = {}
            for rem, edges in by_rem.items():
                degree = {}
                for u, v in edges:
                    degree[u] = degree.get(u, 0) + 1
                    degree[v] = degree.get(v, 0) + 1
                    all_degree[u] = all_degree.get(u, 0) + 1
                    all_degree[v] = all_degree.get(v, 0) + 1
                rem_degree[rem] = degree
            for rem, edges in by_rem.items():
                other_total = total - len(edges)
                degree = rem_degree[rem]
                for u, v in edges:
                    incident = (all_degree.get(u, 0) - degree.get(u, 0)
                                + all_degree.get(v, 0) - degree.get(v, 0))
                    if other_total > incident:
                        return True
        return False

    def four_cycles(rows):
        # Count pairs of checks sharing two or more qubits.
        overlaps = {}
        for row in rows:
            cols = []
            while row:
                bit = row & -row
                cols.append(bit.bit_length() - 1)
                row ^= bit
            for x in range(len(cols)):
                for y in range(x + 1, len(cols)):
                    p = (cols[x], cols[y])
                    overlaps[p] = overlaps.get(p, 0) + 1
        return sum(v * (v - 1) // 2 for v in overlaps.values())

    best = None
    for _ in range(6):
        # One or two terms per cell: the resulting supports are sparse,
        # and the scaffold's exact row-weight check remains authoritative.
        sa = [[rng.randint(1, 2) for _ in range(A_SHAPE[1])]
              for _ in range(A_SHAPE[0])]
        sb = [[rng.randint(1, 2) for _ in range(B_SHAPE[1])]
              for _ in range(B_SHAPE[0])]
        if _grid_weights(sa, sb) > W_MAX or not _no_dead_lines(sa, sb):
            continue

        cells = {}
        for side, supports in (("A", sa), ("B", sb)):
            cells[side] = [
                [rng.sample(G.elements, count) for count in row]
                for row in supports
            ]

        hx, hz, n = lifted_matrices(G, cells["A"], cells["B"])
        if not css_ok(hx, hz):
            continue
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k <= 0:
            continue
        low = (has_short_logical(hz, hx, n)
               or has_short_logical(hx, hz, n))
        rate = k / n
        in_band = 0.125 <= rate < 0.175
        cycles = four_cycles(hx) + four_cycles(hz)
        key = (low, not in_band, -k, cycles)
        if best is None or key < best[0]:
            best = (key, cells, hx, hz)

    if best is None:
        raise ValueError("no legal positive-k sparse candidate found")
    if not css_ok(best[2], best[3]):
        raise ValueError("selected candidate failed the exact CSS check")
    return {"idx": idx, "cells": best[1]}


def CELL(rng, G, side, i, j, ctx):
    """Return the selected full cell for this slot."""
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
