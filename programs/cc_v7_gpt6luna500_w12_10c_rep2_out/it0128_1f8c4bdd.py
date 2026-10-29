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
CONCEPT = ("For the dihedral group D27 of order 54 on shapes (1,2)x(3,4), screening eight candidate fillings per slot through weight 4 will improve on the parent's five-code fitness of 0.7732.")
GROUP = ("Keep D27 of order 54, as required by L1. The parent’s measured codes had d30s readings of 2, 2, 3, 3, and 4; this tests a filling change while keeping the group fixed.")
SHAPES = ("Keep (1,2)x(3,4): the rank-count bound is k>=54 and k/n>=1/11 ~= 0.091. Here n=(2*4+1*3)*54=594, and q=54 is the largest legal lift order. The component profiles are A [2,1] and B [4,3], with unequal sector lengths. The parent's lightest sector-1 Z logicals touched all Acol0-1/Bcol0-3 blocks.")
FILLING = ("Include an all-two-term baseline and sample other legal profiles using one to three distinct elements per cell. Screen the exact lifted matrices for non-stabilizer logicals through weight 3, then check weight 4 for the best candidates; select by screened floor and k. This targets the parent's low-weight witnesses.")
FAMILY_NAME = "r010_dihedral27_q54_12x34_elemsearch4"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 54

def custom_ops(q):
    # Dihedral group of order 54: x^27 = y^2 = 1, yxy = x^-1.
    m = 27
    els = [(a, b) for b in range(2) for a in range(m)]
    pos = {g: i for i, g in enumerate(els)}

    def mul(g, h):
        sign = 1 if g[1] == 0 else -1
        return ((g[0] + sign * h[0]) % m, (g[1] + h[1]) % 2)

    def inv(g):
        sign = 1 if g[1] == 0 else -1
        return ((-sign * g[0]) % m, g[1])

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=lambda g: pos[tuple(g)], red=lambda g: tuple(g))
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (3, 4)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Search legal support and labeling candidates using the verified lift."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    pool = []

    def span(v, piv):
        while v:
            p = v.bit_length() - 1
            if p not in piv:
                return False
            v ^= piv[p]
        return True

    def floor3(checks, stabs, n):
        piv = {}
        for row in stabs:
            while row:
                p = row.bit_length() - 1
                if p in piv:
                    row ^= piv[p]
                else:
                    piv[p] = row
                    break

        cols = [0] * n
        for r, row in enumerate(checks):
            bit = 1 << r
            while row:
                low = row & -row
                cols[low.bit_length() - 1] |= bit
                row ^= low

        by = {}
        for i, sig in enumerate(cols):
            by.setdefault(sig, []).append(i)

        for i, sig in enumerate(cols):
            if sig == 0 and not span(1 << i, piv):
                return 1, (cols, piv, by)

        for ids in by.values():
            for a in range(len(ids)):
                i = ids[a]
                for b in range(a + 1, len(ids)):
                    j = ids[b]
                    if not span((1 << i) | (1 << j), piv):
                        return 2, (cols, piv, by)

        for i in range(n):
            for j in range(i + 1, n):
                for k in by.get(cols[i] ^ cols[j], ()):
                    if k > j:
                        v = (1 << i) | (1 << j) | (1 << k)
                        if not span(v, piv):
                            return 3, (cols, piv, by)
        return 4, (cols, piv, by)

    def has_weight_four(data):
        cols, piv, _ = data
        pairs = {}
        for i in range(len(cols)):
            bi = 1 << i
            for j in range(i + 1, len(cols)):
                sig = cols[i] ^ cols[j]
                old = pairs.get(sig)
                if old is None:
                    pairs[sig] = [(i, j)]
                    continue
                for a, b in old:
                    if a != i and a != j and b != i and b != j:
                        v = bi | (1 << j) | (1 << a) | (1 << b)
                        if not span(v, piv):
                            return True
                old.append((i, j))
        return False

    def supports():
        for _ in range(200):
            sa = [rng.choices((1, 2, 3), weights=(2, 6, 2), k=ca)
                  for _ in range(ra)]
            sb = [rng.choices((1, 2, 3), weights=(2, 6, 2), k=cb)
                  for _ in range(rb)]
            if _grid_weights(sa, sb) <= W_MAX:
                return sa, sb
        return [[2] * ca for _ in range(ra)], [[2] * cb for _ in range(rb)]

    for trial in range(8):
        if trial == 0:
            sa = [[2] * ca for _ in range(ra)]
            sb = [[2] * cb for _ in range(rb)]
        else:
            sa, sb = supports()

        A = [[rng.sample(G.elements, sa[i][j]) for j in range(ca)]
             for i in range(ra)]
        B = [[rng.sample(G.elements, sb[a][j]) for j in range(cb)]
             for a in range(rb)]
        hx, hz, n = lifted_matrices(G, A, B)
        if not css_ok(hx, hz):
            continue
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k <= 0:
            continue

        dx, xdata = floor3(hz, hx, n)
        dz, zdata = floor3(hx, hz, n)
        d0 = min(dx, dz)
        pool.append({
            "A": A, "B": B, "k": k, "n": n, "dx": dx, "dz": dz,
            "xdata": xdata, "zdata": zdata, "pre": d0,
            "rate": int(8 * k < n), "tie": rng.random()
        })

    if not pool:
        raise ValueError("no positive-k screened filling")

    pool.sort(key=lambda c: (c["rate"], c["k"] * c["pre"] ** 2,
                             c["pre"], c["k"], c["tie"]), reverse=True)
    finalists = pool[:3]
    for c in finalists:
        if c["pre"] < 4:
            c["d"] = c["pre"]
        else:
            c["d"] = 5
            if (has_weight_four(c["xdata"]) or
                    has_weight_four(c["zdata"])):
                c["d"] = 4

    best = max(finalists, key=lambda c: (
        c["rate"], c["k"] * c["d"] ** 2, c["d"], c["k"], c["tie"]))
    return {"idx": idx, "cells": {"A": best["A"], "B": best["B"]}}


def CELL(rng, G, side, i, j, ctx):
    """Return the full screened cell chosen by NEW_CTX."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# The order-23 rolled filling on (1,2)x(4,5) had d30s=2 in all five measured codes; narrowing the A-column footprint alone did not prevent the low-weight witnesses.
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
