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
CONCEPT = ("Falsifiable: for cyclic C23 of order 23 on A=(1,5), B=(1,5), the existing profile-and-label screening will produce a top-five mean fitness above 8.5650 in rate cell 0.550.")
GROUP = ("Keep the cyclic construction rule and set its order to 23 for the new shape. The RUN STATE includes a C23 bed scoring 24.0 on different shapes; order 23 is also the largest lift order allowed because the shape factor is 26 and 26*23=598.")
SHAPES = ("Use (1,5)x(1,5): the rank-count bound gives k>=23*(5-1)*(5-1)=368, or k/n>=16/26 about 0.615, in the open-ended 0.550 cell; n=598. Both sides have equal-length nominal [5,4] component codes. The parent's weight-4 logicals were spread across many Acol/Bcol blocks; this changes the sector-1 block grid from 4-by-4 to 5-by-5.")
FILLING = ("Retain legal term-count profiles with counts 1-5, excluding the all-two pair, and generate eight structured cyclic labelings per profile. The verified-lift screen checks all four sectors through weight 3, then checks weight 4 for the four best candidates. Rank by screened distance and k to test whether the larger component layout avoids the parent's lightest witnesses.")
FAMILY_NAME = "r0615_C23_15x15_profile_quad4"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 23

def custom_ops(q):
    # Additive cyclic group C_34.
    els = [(a,) for a in range(q)]

    def mul(g, h):
        return ((g[0] + h[0]) % q,)

    def inv(g):
        return ((-g[0]) % q,)

    def idx(g):
        return g[0] % q

    def red(g):
        return (g[0] % q,)

    return dict(elements=els, ident=(0,), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 5)               # (ra, ca)
B_SHAPE = (1, 5)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Use a slot-specific legal degree profile and screen structured labels."""
    from itertools import combinations_with_replacement
    from math import gcd

    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    av = list(combinations_with_replacement(range(1, 6), ca))
    bv = list(combinations_with_replacement(range(1, 6), cb))
    profiles = []
    parent = (2,) * ca
    for a in av:
        for b in bv:
            if a > b or (a == parent and b == parent):
                continue
            sa0, sb0 = [list(a)], [list(b)]
            if _grid_weights(sa0, sb0) <= W_MAX:
                profiles.append((a, b))
    profiles.sort(key=lambda p: (
        -(sum(p[0]) + sum(p[1])),
        -min(sum(p[0]), sum(p[1])),
        p[0], p[1]))
    a_profile, b_profile = profiles[idx % len(profiles)]
    units = [x for x in range(1, G.q) if gcd(x, G.q) == 1]

    def basis(rows):
        piv = {}
        for v in rows:
            while v:
                b = v.bit_length() - 1
                if b in piv:
                    v ^= piv[b]
                else:
                    piv[b] = v
                    break
        return piv

    def in_span(v, piv):
        while v:
            b = v.bit_length() - 1
            if b not in piv:
                return False
            v ^= piv[b]
        return True

    def small_distance(op_rows, piv, offset, length, upto):
        lim = (1 << length) - 1
        cols = [0] * length
        for r, row in enumerate(op_rows):
            m = (row >> offset) & lim
            rb = 1 << r
            while m:
                bit = m & -m
                cols[bit.bit_length() - 1] |= rb
                m ^= bit

        def logical(indices):
            v = 0
            for x in indices:
                v |= 1 << (offset + x)
            return not in_span(v, piv)

        for i, sig in enumerate(cols):
            if sig == 0 and logical((i,)):
                return 1

        seen = {}
        for i, sig in enumerate(cols):
            for j in seen.get(sig, ()):
                if logical((i, j)):
                    return 2
            seen.setdefault(sig, []).append(i)

        pairs = {}
        for i in range(length):
            for j in range(i + 1, length):
                pairs.setdefault(cols[i] ^ cols[j], []).append((i, j))

        for k, sig in enumerate(cols):
            for i, j in pairs.get(sig, ()):
                if k != i and k != j and logical((i, j, k)):
                    return 3
        if upto <= 3:
            return 4

        for group in pairs.values():
            for a in range(len(group)):
                i, j = group[a]
                for b in range(a + 1, len(group)):
                    k, ell = group[b]
                    if i != k and i != ell and j != k and j != ell:
                        if logical((i, j, k, ell)):
                            return 4
        return 5

    def orient(profile, phase):
        vals = list(profile)
        shift = phase % len(vals)
        vals = vals[shift:] + vals[:shift]
        if (phase // len(vals)) % 2:
            vals.reverse()
        return vals

    def cell_labels(count, side, i, j, trial):
        salt = 11 * idx + 7 * trial + 13 * (side == "B") + 5 * i + 3 * j
        start = (rng.randrange(G.q) + salt) % G.q
        step = units[(idx + 3 * trial + 5 * i + 7 * j
                      + (side == "B")) % len(units)]
        bend = (rng.randrange(G.q) + 3 * idx * trial + i + 2 * j) % G.q
        values = []
        for t in range(count):
            value = (start + step * t + bend * (t * (t - 1) // 2)) % G.q
            while value in values:
                value = (value + 1) % G.q
            values.append(value)
        return [G.elements[x] for x in values]

    n1 = ca * cb * G.q
    n2 = ra * rb * G.q
    records = []
    for trial in range(8):
        ac = orient(a_profile, idx + trial)
        bc = orient(b_profile, idx + 2 * trial + 1)
        sa, sb = [ac], [bc]
        if _grid_weights(sa, sb) > W_MAX or not _no_dead_lines(sa, sb):
            continue
        A = [[cell_labels(sa[i][j], "A", i, j, trial)
              for j in range(ca)] for i in range(ra)]
        B = [[cell_labels(sb[i][j], "B", i, j, trial)
              for j in range(cb)] for i in range(rb)]
        hx, hz, n = lifted_matrices(G, A, B)
        if not css_ok(hx, hz):
            continue
        bhx, bhz = basis(hx), basis(hz)
        k = n - len(bhx) - len(bhz)
        if k <= 0:
            continue
        witnesses = (
            small_distance(hz, bhx, 0, n1, 3),
            small_distance(hz, bhx, n1, n2, 3),
            small_distance(hx, bhz, 0, n1, 3),
            small_distance(hx, bhz, n1, n2, 3))
        records.append({
            "cells": {"A": A, "B": B}, "hx": hx, "hz": hz,
            "bhx": bhx, "bhz": bhz, "n": n, "k": k,
            "terms": sum(ac) + sum(bc), "w": witnesses})

    if not records:
        raise ValueError("no positive-k legal profile candidate")
    records.sort(key=lambda r: (
        min(r["w"]), sum(r["w"]), r["k"], r["terms"]), reverse=True)

    best = None
    for rec in records[:4]:
        w = (
            small_distance(rec["hz"], rec["bhx"], 0, n1, 4),
            small_distance(rec["hz"], rec["bhx"], n1, n2, 4),
            small_distance(rec["hx"], rec["bhz"], 0, n1, 4),
            small_distance(rec["hx"], rec["bhz"], n1, n2, 4))
        d = min(w)
        score = (rec["k"] * d * d, d, sum(w), rec["k"],
                 rec["terms"], rng.random())
        if best is None or score > best[0]:
            best = (score, rec["cells"])
    return {"idx": idx, "cells": best[1]}


def CELL(rng, G, side, i, j, ctx):
    """Return the screened full cell for this slot."""
    return ctx["cells"][side][i][j]

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
