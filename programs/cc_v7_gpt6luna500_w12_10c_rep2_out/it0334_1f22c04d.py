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
CONCEPT = ("Falsifiable: for cyclic C37 of order 37 on A=(1,3), B=(1,5), varied legal profiles with locally quadratic labels screened through weight 4 will yield at least three of five measured codes with d30s>=4 and fitness above 5.3286 in the k/n~0.500 cell.")
GROUP = ("Keep C37: the current measured bed has fitness 5.3286, with d30s=3 in its five measured codes. For shape factor 16, order 37 is the largest lift order under n<600, giving n=592.")
SHAPES = ("Keep (1,3)x(1,5): the guaranteed rate is 8/16=0.500, and q=37 gives n=592. The one-row component codes have nominal parameters [3,2] and [5,4], so their lengths differ; the lifted sectors have 555 and 37 qubits. The parent's weight-3 X witnesses were concentrated in sector 1, especially Acol1/2 with Bcol2/3; Z witnesses also crossed Acol0-2 and multiple Bcol blocks.")
FILLING = ("Cycle through 16 legal nonzero support profiles and generate up to 20 distinct locally quadratic labelings per slot. Screen the verified lifted matrices for non-stabilizer logicals through weight 3, then check weight 4 for the best candidates that pass. Prefer candidates in the target rate band with larger screened k*d^2/n and fewer 4-cycles. This targets the parent's measured weight-3 witnesses; screening is a selection heuristic, not a distance guarantee.")
FAMILY_NAME = "r050_C37_13x15_curved_profiles_screen4"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 37

def custom_ops(q):
    # Cyclic group C37, represented additively by one-tuples.
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

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 5)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Search varied legal profiles and refine low-weight logical screens."""
    profiles = (
        ((2, 2, 2), (2, 2, 2, 2, 2)),
        ((1, 2, 3), (1, 2, 2, 2, 2)),
        ((3, 2, 1), (2, 2, 1, 2, 2)),
        ((2, 1, 3), (2, 1, 2, 2, 2)),
        ((3, 2, 2), (1, 2, 2, 2, 2)),
        ((2, 3, 2), (2, 1, 2, 2, 2)),
        ((2, 2, 3), (2, 2, 1, 2, 2)),
        ((1, 2, 2), (2, 2, 2, 2, 2)),
        ((2, 1, 2), (2, 2, 2, 2, 2)),
        ((2, 2, 1), (2, 2, 2, 2, 2)),
        ((2, 2, 2), (3, 1, 2, 2, 2)),
        ((2, 2, 2), (2, 3, 1, 2, 2)),
        ((2, 2, 2), (2, 2, 3, 1, 2)),
        ((3, 1, 2), (2, 2, 2, 1, 2)),
        ((1, 3, 2), (2, 2, 1, 2, 2)),
        ((2, 3, 1), (2, 1, 2, 2, 2)),
    )

    def rotate(values, amount):
        amount %= len(values)
        return list(values[amount:]) + list(values[:amount])

    def make_cell(count, side, i, j, trial):
        q = G.q
        salt = idx * 7 + trial * 11 + 13 * (side == "B") + 5 * i + 3 * j
        start = (rng.randrange(q) + salt) % q
        step = 1 + (rng.randrange(q - 1) + salt) % (q - 1)
        bend = (rng.randrange(q) + 3 * salt + trial * j) % q
        values = []
        for t in range(count):
            value = (start + step * t + bend * (t * (t - 1) // 2)) % q
            while value in values:
                value = (value + 1) % q
            values.append(value)
        return [G.elements[v] for v in values]

    def row_basis(rows):
        pivots = {}
        for row in rows:
            v = row
            while v:
                p = v.bit_length() - 1
                if p in pivots:
                    v ^= pivots[p]
                else:
                    pivots[p] = v
                    break
        return pivots

    def in_span(v, pivots):
        while v:
            p = v.bit_length() - 1
            if p not in pivots:
                return False
            v ^= pivots[p]
        return True

    def columns(checks, n):
        cols = [0] * n
        for r, row in enumerate(checks):
            bit = 1 << r
            while row:
                low = row & -row
                cols[low.bit_length() - 1] |= bit
                row ^= low
        return cols

    def logical_floor(cols, pivots):
        """Exact minimum non-stabilizer kernel weight through three; 4 means none found."""
        n = len(cols)
        by_sig = {}
        for i, sig in enumerate(cols):
            bit_i = 1 << i
            if sig == 0 and not in_span(bit_i, pivots):
                return 1
            for j in by_sig.get(sig, ()):
                if not in_span(bit_i | (1 << j), pivots):
                    return 2
            by_sig.setdefault(sig, []).append(i)

        by_sig = {}
        for i, sig in enumerate(cols):
            by_sig.setdefault(sig, []).append(i)
        for i in range(n):
            bit_i = 1 << i
            for j in range(i + 1, n):
                target = cols[i] ^ cols[j]
                for k in by_sig.get(target, ()):
                    if k > j and not in_span(bit_i | (1 << j) | (1 << k), pivots):
                        return 3
        return 4

    def has_weight4(cols, pivots, limit=80000):
        """True if found, False if exhausted, None if the collision scan is capped."""
        seen = {}
        comparisons = 0
        n = len(cols)
        for i in range(n):
            bit_i = 1 << i
            for j in range(i + 1, n):
                sig = cols[i] ^ cols[j]
                old = seen.get(sig)
                if old is None:
                    seen[sig] = [(i, j)]
                    continue
                bit_j = 1 << j
                for a, b in old:
                    if a == i or a == j or b == i or b == j:
                        continue
                    comparisons += 1
                    if comparisons > limit:
                        return None
                    mask = bit_i | bit_j | (1 << a) | (1 << b)
                    if not in_span(mask, pivots):
                        return True
                old.append((i, j))
        return False

    def four_cycles(rows):
        seen = {}
        total = 0
        for row in rows:
            cols = []
            while row:
                bit = row & -row
                cols.append(bit.bit_length() - 1)
                row ^= bit
            for a in range(len(cols)):
                for b in range(a):
                    pair = (cols[b], cols[a])
                    old = seen.get(pair, 0)
                    total += old
                    seen[pair] = old + 1
        return total

    candidates = []
    for trial in range(20):
        ap, bp = profiles[(idx + 5 * trial) % len(profiles)]
        ap = rotate(ap, idx + 2 * trial)
        bp = rotate(bp, 2 * idx + trial)
        sa, sb = [list(ap)], [list(bp)]
        if _grid_weights(sa, sb) > W_MAX or not _no_dead_lines(sa, sb):
            continue

        A = [[make_cell(sa[i][j], "A", i, j, trial)
              for j in range(A_SHAPE[1])] for i in range(A_SHAPE[0])]
        B = [[make_cell(sb[a][j], "B", a, j, trial)
              for j in range(B_SHAPE[1])] for a in range(B_SHAPE[0])]
        hx, hz, n = lifted_matrices(G, A, B)
        if not css_ok(hx, hz):
            continue
        bhx, bhz = row_basis(hx), row_basis(hz)
        k = n - len(bhx) - len(bhz)
        if k <= 0:
            continue

        sx, sz = columns(hz, n), columns(hx, n)
        dx, dz = logical_floor(sx, bhx), logical_floor(sz, bhz)
        c4 = four_cycles(hx) + four_cycles(hz)
        candidates.append({
            "A": A, "B": B, "n": n, "k": k, "bhx": bhx, "bhz": bhz,
            "sx": sx, "sz": sz, "dx": dx, "dz": dz, "c4": c4,
            "band": 19 * n <= 40 * k <= 21 * n, "tie": rng.random()
        })

    if not candidates:
        raise ValueError("no positive-k screened filling")

    def preliminary(c):
        d = min(c["dx"], c["dz"])
        return (int(c["band"]), d, c["dx"] + c["dz"],
                c["k"] * d * d / c["n"], c["k"], -c["c4"], c["tie"])

    finalists = sorted(
        (c for c in candidates if min(c["dx"], c["dz"]) >= 4),
        key=preliminary, reverse=True
    )[:4]

    for c in finalists:
        wx = has_weight4(c["sx"], c["bhx"])
        wz = has_weight4(c["sz"], c["bhz"])
        c["dx"] = 4 if wx is not False else 5
        c["dz"] = 4 if wz is not False else 5

    def final_key(c):
        d = min(c["dx"], c["dz"])
        return (int(c["band"]), d, c["k"] * d * d / c["n"],
                c["dx"] + c["dz"], c["k"], -c["c4"], c["tie"])

    best = max(candidates, key=final_key)
    return {"idx": idx, "cells": {"A": best["A"], "B": best["B"]}}


def CELL(rng, G, side, i, j, ctx):
    """Return the full cell selected by the profile and logical screen."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Rolled D11 fillings on (1,5)x(2,5) measured distance 2 in all five measured codes, refuting the hypothesis that this block layout avoids weight-2 logicals.
# Rolled C37 fillings on (1,3)x(1,5) measured d30s=2 in all five codes, refuting the hypothesis that rolled supports and labels reach mean distance 3.
# Quadratic-progression fillings left weight-3 witnesses in the measured C37 codes; balanced profiles screened through weight 3 are the new hypothesis for reaching d30s>=4.
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
