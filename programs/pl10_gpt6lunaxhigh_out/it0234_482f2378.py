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
CONCEPT = ("Falsifiable: for cyclic C29 of order 29 on A=(2,4), B=(2,4), screened 20-22-term fillings with balanced lifted degrees will raise the five-code mean measured s above 6.9312.")
GROUP = ("Keep C29 to isolate the filling change. The order-29 (2,4)x(2,4) bed is among the run's strongest, with a recorded best near 6.9; the current program scored 6.9312.")
SHAPES = ("Keep (2,4)x(2,4): q=29 gives n=29*(16+4)=580, guaranteed k=29*(4-2)*(4-2)=116, and rate 0.200. Both sides have equal-length components, each with 116 variables and 58 checks. Parent failures were mostly diffuse: slot 32 had 30 Z failures, with modest enrichment at Acol1/Bcol2 and Acol3/Bcol3, while slot 53 had 8 X failures and a 1.5x enrichment at Acol1/Bcol2. Slot 32 also had 29 stage-3 Z decodes, and slot 53 had 81 stage-3 X decodes; neither points to a universal single-block fix.")
FILLING = ("Use 0/1/2-term cells, with every protograph row and column nonzero, and 20-22 total terms per slot. Search 16 exact lifted candidates; prefer no weight-1-to-3 logicals, then test finalists for weight-4 logicals, short cycles, and balanced qubit/check degrees. The parent’s measured designs used about 24-25k CNOTs; this lower-edge profile tests whether reducing circuit exposure and decoder-hard cases improves measured pL.")
FAMILY_NAME = "r020_c29_24x24_sparse20_22_l4"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 29

def custom_ops(q):
    # Cyclic group C_29, written additively as tuples (a, 0).
    els = [(a, 0) for a in range(q)]

    def red(g):
        return (int(g[0]) % q, 0)

    def idx(g):
        return red(g)[0]

    def mul(g, h):
        return ((g[0] + h[0]) % q, 0)

    def inv(g):
        return ((-g[0]) % q, 0)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 4)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Search low-edge supports and distinct labels using exact lifted checks."""
    def support(total, nr, nc):
        # Use active 1-term cells plus a few doubles; permit zero blocks,
        # but retain every protograph row and column.
        for _ in range(200):
            low = max(6, total - 3, nr + nc - 1)
            high = min(nr * nc, total - 1)
            active = rng.randint(low, high)
            places = rng.sample(range(nr * nc), active)
            grid = [[0] * nc for _ in range(nr)]
            for p in places:
                grid[p // nc][p % nc] = 1
            if (any(sum(row) == 0 for row in grid)
                    or any(sum(grid[i][j] for i in range(nr)) == 0
                           for j in range(nc))):
                continue
            extras = total - active
            while extras:
                choices = [(i, j) for i in range(nr) for j in range(nc)
                           if grid[i][j] == 1]
                if not choices:
                    break
                i, j = rng.choice(choices)
                grid[i][j] = 2
                extras -= 1
            if (extras == 0
                    and max(map(sum, grid)) <= 6
                    and max(sum(grid[i][j] for i in range(nr))
                            for j in range(nc)) <= 4):
                return grid
        raise ValueError("could not draw a balanced support grid")

    def logical_floor(checks, stabilizers, n):
        """Least non-stabilizer kernel weight found through weight 3;
        4 means none was found in that range."""
        pivots = {}
        for row in stabilizers:
            x = row
            while x:
                p = x.bit_length() - 1
                if p in pivots:
                    x ^= pivots[p]
                else:
                    pivots[p] = x
                    break

        def in_span(x):
            while x:
                p = x.bit_length() - 1
                if p not in pivots:
                    return False
                x ^= pivots[p]
            return True

        sig = [0] * n
        for ri, row in enumerate(checks):
            bit = 1 << ri
            while row:
                b = row & -row
                sig[b.bit_length() - 1] |= bit
                row ^= b

        buckets = {}
        for col, s in enumerate(sig):
            buckets.setdefault(s, []).append(col)

        for i, s in enumerate(sig):
            if s == 0 and not in_span(1 << i):
                return 1
        for cols in buckets.values():
            for u in range(len(cols)):
                for v in range(u + 1, len(cols)):
                    if not in_span((1 << cols[u]) | (1 << cols[v])):
                        return 2
        for i in range(n):
            for j in range(i + 1, n):
                for k in buckets.get(sig[i] ^ sig[j], ()):
                    if k > j and not in_span((1 << i) | (1 << j) | (1 << k)):
                        return 3
        return 4

    def logical_weight4(checks, stabilizers, n):
        """Return 4 if a weight-4 logical is found (or search is capped),
        otherwise 5 when no weight-4 logical is found."""
        pivots = {}
        for row in stabilizers:
            x = row
            while x:
                p = x.bit_length() - 1
                if p in pivots:
                    x ^= pivots[p]
                else:
                    pivots[p] = x
                    break

        def in_span(x):
            while x:
                p = x.bit_length() - 1
                if p not in pivots:
                    return False
                x ^= pivots[p]
            return True

        sig = [0] * n
        for ri, row in enumerate(checks):
            bits = row
            mark = 1 << ri
            while bits:
                b = bits & -bits
                sig[b.bit_length() - 1] |= mark
                bits ^= b

        pairs = {}
        for i in range(n):
            for j in range(i + 1, n):
                pairs.setdefault(sig[i] ^ sig[j], []).append((i, j))

        tested = 0
        for group in pairs.values():
            if len(group) < 2:
                continue
            for u, (i, j) in enumerate(group):
                for k, ell in group[u + 1:]:
                    tested += 1
                    if tested > 12000:
                        return 4
                    if i == k or i == ell or j == k or j == ell:
                        continue
                    word = (1 << i) | (1 << j) | (1 << k) | (1 << ell)
                    if not in_span(word):
                        return 4
        return 5

    def four_cycles(rows):
        total = 0
        for u, row in enumerate(rows):
            for other in rows[:u]:
                common = _popcount(row & other)
                total += common * (common - 1) // 2
        return total

    def degree_load(rows, n):
        degrees = [0] * n
        for row in rows:
            bits = row
            while bits:
                b = bits & -bits
                degrees[b.bit_length() - 1] += 1
                bits ^= b
        return sum(d * d for d in degrees)

    # Interleave four deliberate term budgets across slots.
    budgets = ((10, 10), (10, 11), (11, 10), (11, 11))
    target_a, target_b = budgets[idx % len(budgets)]
    candidates = []

    for _ in range(16):
        sa = support(target_a, *A_SHAPE)
        sb = support(target_b, *B_SHAPE)
        if not _no_dead_lines(sa, sb) or _grid_weights(sa, sb) > W_MAX:
            continue
        A = [[rng.sample(G.elements, count) for count in row] for row in sa]
        B = [[rng.sample(G.elements, count) for count in row] for row in sb]

        hx, hz, n = lifted_matrices(G, A, B)
        if not css_ok(hx, hz):
            continue
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k <= 0:
            continue

        dx = logical_floor(hz, hx, n)
        dz = logical_floor(hx, hz, n)
        cycles = four_cycles(hx) + four_cycles(hz)
        degrees = degree_load(hx + hz, n)
        row_weights = [_popcount(row) for row in hx + hz]
        row_cost = sum(w * w for w in row_weights)
        max_row = max(row_weights)
        incidences = sum(row_weights)
        key = (min(dx, dz), dx + dz, -cycles, -degrees,
               -row_cost, -max_row, -incidences, -abs(k - 116), rng.random())
        candidates.append((key, {"A": A, "B": B}, dx, dz, hx, hz, n,
                           cycles, degrees, row_cost, max_row, incidences, k))

    if not candidates:
        raise ValueError("no positive-k CSS filling found")

    # Spend the weight-4 search only on the strongest weight-1-to-3 candidates.
    best_key = None
    best_cells = None
    finalists = sorted(candidates, key=lambda x: x[0], reverse=True)[:4]
    for (core, cells, dx, dz, hx, hz, n, cycles, degrees,
         row_cost, max_row, incidences, k) in finalists:
        qx = logical_weight4(hz, hx, n) if dx == 4 else dx
        qz = logical_weight4(hx, hz, n) if dz == 4 else dz
        key = (min(qx, qz), qx + qz, min(dx, dz), dx + dz,
               -cycles, -degrees, -row_cost, -max_row, -incidences,
               -abs(k - 116), core[-1])
        if best_key is None or key > best_key:
            best_key, best_cells = key, cells

    return {"idx": idx, "cells": best_cells}


def CELL(rng, G, side, i, j, ctx):
    """Return the jointly selected full cell, including a possible zero block."""
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
