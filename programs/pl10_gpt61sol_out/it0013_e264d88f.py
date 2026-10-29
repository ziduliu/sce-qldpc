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
CONCEPT = ("For cyclic Z23, order 23, shapes (2,4)x(3,5), balanced degree-3/4 A columns and short-cycle-screened labels should reduce the rolled parent's dominant X-memory failures at check weight <=12.")
GROUP = ("Frozen Z23 bed: measured fitness 4.0963. Order 23 is the largest lift allowed by n=26q<600; this experiment changes filling, not the group.")
SHAPES = ("Frozen (2,4)x(3,5): n=598, k>=92, guaranteed rate 0.15385 in cell 0.150. Component lengths 92 and 115 have dimensions >=46. Parent X failures span Acol0/2 and several B columns with roughly 1.1x enrichment, motivating broad reinforcement rather than a single-block edit.")
FILLING = ("Twenty term-budget designs combine A totals 12..16 with B totals 16/17/19/20. Every column has degree 3 or 4. Labels undergo component short-cycle optimization using the verified lift, followed by sampled kernel-witness screening and exact k screening, preferring k<=104. Targets are the parent's widespread weight-6 Z witnesses and X failures: slot 4 failed 102/1500, while slot 5 with weight-8 witnesses failed 101/33250. These screens are circuit-reliability hypotheses, not distance certificates.")
FAMILY_NAME = "r015_m1r23s1_24x35_balanced_kernel_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 23

def custom_ops(q):
    # metacyclic Z_1 x| Z_23: y x y^-1 = x^1 (s^r = 1 mod m);
    # rolled, not designed - see make_rate_seeds.py
    m, r, s = 1, 23, 1
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 4)               # (ra, ca)
B_SHAPE = (3, 5)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Optimize labels inside an idx-specific, weight-legal design."""
    import math

    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    q = G.q

    # Twenty distinct pairs of component edge totals, not orbit copies.
    profile = idx % 20
    doubled_columns = profile % 5
    b_extra = (1, 2, 4, 5)[profile // 5]
    sa = [[1] * ca for _ in range(ra)]
    for j in range(ca):
        if j < doubled_columns:
            sa[0][j] = sa[1][j] = 2
        else:
            sa[(j - doubled_columns) % 2][j] = 2
    sb = [[1] * cb for _ in range(rb)]
    for j in range(b_extra):
        sb[(j + profile // 5) % rb][j] = 2
    assert _grid_weights(sa, sb) <= W_MAX

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def draw(counts):
        return [[rng.sample(G.elements, w) for w in row]
                for row in counts]

    def component_rows(A, B):
        # Extract both component check matrices from the EXACT CSS lift.
        hx, hz, _ = lifted_matrices(G, A, B)
        mask = (1 << q) - 1
        ha = []
        for i in range(ra):
            for r in range(q):
                row = hx[(i * cb) * q + r]
                compact = 0
                for j in range(ca):
                    compact |= ((row >> (j * cb * q)) & mask) << (j * q)
                ha.append(compact)
        bmask = (1 << (cb * q)) - 1
        hb = [hz[a * q + r] & bmask
              for a in range(rb) for r in range(q)]
        return ha, hb

    def cycle_counts(rows, nc):
        # Exact four-cycle count; triangle proxy subtracts single-column
        # stars and equals the six-cycle count when there are no 4-cycles.
        m = len(rows)
        adj = [0] * m
        degrees = [0] * nc
        four = 0
        for i, row in enumerate(rows):
            bits = row
            while bits:
                bit = bits & -bits
                degrees[bit.bit_length() - 1] += 1
                bits ^= bit
            for j in range(i):
                overlap = (row & rows[j]).bit_count()
                four += overlap * (overlap - 1) // 2
                if overlap:
                    adj[i] |= 1 << j
                    adj[j] |= 1 << i
        triangles = 0
        for i in range(m):
            later = adj[i] >> (i + 1)
            while later:
                bit = later & -later
                j = i + 1 + bit.bit_length() - 1
                triangles += ((adj[i] & adj[j]) >> (j + 1)).bit_count()
                later ^= bit
        stars = sum(d * (d - 1) * (d - 2) // 6 for d in degrees)
        return four, max(0, triangles - stars)

    def analyze(A, B):
        ha, hb = component_rows(A, B)
        a4, a6 = cycle_counts(ha, ca * q)
        b4, b6 = cycle_counts(hb, cb * q)
        four, six = a4 + b4, a6 + b6
        energy = (100.0 * four + six / 5.0) / q
        return energy, ha, hb, four, six

    def kernel_scan(rows, nc, seed, trials):
        # Random systematic kernel bases yield verified codewords, hence
        # upper bounds only. A large returned value is not a certificate.
        srng = random.Random(seed)
        best = nc + 1
        small = set()
        for _ in range(trials):
            rr = list(rows)
            order = list(range(nc))
            srng.shuffle(order)
            pivots = []
            rank = 0
            for c in order:
                bit = 1 << c
                hit = next((t for t in range(rank, len(rr))
                            if rr[t] & bit), None)
                if hit is None:
                    continue
                rr[rank], rr[hit] = rr[hit], rr[rank]
                pivot_row = rr[rank]
                for t in range(len(rr)):
                    if t != rank and rr[t] & bit:
                        rr[t] ^= pivot_row
                pivots.append(c)
                rank += 1
                if rank == len(rr):
                    break
            occupied = set(pivots)
            for c in order:
                if c in occupied:
                    continue
                v = 1 << c
                for p, row in zip(pivots, rr[:rank]):
                    if (row >> c) & 1:
                        v |= 1 << p
                weight = v.bit_count()
                best = min(best, weight)
                if weight <= 10:
                    small.add(v)
            if best <= 6:
                break
        return best, len(small)

    winner = None
    winner_quality = None
    for restart in range(4):
        A, B = draw(sa), draw(sb)
        current = analyze(A, B)
        local_A, local_B = clone(A), clone(B)
        local = current
        for step in range(220):
            grid = A if rng.random() < 0.65 else B
            i = rng.randrange(len(grid))
            j = rng.randrange(len(grid[0]))
            cell = grid[i][j]
            t = rng.randrange(len(cell))
            old = cell[t]
            occupied = {G._key(g) for g in cell}
            new = G.pick(rng)
            while G._key(new) in occupied:
                new = G.pick(rng)
            cell[t] = new
            proposed = analyze(A, B)
            delta = proposed[0] - current[0]
            temperature = 60.0 * (0.97 ** step) + 0.3
            accept = (delta <= 0 or
                      rng.random() < math.exp(-min(700.0,
                                                   delta / temperature)))
            if accept:
                current = proposed
                if current[0] < local[0]:
                    local_A, local_B = clone(A), clone(B)
                    local = current
            else:
                cell[t] = old

        A, B = local_A, local_B
        k = lifted_k(G, A, B)
        da, na = kernel_scan(local[1], ca * q,
                             271828 + idx * 1009, 160)
        db, nb = kernel_scan(local[2], cb * q,
                             314159 + idx * 1009, 96)
        # Reward removing observed small witnesses, but do not reward an
        # arbitrarily optimistic distance estimate from a short search.
        quality = (int(92 <= k <= 104),
                   int(da >= 8 and db >= 10),
                   -local[3],
                   min(da, db, 12),
                   -(na + nb),
                   -local[4],
                   min(da, 16) + min(db, 16))
        if winner_quality is None or quality > winner_quality:
            winner_quality = quality
            winner = {"A": clone(A), "B": clone(B)}

    hx, hz, _ = lifted_matrices(G, winner["A"], winner["B"])
    assert css_ok(hx, hz)
    assert lifted_k(G, winner["A"], winner["B"]) > 0
    return {"idx": idx, "cells": winner}


def CELL(rng, G, side, i, j, ctx):
    """Return the selected full cell without further random draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Stage-1 settlement is not a reliability certificate: parent slot 4 settled 1498/1500 X shots at stage 1 yet failed 102/1500.
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
