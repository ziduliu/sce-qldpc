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
CONCEPT = ("For metacyclic Z14:Z2(s=13), order 28, shapes (1,2)x(4,8), balanced component protection and short-cycle-screened labels should improve worst-basis circuit fitness over the rolled parent's 5.766.")
GROUP = ("Keep the exact parent group and order for a controlled shape comparison. Its measured fitness is 5.766; q=28 remains legal at n=560. No improvement is assumed from the group alone.")
SHAPES = ("Target cell 0.200: (1,2)x(4,8) gives n=560 and k>=112, exactly the parent's bound and length. Components have 56 variables/28 checks and 224 variables/112 checks. Parent failures spread across Acol/Bcol blocks with mostly 1.1x enrichment; replace uneven component protection rather than patching an unproven hotspot.")
FILLING = ("Twenty support designs cross A weights (3,3)/(3,4) with ten balanced B zero/binomial geometries. Every B row has eight terms and every B column four. Maximum HX/HZ weights are 10/11 or 11/12. Mixed-coset A supports and full-order rotation-difference B binomials undergo verified-component short-dependency and cycle screening; final actual k is restricted to 112..125. Six-round CNOT counts are 28224 or 30240, so the denser profile explicitly tests protection against additional gate exposure. Parent failures and logical flips were widespread; measured pL, not these screens, decides success.")
FAMILY_NAME = "r020_m14r2s13_12x48_balanced_component_v1"

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

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (4, 8)               # (rb, cb)


# Each cycle moves one term per participating row from one column
# to the next. All B row totals stay eight and column totals stay four.
# These change support geometry, not just group labels.
_B_GEOMETRIES = (
    (),
    (((0, 1), (0, 1)),),
    (((0, 1, 2), (0, 1, 2)),),
    (((0, 1, 2, 3), (0, 1, 2, 3)),),
    (((0, 1), (0, 1)), ((2, 3), (2, 3))),
    (((0, 1), (0, 1)), ((0, 1), (2, 3))),
    (((0, 1, 2), (0, 1, 2)), ((0, 3), (3, 4))),
    (((0, 1, 2), (0, 1, 2)), ((1, 2, 3), (3, 4, 5))),
    (((0, 1, 2, 3), (0, 1, 2, 3)),
     ((0, 1, 2, 3), (4, 5, 6, 7))),
    (((0, 1), (0, 1)), ((0, 1), (2, 3)),
     ((0, 1), (4, 5)), ((0, 1), (6, 7))),
)


def _copy_cells(grid):
    return [[list(cell) for cell in row] for row in grid]


def _verified_component(G, side, grid):
    """Extract a component from exactly the scaffold's CSS lift."""
    q = G.q
    mask = (1 << q) - 1
    if side == "A":
        probe_B = [[[G.e] for _ in range(B_SHAPE[1])]
                   for _ in range(B_SHAPE[0])]
        hx, _, _ = lifted_matrices(G, grid, probe_B)
        stride = B_SHAPE[1] * q
        return [(row & mask) |
                (((row >> stride) & mask) << q)
                for row in hx[:q]]
    probe_A = [[[G.e] for _ in range(A_SHAPE[1])]]
    _, hz, _ = lifted_matrices(G, probe_A, grid)
    bmask = (1 << (B_SHAPE[1] * q)) - 1
    return [row & bmask for row in hz[:B_SHAPE[0] * q]]


def _cycle_exposure(rows, width):
    """Exact four-cycles and a six-cycle proxy from verified rows."""
    adj = [0] * len(rows)
    degrees = [0] * width
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
    for i in range(len(rows)):
        later = adj[i] >> (i + 1)
        while later:
            bit = later & -later
            j = i + 1 + bit.bit_length() - 1
            triangles += ((adj[i] & adj[j]) >> (j + 1)).bit_count()
            later ^= bit
    stars = sum(d * (d - 1) * (d - 2) // 6 for d in degrees)
    # This equals the six-cycle count when four == 0.
    return four, max(0, triangles - stars)


def _dependency_exposure(rows, width, higher=False):
    """Kernel collision exposure; not a claim of logical distance."""
    cols = [0] * width
    for r, row in enumerate(rows):
        bits = row
        while bits:
            bit = bits & -bits
            cols[bit.bit_length() - 1] |= 1 << r
            bits ^= bit
    singles = {}
    low = 0
    for c in cols:
        low += int(c == 0) + singles.get(c, 0)
        singles[c] = singles.get(c, 0) + 1
    pairs = {}
    for i, c in enumerate(cols):
        for j in range(i):
            s = c ^ cols[j]
            low += int(s == 0) + int(s in singles) + pairs.get(s, 0)
            pairs[s] = pairs.get(s, 0) + 1
    if not higher:
        return low, 0, 0
    triples = {}
    five = six = 0
    for i in range(width):
        for j in range(i):
            ab = cols[i] ^ cols[j]
            for h in range(j):
                s = ab ^ cols[h]
                five += int(s in pairs)
                six += triples.get(s, 0)
                triples[s] = triples.get(s, 0) + 1
    # With low == 0, these are weight-five/six kernel partitions.
    return low, five, six


def _sample_kernel_min(rows, width, seed):
    """Verified kernel-word upper bound from randomized systematic bases."""
    srng = random.Random(seed)
    best = width + 1
    for _ in range(48):
        rr = list(rows)
        order = list(range(width))
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
        basis = []
        for c in order:
            if c in occupied:
                continue
            v = 1 << c
            for p, row in zip(pivots, rr[:rank]):
                if (row >> c) & 1:
                    v |= 1 << p
            basis.append(v)
            best = min(best, v.bit_count())
        for i, v in enumerate(basis):
            for u in basis[:i]:
                best = min(best, (v ^ u).bit_count())
        if best <= 4:
            break
    return best


def NEW_CTX(rng, G, idx):
    """Optimize independent labels within one of twenty support designs."""
    import math

    case = idx % 20
    sa = [[3, 3 if case < 10 else 4]]
    sb = [[1] * B_SHAPE[1] for _ in range(B_SHAPE[0])]
    for rows, cols in _B_GEOMETRIES[case % 10]:
        for t, i in enumerate(rows):
            sb[i][cols[t]] -= 1
            sb[i][cols[(t + 1) % len(cols)]] += 1

    assert all(sum(row) == 8 for row in sb)
    assert all(sum(row[j] for row in sb) == 4 for j in range(8))
    assert all(0 <= w <= 2 for row in sb for w in row)
    assert _grid_weights(sa, sb) <= 12
    assert _no_dead_lines(sa, sb)

    pools = [[g for g in G.elements if g[1] == b] for b in (0, 1)]
    m = G.q // 2

    def draw(side, counts):
        out = []
        for row in counts:
            cells = []
            for w in row:
                if w == 0:
                    cell = []
                elif w == 1:
                    cell = [G.pick(rng)]
                elif side == "B":
                    b = rng.randrange(2)
                    while True:
                        cell = rng.sample(pools[b], 2)
                        if math.gcd((cell[0][0] - cell[1][0]) % m, m) == 1:
                            break
                else:
                    # Both A cells are mixed; preserve this during search.
                    # Pure rotation-only A cells can introduce cheap
                    # commutative syzygies in this small lift.
                    b = rng.randrange(2)
                    cell = (rng.sample(pools[b], w - 1) +
                            rng.sample(pools[1 - b], 1))
                cells.append(cell)
            out.append(cells)
        return out

    def mutate(grid, side):
        active = [(i, j) for i, row in enumerate(grid)
                  for j, cell in enumerate(row) if cell]
        i, j = rng.choice(active)
        cell = grid[i][j]
        t = rng.randrange(len(cell))
        old = cell[t]
        pool = G.elements if len(cell) == 1 else pools[old[1]]
        occupied = {G._key(g) for g in cell}
        while True:
            new = rng.choice(pool)
            if G._key(new) in occupied:
                continue
            if side == "B" and len(cell) == 2:
                other = cell[1 - t]
                if math.gcd((new[0] - other[0]) % m, m) != 1:
                    continue
            break
        cell[t] = new
        return cell, t, old

    def analyze(side, grid):
        rows = _verified_component(G, side, grid)
        width = (A_SHAPE[1] if side == "A" else B_SHAPE[1]) * G.q
        four, six = _cycle_exposure(rows, width)
        low = _dependency_exposure(rows, width)[0] if side == "A" else 0
        energy = 100000.0 * low + (100.0 * four + six / 10.0) / G.q
        return energy, rows, four, six

    def optimize(side, counts, restarts, steps):
        candidates = []
        for restart in range(restarts):
            grid = draw(side, counts)
            current = analyze(side, grid)
            local_grid = _copy_cells(grid)
            local = current
            for step in range(steps):
                cell, t, old = mutate(grid, side)
                proposed = analyze(side, grid)
                delta = proposed[0] - current[0]
                temperature = 12.0 * (0.975 ** step) + 0.15
                if delta <= 0 or rng.random() < math.exp(
                        -min(700.0, delta / temperature)):
                    current = proposed
                    if current[0] < local[0]:
                        local_grid = _copy_cells(grid)
                        local = current
                else:
                    cell[t] = old
            width = (A_SHAPE[1] if side == "A" else B_SHAPE[1]) * G.q
            dep = _dependency_exposure(local[1], width, side == "A")
            distance = (_sample_kernel_min(
                local[1], width, 770003 + 1009 * idx + restart)
                if side == "A" else 0)
            candidates.append((local_grid, local, dep, distance))
        return candidates

    acandidates = optimize("A", sa, 4, 140)
    bcandidates = optimize("B", sb, 2, 180)
    best = None
    best_key = None
    best_matrices = None

    for A, aq, adep, da in acandidates:
        for B, bq, bdep, _ in bcandidates:
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not 112 <= k <= 125:
                continue
            # Remove observed cheap ambiguities first, then prioritize
            # B girth and A/B cycle exposure. Distance is only a bounded
            # tie-breaker, never a substitute for measured circuit pL.
            key = (adep[0] + bdep[0], adep[1], adep[2],
                   bq[2], aq[2], bq[3], aq[3],
                   -min(da, 12), k - 112)
            if best_key is None or key < best_key:
                best_key = key
                best = {"A": _copy_cells(A), "B": _copy_cells(B)}
                best_matrices = (hx, hz)

    # A bounded fresh-label rescue retains the same deliberate support
    # design and the actual-rate gate; it never repairs an illegal slot.
    if best is None:
        for _ in range(128):
            A, B = draw("A", sa), draw("B", sb)
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if 112 <= k <= 125:
                best = {"A": A, "B": B}
                best_matrices = (hx, hz)
                break
    if best is None:
        raise ValueError("no filling retained the 0.200 actual-rate band")
    hx, hz = best_matrices
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) <= 12
    return {"idx": idx, "case": case, "cells": best}


def CELL(rng, G, side, i, j, ctx):
    """Return the complete selected cell without additional label draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: d2s=10 alone ensures near-ceiling circuit reliability on the parent bed: slots 22 and 39 scored only 6.026 and 6.122, versus ceilings 7.54 and 7.58.
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
