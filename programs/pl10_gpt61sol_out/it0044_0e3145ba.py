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
CONCEPT = ("For dicyclic Dic7, order 28, shapes (2,4)x(2,4), balanced degree-3/4 component columns and no involutory within-cell differences should improve worst-basis circuit reliability over the rolled dihedral parent.")
GROUP = ("Dic7 has presentation x^14=1, y^2=x^7, y*x*y^-1=x^-1. Its unique involution distinguishes it from the parent's dihedral group of order 28. Order-four and order-fourteen relative elements provide alternatives to reflection-type order-two differences. Strong nonabelian beds motivate this untried group, without certifying its reliability.")
SHAPES = ("Keep (2,4)x(2,4): n=560, guaranteed k>=112 and rate 0.200. Both components have 56 checks on 112 variables. Parent failures and weight-8/10 sector-1 witnesses spread across Acol/Bcol blocks, usually with only 1.0-1.2x enrichment; a global component change is better motivated than a single-block repair. Screen actual k to 112..124.")
FILLING = ("Twenty prescribed pairs of degree-3/4 column profiles use balanced component row sums <=8, so both CSS row-weight budgets are <=12. Binomials have relative order 14 or 4; mixed-coset trinomials exclude central-involution differences. Independently optimize component short cycles through the verified lift, then compare sampled kernel witnesses and exact ranks. Six-round CNOT counts are 24192..31248. The intended improvement is reduced diffuse, basis-asymmetric circuit failure, not distance as fitness.")
FAMILY_NAME = "r020_Dic7_24x24_balanced_ordercycle_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 28

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Dic7 requires order 28")
    # Normal form x^a y^b; y^2=x^7, not the identity.
    els = [(a, b) for b in range(2) for a in range(14)]

    def red(g):
        return (int(g[0]) % 14, int(g[1]) % 2)

    def mul(g, h):
        a, b = g
        c, d = h
        return ((a + (c if b == 0 else -c) + 7*b*d) % 14,
                (b + d) % 2)

    def inv(g):
        a, b = g
        return ((-a) % 14, 0) if b == 0 else ((a + 7) % 14, 1)

    def idx(g):
        a, b = red(g)
        return a + 14*b

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 4)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb)


def _dic_counts(rng, heavy):
    """Four live columns of degree 3/4; two nearly balanced rows."""
    import itertools
    totals = [4] * heavy + [3] * (4 - heavy)
    choices = []
    for first in itertools.product(*(range(1, t) for t in totals)):
        second = [t - a for t, a in zip(totals, first)]
        if max(first) > 3 or max(second) > 3:
            continue
        if max(sum(first), sum(second)) > 8:
            continue
        choices.append((abs(sum(first) - sum(second)), first, second))
    balance = min(c[0] for c in choices)
    choices = [c for c in choices if c[0] == balance]
    _, first, second = choices[rng.randrange(len(choices))]
    return [list(first), list(second)]


def _dic_cell(rng, G, count, policy):
    """Order-14 rotational or order-4 dicyclic binomials."""
    if count == 1:
        return [G.pick(rng)]
    if count == 2:
        g = G.pick(rng)
        if policy == 0:
            relative = (rng.choice((1, 3, 5, 9, 11, 13)), 0)
        else:
            relative = (rng.randrange(14), 1)
        return [g, G.mul(g, relative)]

    # Three terms: two in one coset and one in the other.
    # Difference (7,0), the unique involution, is never admitted.
    b = rng.randrange(2)
    a = rng.randrange(14)
    step = rng.choice((1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13))
    return [(a, b), ((a + step) % 14, b),
            (rng.randrange(14), 1 - b)]


def _dic_component_rows(G, A, B):
    """Extract component matrices from the evaluator's exact lift."""
    hx, hz, n = lifted_matrices(G, A, B)
    q = G.q
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    ar = []
    for i in range(ra):
        for t in range(q):
            source = hx[(i * cb) * q + t]
            row = 0
            for j in range(ca):
                row |= ((source >> (j * cb * q)) & mask) << (j * q)
            ar.append(row)
    bmask = (1 << (cb * q)) - 1
    br = [hz[a * q + t] & bmask
          for a in range(rb) for t in range(q)]
    return ar, br, hx, hz, n


def _dic_cycles(rows, width):
    """Exact four-cycles; six-cycle proxy is exact when four=0."""
    adjacent = [0] * len(rows)
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
                adjacent[i] |= 1 << j
                adjacent[j] |= 1 << i
    triangles = 0
    for i, neighbors in enumerate(adjacent):
        later = neighbors >> (i + 1)
        while later:
            bit = later & -later
            j = i + 1 + bit.bit_length() - 1
            triangles += ((neighbors & adjacent[j]) >> (j + 1)).bit_count()
            later ^= bit
    stars = sum(d * (d - 1) * (d - 2) // 6 for d in degrees)
    return four, max(0, triangles - stars)


def _dic_small_kernel(rows, width):
    """Exactly determine whether the component has kernel weight <=4."""
    columns = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    if 0 in columns:
        return 1
    if len(set(columns)) != width:
        return 2
    singles = set(columns)
    pairs = set()
    found_four = False
    for a in range(width):
        for b in range(a):
            s = columns[a] ^ columns[b]
            if s in singles:
                return 3
            if s in pairs:
                found_four = True
            pairs.add(s)
    return 4 if found_four else 5


def _dic_kernel_scan(rows, width, seed, trials=48):
    """Sample actual kernel words; returned distances are upper bounds."""
    srng = random.Random(seed)
    best = width + 1
    small = set()
    for _ in range(trials):
        rr = list(rows)
        order = list(range(width))
        srng.shuffle(order)
        pivots = []
        rank = 0
        for c in order:
            bit = 1 << c
            hit = next((r for r in range(rank, len(rr))
                        if rr[r] & bit), None)
            if hit is None:
                continue
            rr[rank], rr[hit] = rr[hit], rr[rank]
            pivot = rr[rank]
            for r in range(len(rr)):
                if r != rank and rr[r] & bit:
                    rr[r] ^= pivot
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
        if best <= 4:
            break
    return best, len(small)


def NEW_CTX(rng, G, idx):
    """Independent searches for twenty degree-profile design briefs."""
    import math
    profile = idx % 20
    ah, bh = profile // 4, profile % 4
    # Extra harness slots remain independent searches, not orbit copies.
    mode = (profile + idx // 20) % 3
    width_a, width_b = A_SHAPE[1] * G.q, B_SHAPE[1] * G.q

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def policy(side, i, j):
        if mode == 0:
            return 0
        if mode == 2:
            return 1
        return (i + j + (side == "B")) % 2

    def draw(sa, sb):
        return {
            side: [[_dic_cell(rng, G, w, policy(side, i, j))
                    for j, w in enumerate(row)]
                   for i, row in enumerate(counts)]
            for side, counts in (("A", sa), ("B", sb))
        }

    def analyze(cells):
        ar, br, hx, hz, n = _dic_component_rows(
            G, cells["A"], cells["B"])
        a4, a6 = _dic_cycles(ar, width_a)
        b4, b6 = _dic_cycles(br, width_b)
        four, six = a4 + b4, a6 + b6
        energy = (100.0 * four + six / 6.0) / G.q
        return energy, four, six, ar, br, hx, hz, n

    winner = None
    winner_key = None
    for restart in range(6):
        sa, sb = _dic_counts(rng, ah), _dic_counts(rng, bh)
        # Both sums are checked, not just the total term count.
        wx = max(map(sum, sa)) + max(sum(c) for c in zip(*sb))
        wz = max(map(sum, sb)) + max(sum(c) for c in zip(*sa))
        assert max(wx, wz) <= W_MAX
        cells = draw(sa, sb)
        current = analyze(cells)
        local = current
        local_cells = {s: clone(cells[s]) for s in ("A", "B")}

        for step in range(160):
            side = "A" if rng.randrange(2) == 0 else "B"
            i, j = rng.randrange(2), rng.randrange(4)
            old = cells[side][i][j]
            cells[side][i][j] = _dic_cell(
                rng, G, len(old), policy(side, i, j))
            proposed = analyze(cells)
            delta = proposed[0] - current[0]
            temperature = 30.0 * (0.965 ** step) + 0.15
            if delta <= 0 or rng.random() < math.exp(
                    -min(700.0, delta / temperature)):
                current = proposed
                if current[:3] < local[:3]:
                    local = current
                    local_cells = {s: clone(cells[s])
                                   for s in ("A", "B")}
            else:
                cells[side][i][j] = old

        _, four, six, ar, br, hx, hz, n = local
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not 112 <= k <= 124:
            continue
        small_a = _dic_small_kernel(ar, width_a)
        small_b = _dic_small_kernel(br, width_b)
        da, na = _dic_kernel_scan(
            ar, width_a, 700001 + 1009*idx + 53*restart)
        db, nb = _dic_kernel_scan(
            br, width_b, 900001 + 1013*idx + 59*restart)
        # A short kernel search is not a distance certificate, and neither
        # cycles nor kernel estimates are claimed to be circuit fitness.
        key = (min(small_a, small_b),
               int(min(da, db) >= 8),
               min(da, db, 12),
               -(na + nb),
               -four, -six,
               -abs(min(da, 16) - min(db, 16)),
               -(wx + wz), -(k - 112))
        if winner_key is None or key > winner_key:
            winner_key = key
            winner = local_cells

    # Do not void a slot merely because all annealed endpoints missed
    # the target band; search new independent fillings of its own brief.
    if winner is None:
        for _ in range(256):
            sa, sb = _dic_counts(rng, ah), _dic_counts(rng, bh)
            cells = draw(sa, sb)
            k = lifted_k(G, cells["A"], cells["B"])
            if 112 <= k <= 124:
                winner = cells
                break
    if winner is None:
        raise ValueError("no Dic7 filling passed the exact rate screen")

    hx, hz, n = lifted_matrices(G, winner["A"], winner["B"])
    assert css_ok(hx, hz)
    assert 112 <= n - gf2_rank(hx) - gf2_rank(hz) <= 124
    return {"idx": idx, "cells": winner}


def CELL(rng, G, side, i, j, ctx):
    """Return the independently selected full cell, without new draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal rolled supports and near-bound k alone do not ensure balanced memory reliability: parent slot 21 failed 100/46750 X shots versus 101/150500 Z shots, despite k-kb=4 and diffuse block enrichment.
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
