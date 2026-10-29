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
CONCEPT = ("On inversion-metacyclic Z17:Z2, order 34, A=(2,6), B=(2,6), balanced cubic components with nontrivial reflection holonomy and verified kernel-witness selection should produce five measured distance readings >=12; this is a falsifiable selection hypothesis, not a distance certificate.")
GROUP = ("Keep the parent's inversion-metacyclic construction rule, resizing m=74 to m=17 and retaining r=2, s=m-1. Factor 40 permits n=1360 at q=34; 17 is the largest odd prime rotation order fitting this factor. Every nonidentity rotation has order 17. Related inversion beds measured five distances of ten at q=44 and twelve at q=34, but this new shape bed has no established distance improvement.")
SHAPES = ("A=B=(2,6): n=1360, k>=34*4*4=544, guaranteed rate 0.400 targets cell 0.400. Equal components have 204 variables and 68 checks each. Parent selected minima were four and occupied sector-1 Acol/Bcol blocks across both sides, prominently Acol1 and Bcol1; replace both single-row components rather than reinforce only one block.")
FILLING = ("Each component row has nine terms and each column three, so HX and HZ both have maximum weight twelve. Twenty cards cross five cubic support-profile pairs with four reflection-holonomy/cycle objectives. Cells stay in one coset, avoiding involutory binomial ratios; split columns carry differing reflection holonomies. Optimize actual component short-cycle statistics extracted from verified lifts, then select using exact short-kernel screens and accumulated randomized kernel-witness readings. Exact full lifted ranks gate [0.375,0.425), with structurally imperfect but valid fallbacks. No component screen or search reading certifies quantum distance.")
HOW_AB = FILLING
FAMILY_NAME = "r040f_m17r2s16_26x26_cubic_holonomy_kernel"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 34

def custom_ops(q):
    # Same inversion-metacyclic construction, resized for factor 40.
    # Z_17 x| Z_2: y x y^-1 = x^16; 16^2 = 1 mod 17.
    m, r, s = 17, 2, 16
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 6)               # (ra, ca)
B_SHAPE = (2, 6)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Construct one deliberate cubic/holonomy card on verified lifts."""
    import math

    q = G.q
    m = q // 2
    card, cohort = idx % 20, idx // 20
    mode = card % 4
    profiles = (
        [(2, 1)] * 3 + [(1, 2)] * 3,
        [(3, 0), (2, 1)] + [(1, 2)] * 4,
        [(3, 0), (0, 3)] + [(2, 1)] * 2 + [(1, 2)] * 2,
    )
    pairs = ((0, 0), (0, 1), (1, 1), (0, 2), (1, 2))
    pa, pb = pairs[card // 4]

    def copy_grid(grid):
        return [[list(cell) for cell in row] for row in grid]

    def signature(grid):
        return tuple(tuple(sorted(G.idx(g) for g in cell))
                     for row in grid for cell in row)

    def prescription(profile, side):
        pattern = list(profiles[profile])
        rng.shuffle(pattern)
        if rng.randrange(2):
            pattern = [(b, a) for a, b in pattern]
        support = [[v[i] for v in pattern] for i in range(2)]
        bits = [[rng.randrange(2) for _ in pattern] for _ in range(2)]
        split = [j for j, (a, b) in enumerate(pattern) if a and b]
        rng.shuffle(split)
        # Both holonomies occur; these are not global coset flips.
        ones = 1 + ((mode + side + cohort) % (len(split) - 1))
        for t, j in enumerate(split):
            bits[1][j] = bits[0][j] ^ int(t < ones)
        assert [sum(row) for row in support] == [9, 9]
        assert all(sum(support[i][j] for i in range(2)) == 3
                   for j in range(6))
        return support, bits

    sa, ba = prescription(pa, 0)
    sb, bb = prescription(pb, 1)
    assert _grid_weights(sa, sb) == 12
    assert _no_dead_lines(sa, sb)

    def legal_exponents(values):
        if len(set(values)) != len(values):
            return False
        gaps = []
        for a, x in enumerate(values):
            for y in values[:a]:
                d = (x - y) % m
                gaps.append(min(d, m - d))
        return len(gaps) == len(set(gaps))

    def fresh_cell(count, bit):
        while True:
            values = rng.sample(range(m), count)
            if legal_exponents(values):
                return [(a, bit) for a in values]

    def fresh(support, bits):
        return [[fresh_cell(count, bits[i][j])
                 for j, count in enumerate(row)]
                for i, row in enumerate(support)]

    def component(side, grid):
        """Extract actual components through the verified constructor."""
        width = len(grid[0]) * q
        mask = (1 << width) - 1
        if side == "A":
            hx, _, _ = lifted_matrices(G, grid, [[[]]])
            rows = [row & mask for row in hx]
        else:
            _, hz, _ = lifted_matrices(G, [[[]]], grid)
            rows = [row & mask for row in hz]
        columns = [0] * width
        for r, source in enumerate(rows):
            while source:
                low = source & -source
                columns[low.bit_length() - 1] |= 1 << r
                source ^= low
        assert all(c.bit_count() == 3 for c in columns)
        return rows, columns

    def cycle_statistics(columns):
        """Exact Tanner C4; exact C6 when C4 is zero."""
        h = 2 * q
        adjacency = [0] * h
        incidence = [[] for _ in range(h)]
        four = 0
        for v, syndrome in enumerate(columns):
            checks = []
            while syndrome:
                low = syndrome & -syndrome
                u = low.bit_length() - 1
                checks.append(u)
                incidence[u].append(v)
                syndrome ^= low
            for a, u in enumerate(checks):
                for vcheck in checks[:a]:
                    four += int(bool(adjacency[u] & (1 << vcheck)))
                    adjacency[u] |= 1 << vcheck
                    adjacency[vcheck] |= 1 << u
        if four:
            return (four, 0, 0, 0)

        triangles = 0
        projected_four = 0
        peaks = []
        for u in range(h):
            local = 0
            for v in range(u + 1, h):
                common = (adjacency[u] & adjacency[v]).bit_count()
                projected_four += common * (common - 1) // 2
                if adjacency[u] & (1 << v):
                    triangles += common
            remaining = adjacency[u]
            while remaining:
                low = remaining & -remaining
                v = low.bit_length() - 1
                local += (adjacency[u] & adjacency[v]).bit_count()
                remaining ^= low
            peaks.append(local // 2 - len(incidence[u]))
        return (0, triangles // 3 - len(columns),
                projected_four // 2, max(peaks))

    def inspect(side, grid):
        rows, columns = component(side, grid)
        stats = cycle_statistics(columns)
        four, six, projected, peak = stats
        if four:
            cost = 1000000 * four
        else:
            cost = six
            if mode == 1:
                cost += projected / 24.0
            elif mode == 2:
                cost += 4.0 * peak
            elif mode == 3:
                cost += projected / 8.0 + 2.0 * peak
            cost += (cohort % 3) * peak / 2.0
        return cost, stats, rows, columns

    def tiny_kernel(columns):
        """Exact detection of nonzero component words of weight <=4."""
        singles = set(columns)
        if 0 in singles or len(singles) != len(columns):
            return True
        pairs_seen = set()
        for a, x in enumerate(columns):
            for y in columns[:a]:
                syndrome = x ^ y
                if syndrome in singles or syndrome in pairs_seen:
                    return True
                pairs_seen.add(syndrome)
        return False

    def weight_six(columns):
        """Exact six-word screen after <=4 and Tanner C4 exclusion.

        Cubic columns in a six-word kernel each have at least three
        distinct neighbors within that word. Two disjoint adjacent pairs
        exist, so a partition into two triples can have an adjacent pair
        in each triple. Enumerate this sufficient class of triples.
        """
        nc = len(columns)
        incidence = [[] for _ in range(2 * q)]
        for v, syndrome in enumerate(columns):
            while syndrome:
                low = syndrome & -syndrome
                incidence[low.bit_length() - 1].append(v)
                syndrome ^= low
        adjacency = [set() for _ in columns]
        for variables in incidence:
            for v in variables:
                adjacency[v].update(variables)
                adjacency[v].discard(v)
        seen = {}
        for a in range(nc):
            for b in sorted(adjacency[a]):
                if b <= a:
                    continue
                pair = columns[a] ^ columns[b]
                for c in range(nc):
                    if c == a or c == b:
                        continue
                    lo, hi = min(a, c), max(b, c)
                    mid = a + b + c - lo - hi
                    tag = lo + nc * mid + nc * nc * hi
                    syndrome = pair ^ columns[c]
                    old = seen.get(syndrome)
                    if old is not None and old != tag:
                        return True
                    seen[syndrome] = tag
        return False

    def kernel_reading(columns, trials):
        """Upper reading from genuine kernel vectors, not a lower bound."""
        width = len(columns)
        best = width + 1
        order = list(range(width))
        for _ in range(trials):
            rng.shuffle(order)
            pivots, words = {}, []
            for v in order:
                syndrome, word = columns[v], 1 << v
                while syndrome:
                    p = syndrome.bit_length() - 1
                    previous = pivots.get(p)
                    if previous is None:
                        pivots[p] = (syndrome, word)
                        break
                    syndrome ^= previous[0]
                    word ^= previous[1]
                if not syndrome:
                    words.append(word)
                    best = min(best, word.bit_count())

            words.sort(key=int.bit_count)
            for i, word in enumerate(words[:24]):
                for other in words[i + 1:]:
                    candidate = word ^ other
                    if candidate:
                        best = min(best, candidate.bit_count())
            short = words[:12]
            for i, word in enumerate(short):
                for j in range(i):
                    pair = word ^ short[j]
                    for t in range(j):
                        candidate = pair ^ short[t]
                        if candidate:
                            best = min(best, candidate.bit_count())
            if best <= 6:
                break
        return best

    def optimize(side, support, bits):
        keepers = {}

        def remember(grid, record):
            key = signature(grid)
            if key not in keepers:
                keepers[key] = (copy_grid(grid), record)
            if len(keepers) > 6:
                worst = max(keepers, key=lambda k: keepers[k][1][0])
                del keepers[worst]

        starts = []
        for _ in range(8):
            grid = fresh(support, bits)
            record = inspect(side, grid)
            starts.append((record[0], grid, record))
            remember(grid, record)
        starts.sort(key=lambda entry: entry[0])

        for _, initial, initial_record in starts[:2]:
            grid = copy_grid(initial)
            current = initial_record
            positions = [(i, j) for i, row in enumerate(grid)
                         for j, cell in enumerate(row) if cell]
            steps = 160 + 16 * (cohort % 3)
            for step in range(steps):
                i, j = rng.choice(positions)
                old = list(grid[i][j])
                bit = bits[i][j]
                if step % 11 == 0:
                    new = fresh_cell(len(old), bit)
                else:
                    values = [g[0] for g in old]
                    values[rng.randrange(len(values))] = rng.randrange(m)
                    if not legal_exponents(values):
                        continue
                    new = [(a, bit) for a in values]
                if new == old:
                    continue
                grid[i][j] = new
                proposed = inspect(side, grid)
                temperature = q * (5.0 * (1.0 - step / steps) ** 2 + 0.08)
                delta = proposed[0] - current[0]
                accept = delta <= 0
                if not accept:
                    accept = rng.random() < math.exp(-delta / temperature)
                if accept:
                    current = proposed
                    remember(grid, current)
                else:
                    grid[i][j] = old

        finalists = []
        for grid, record in keepers.values():
            columns = record[3]
            reading = kernel_reading(columns, 12)
            if tiny_kernel(columns):
                reading = min(reading, 4)
            elif record[1][0] == 0 and weight_six(columns):
                reading = min(reading, 6)
            finalists.append([reading, -record[0], grid, record])
        finalists.sort(key=lambda entry: entry[:2], reverse=True)
        finalists = finalists[:3]
        for entry in finalists:
            entry[0] = min(entry[0], kernel_reading(entry[3][3], 48))
        finalists.sort(key=lambda entry: entry[:2], reverse=True)
        for entry in finalists[:2]:
            entry[0] = min(entry[0], kernel_reading(entry[3][3], 96))
        finalists.sort(key=lambda entry: entry[:2], reverse=True)
        return finalists

    afinal = optimize("A", sa, ba)
    bfinal = optimize("B", sb, bb)
    proposals = []
    for da, ac, A, ast in afinal:
        for db, bc, B, bst in bfinal:
            quality = (min(da, db), da + db, ac + bc)
            proposals.append((quality, A, B, da, db))
    proposals.sort(key=lambda entry: entry[0], reverse=True)

    for quality, A, B, da, db in proposals:
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (k > 0 and 375 * n <= 1000 * k < 425 * n):
            continue
        assert n == 1360
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx + hz) <= 12
        assert _no_dead_lines(
            [[len(cell) for cell in row] for row in A],
            [[len(cell) for cell in row] for row in B])
        return {"idx": idx, "card": card, "cells": {"A": A, "B": B},
                "screen_k": k, "component_readings": (da, db),
                "quality": quality}
    raise ValueError("no optimized cubic filling in rate cell 0.400")


def CELL(rng, G, side, i, j, ctx):
    """Return the selected complete cell without further randomization."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal rolled single-row supports with unrestricted inversion-group labels reliably suppress very short logicals; the order-148 (1,3)x(1,3) parent instead measured five minima of four, with abundant sector-1 witnesses and many screened distances of two.
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
