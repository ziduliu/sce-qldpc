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
CONCEPT = ("Metacyclic Z29:Z1, order 29, shapes (3,6)x(3,7): balanced component degrees and verified kernel-witness screening should yield five new distance readings above 14 in rate cell 0.250.")
GROUP = ("Keep the parent's metacyclic construction rule and cyclic specialization; resize to m=29,r=1,s=1. The destination Z29 (3,6)x(3,7) bed measured fitness 46.2 with five distance-14 readings, supporting this bed without proving its distances.")
SHAPES = ("A=(3,6), B=(3,7), q=29 give n=1479 and k>=348, rate>=12/51=0.235294. Components have lengths 174 and 203 with 87 checks each. Parent weight-6 logicals occupied sector 1 across many Acol/Bcol blocks; strengthen both components rather than one block.")
FILLING = ("Twenty prescribed zero-budget pairs distinguish support designs. Both component row sums are 8; A columns sum to 4, B columns to 3 or 4, so both quantum weight maxima are 12. Optimize cyclic voltage collisions, screen verified component-kernel witnesses from minors and randomized information sets, and retain actual lifted rates in [0.225,0.275). Witness readings are heuristics, not certified quantum distances.")
HOW_AB = FILLING
FAMILY_NAME = "r025_m29r1s1_36x37_balanced_kernel_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 29

def custom_ops(q):
    # Same metacyclic rule, resized cyclic specialization.
    m, r, s = 29, 1, 1
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (3, 6)               # (ra, ca)
B_SHAPE = (3, 7)               # (rb, cb)


def _balanced_support(rng, columns, zeros):
    """Exact margins; each column touches at least two base checks."""
    from functools import lru_cache
    options = []
    for total in columns:
        opts = [
            (a, b, total-a-b)
            for a in range(4) for b in range(4)
            if 0 <= total-a-b <= 3
            and sum(v > 0 for v in (a, b, total-a-b)) >= 2
        ]
        rng.shuffle(opts)
        options.append(opts)

    @lru_cache(None)
    def feasible(j, a, b, c, z):
        if min(a, b, c, z) < 0:
            return False
        if j == len(columns):
            return (a, b, c, z) == (0, 0, 0, 0)
        if a+b+c != sum(columns[j:]) or z > len(columns)-j:
            return False
        for v in options[j]:
            dz = sum(x == 0 for x in v)
            if feasible(j+1, a-v[0], b-v[1], c-v[2], z-dz):
                return True
        return False

    state = (8, 8, 8, zeros)
    if not feasible(0, *state):
        raise ValueError("infeasible deliberate support profile")
    grid = [[], [], []]
    for j in range(len(columns)):
        choices = []
        for v in options[j]:
            nxt = tuple(state[t]-v[t] for t in range(3))
            nz = state[3]-sum(x == 0 for x in v)
            if feasible(j+1, *nxt, nz):
                choices.append((v, nxt+(nz,)))
        v, state = choices[rng.randrange(len(choices))]
        for i in range(3):
            grid[i].append(v[i])
    return grid


def _count_minor_floor(grid):
    """Term-count bound used only to select a support topology."""
    from itertools import combinations, permutations
    det = {}
    for cols in combinations(range(len(grid[0])), 3):
        det[cols] = sum(
            grid[0][cols[p[0]]] *
            grid[1][cols[p[1]]] *
            grid[2][cols[p[2]]]
            for p in permutations(range(3))
        )
    return min(
        sum(det[tuple(c for c in cols if c != omitted)]
            for omitted in cols)
        for cols in combinations(range(len(grid[0])), 4)
    )


def _voltage_cost(cells, q):
    """Algebraic repeated-difference objective, not a substitute lift."""
    score = 0
    for i in range(3):
        counts = {}
        for cell in cells[i]:
            for u in range(len(cell)):
                for v in range(u):
                    d = (cell[u][0]-cell[v][0]) % q
                    d = min(d, q-d)
                    score += counts.get(d, 0)
                    counts[d] = counts.get(d, 0)+1
        for k in range(i):
            counts = {}
            for j in range(len(cells[0])):
                for g in cells[i][j]:
                    for h in cells[k][j]:
                        d = (g[0]-h[0]) % q
                        score += counts.get(d, 0)
                        counts[d] = counts.get(d, 0)+1
    return score


def _optimized_labels(rng, G, support):
    cells = [
        [rng.sample(G.elements, count) for count in row]
        for row in support
    ]
    places = [
        (i, j, t)
        for i, row in enumerate(cells)
        for j, cell in enumerate(row)
        for t in range(len(cell))
    ]
    cost = _voltage_cost(cells, G.q)
    best_cost = cost
    best = [[list(c) for c in row] for row in cells]
    for step in range(360):
        if best_cost == 0:
            break
        i, j, t = places[rng.randrange(len(places))]
        old = cells[i][j][t]
        proposal = G.pick(rng)
        if proposal in cells[i][j]:
            continue
        cells[i][j][t] = proposal
        new_cost = _voltage_cost(cells, G.q)
        neutral = 0.35 if step < 240 else 0.08
        accept = (
            new_cost < cost
            or (new_cost == cost and rng.random() < neutral)
            or (step < 180 and new_cost == cost+1
                and rng.random() < 0.025)
        )
        if accept:
            cost = new_cost
            if cost < best_cost:
                best_cost = cost
                best = [[list(c) for c in row] for row in cells]
        else:
            cells[i][j][t] = old
    return best, best_cost


def _component_rows(hx, hz, q):
    """Extract components from the VERIFIED full quantum construction."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q)-1
    ar = []
    for i in range(ra):
        for t in range(q):
            source = hx[(i*cb)*q+t]
            row = 0
            for j in range(ca):
                row |= ((source >> (j*cb*q)) & mask) << (j*q)
            ar.append(row)
    br = [
        hz[a*q+t] & ((1 << (cb*q))-1)
        for a in range(rb) for t in range(q)
    ]
    return ar, br


def _minor_reading(rows, cells, q, inverse=False):
    """Verified nonzero classical kernel words; not quantum-distance proofs."""
    from itertools import combinations, permutations
    mask = (1 << q)-1

    def multiply(a, b):
        out = 0
        while a:
            bit = a & -a
            shift = bit.bit_length()-1
            out ^= ((b << shift) | (b >> (q-shift))) & mask
            a ^= bit
        return out

    polys = []
    for row in cells:
        prow = []
        for cell in row:
            p = 0
            for g in cell:
                exponent = (-g[0] if inverse else g[0]) % q
                p ^= 1 << exponent
            prow.append(p)
        polys.append(prow)

    minors = {}
    width = len(cells[0])
    for cols in combinations(range(width), 3):
        p = 0
        for perm in permutations(range(3)):
            p ^= multiply(
                multiply(polys[0][cols[perm[0]]],
                         polys[1][cols[perm[1]]]),
                polys[2][cols[perm[2]]]
            )
        minors[cols] = p

    best = width*q+1
    for cols in combinations(range(width), 4):
        word = 0
        for omitted in cols:
            minor = minors[tuple(c for c in cols if c != omitted)]
            word |= minor << (omitted*q)
        if word and all((word & row).bit_count() % 2 == 0
                        for row in rows):
            best = min(best, word.bit_count())
    return best


def _information_set_reading(rows, width, rng, trials):
    """Search genuine component-kernel vectors in random column orders."""
    best = width+1
    for _ in range(trials):
        permutation = list(range(width))
        rng.shuffle(permutation)
        mapping = [1 << p for p in permutation]
        pivots = {}
        for source in rows:
            row = 0
            while source:
                bit = source & -source
                row ^= mapping[bit.bit_length()-1]
                source ^= bit
            while row:
                p = row.bit_length()-1
                if p in pivots:
                    row ^= pivots[p]
                else:
                    pivots[p] = row
                    break

        ordered = sorted(pivots)
        for at, p in enumerate(ordered):
            bit = 1 << p
            reduced = pivots[p]
            for higher in ordered[at+1:]:
                if pivots[higher] & bit:
                    pivots[higher] ^= reduced

        words = []
        for f in range(width):
            if f in pivots:
                continue
            bit = 1 << f
            word = bit
            for p in ordered:
                if pivots[p] & bit:
                    word |= 1 << p
            best = min(best, word.bit_count())
            words.append(word)

        words.sort(key=int.bit_count)
        shortlist = words[:18]
        for i, word in enumerate(shortlist):
            for other in shortlist[:i]:
                best = min(best, (word ^ other).bit_count())
        if best <= 8:
            break
    return best


def NEW_CTX(rng, G, idx):
    """Twenty support hypotheses, each internally optimized and screened."""
    design = idx % 20
    zero_a = design % 4
    zero_b = design // 4
    sa = max(
        (_balanced_support(rng, [4]*6, zero_a) for _ in range(5)),
        key=_count_minor_floor
    )
    sb = max(
        (_balanced_support(rng, [4, 4, 4, 3, 3, 3, 3], zero_b)
         for _ in range(5)),
        key=_count_minor_floor
    )
    assert _grid_weights(sa, sb) == 12
    assert _no_dead_lines(sa, sb)
    pool = []
    for attempt in range(10):
        A, cost_a = _optimized_labels(rng, G, sa)
        B, cost_b = _optimized_labels(rng, G, sb)
        hx, hz, n = lifted_matrices(G, A, B)
        k = n-gf2_rank(hx)-gf2_rank(hz)
        if not 0.225 <= k/n < 0.275:
            continue
        ar, br = _component_rows(hx, hz, G.q)
        da = min(
            _minor_reading(ar, A, G.q),
            _information_set_reading(ar, 6*G.q, rng, 20)
        )
        db = min(
            _minor_reading(br, B, G.q, inverse=True),
            _information_set_reading(br, 7*G.q, rng, 20)
        )
        key = (min(da, db), da+db, -cost_a-cost_b)
        pool.append((key, A, B, ar, br, hx, hz, k))

    if not pool:
        raise ValueError("no candidate passed the actual lifted-rate gate")
    pool.sort(key=lambda item: item[0], reverse=True)
    finalists = []
    for key, A, B, ar, br, hx, hz, k in pool[:3]:
        da = min(key[0], _information_set_reading(
            ar, 6*G.q, rng, 80))
        db = min(key[0], _information_set_reading(
            br, 7*G.q, rng, 80))
        final_key = (min(da, db), key[0], key[1], key[2])
        finalists.append((final_key, A, B, hx, hz, k))
    _, A, B, hx, hz, k = max(finalists, key=lambda item: item[0])
    assert css_ok(hx, hz)
    return {"idx": idx, "cells": {"A": A, "B": B}, "k": k}


def CELL(rng, G, side, i, j, ctx):
    """Return the already screened complete cell, without further draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal rolled w=12 fillings do not suppress short sector-1 logicals; the parent measured five codes with weight-6 witnesses.
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
