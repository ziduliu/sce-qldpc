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
CONCEPT = ("On generalized quaternion Q32=Dic8, order 32, A=(1,3), B=(2,5), degree-three component columns selected to exclude words through weight seven should raise the five measured distance readings above the parent's eight; this is a falsifiable search hypothesis, not a quantum-distance certificate.")
GROUP = ("Jump from cyclic Z17 to Dic8: x^16=1, y^2=x^8, y*x*y^-1=x^-1. Dicyclic beds reached fitness 51.9 at q=44 and 26.4 at q=84, motivating this new bed without assuming distance transfers. Order 32 gives n=544. Odd-augmentation cells are units over this 2-group; the prescribed component augmentation matrices have full row rank. Verify assembled ranks directly.")
SHAPES = ("Target 0.350 with (1,3)x(2,5): n=17*32=544, kb=6*32=192, guaranteed rate 6/17=0.352941. Components have 32/96 and 64/160 checks/variables. Parent weight-eight minima occupied sector-1 Acol/Bcol blocks broadly, especially Acol1/Acol2. Replace both component geometries and require exact k=192.")
FILLING = ("Twenty recipes cross four quaternion trinomial relative-order models with five B support profiles. Every component column has three terms; A-row total is nine and B-row totals are seven/eight, so HX<=12 and HZ<=11. Exclude central-separated label pairs. Anneal independent voltages against component four-cycles and weight-four kernel collisions, exhaustively screen component words through seven, and deepen sampled kernel-witness searches. Reject assembled logicals through weight four, verify CSS and exact rate, and return stored cells. Component readings remain upper-bound witnesses, not quantum-distance certificates.")
FAMILY_NAME = "r035_dic8_Q32_13x25_oddcolumns_screen7_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 32

def custom_ops(q):
    # Dic8 = Q32: x^16=1, y^2=x^8, y*x*y^-1=x^-1.
    if not valid_q(q):
        raise ValueError("Dic8 requires order 32")
    els = [(a, b) for b in range(2) for a in range(16)]

    def red(g):
        a, b = g
        return ((a + 8 * (b // 2)) % 16, b % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c) + 8*b*d) % 16,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % 16, 0) if b == 0 else ((a + 8) % 16, 1)

    def idx(g):
        a, b = red(g)
        return 16*b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


# Five canonical B incidence profiles; every column has three terms.
# Their top/bottom row totals are seven/eight. Every profile has both
# augmentation-column types, hence augmentation rank two.
_Q32_B_TOPS = (
    (1, 1, 1, 2, 2),
    (0, 1, 2, 2, 2),
    (0, 1, 1, 2, 3),
    (0, 0, 2, 2, 3),
    (0, 0, 1, 3, 3),
)


def _q32_copy(grid):
    return [[list(cell) for cell in row] for row in grid]


def _q32_key(grid):
    return tuple(tuple(sorted(cell)) for row in grid for cell in row)


def _q32_triple(rng, mode):
    """Four relative geometries, not four global automorphism images.

    Modes 0/1 have a rotation of order 16/8 and one y-coset term.
    Modes 2/3 have two y-coset terms whose relative rotation has
    order 16/8. The first term is a fixed local anchor.
    """
    steps = range(1, 16, 2) if mode % 2 == 0 else (2, 6, 10, 14)
    d, u = rng.choice(steps), rng.randrange(16)
    if mode < 2:
        return [(0, 0), (d, 0), (u, 1)]
    return [(0, 0), (u, 1), ((u + d) % 16, 1)]


def _q32_btriple(rng, G):
    """Independent mixed-coset voltages, excluding central separation."""
    while True:
        cell = [G.e] + rng.sample(
            [g for g in G.elements if g not in (G.e, (8, 0))], 2)
        if not any(g[1] for g in cell):
            continue
        if any(G.mul(G.inv(cell[i]), cell[j]) == (8, 0)
               for i in range(3) for j in range(i)):
            continue
        return cell


def _q32_draw(rng, G, side, mode, tops):
    if side == "A":
        return [[_q32_triple(rng, mode) for _ in range(3)]]
    grid = [[], []]
    for top in tops:
        cell = _q32_btriple(rng, G)
        grid[0].append(cell[:top])
        grid[1].append(cell[top:])
    return grid


def _q32_mutate(rng, G, grid, side, mode, tops):
    trial = _q32_copy(grid)
    width = 3 if side == "A" else 5
    # Occasional two-column moves escape single-column local minima.
    amount = 2 if rng.random() < 0.15 else 1
    for j in rng.sample(range(width), amount):
        if side == "A":
            trial[0][j] = _q32_triple(rng, mode)
        else:
            cell = _q32_btriple(rng, G)
            trial[0][j] = cell[:tops[j]]
            trial[1][j] = cell[tops[j]:]
    return trial


def _q32_columns(rows, width):
    columns = [0] * width
    for r, row in enumerate(rows):
        marker = 1 << r
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= marker
            row ^= bit
    return columns


def _q32_component(G, side, grid):
    """Extract both regular actions only from the verified LP helper."""
    q = G.q
    if side == "A":
        zero = [[[] for _ in range(5)] for _ in range(2)]
        hx, _, _ = lifted_matrices(G, grid, zero)
        mask = (1 << q) - 1
        rows = []
        for row in hx[:q]:
            rows.append(sum(
                ((row >> (j * 5 * q)) & mask) << (j * q)
                for j in range(3)))
        return _q32_columns(rows, 3 * q)
    zero = [[[] for _ in range(3)]]
    _, hz, _ = lifted_matrices(G, zero, grid)
    mask = (1 << (5 * q)) - 1
    return _q32_columns([row & mask for row in hz[:2*q]], 5*q)


def _q32_cost(columns):
    """Four-cycle cost and exact weight-four kernel collision count."""
    assert all(c.bit_count() == 3 for c in columns)
    if len(set(columns)) != len(columns):
        return (1000000, 1000000)

    syndromes = {}
    four = 0
    neighbors_seen = {}
    cycles = 0
    for i, column in enumerate(columns):
        for j in range(i):
            syndrome = column ^ columns[j]
            previous = syndromes.get(syndrome, 0)
            four += previous
            syndromes[syndrome] = previous + 1

        neighbors = []
        word = column
        while word:
            bit = word & -word
            neighbors.append(bit.bit_length() - 1)
            word ^= bit
        for a in range(3):
            for b in range(a):
                pair = (neighbors[b], neighbors[a])
                previous = neighbors_seen.get(pair, 0)
                cycles += previous
                neighbors_seen[pair] = previous + 1
    # Distinct syndromes imply colliding pairs are disjoint.
    # Each weight-four word has three pair partitions.
    return (four // 3, cycles)


def _q32_floor7(columns):
    """Exact component kernel exclusion through seven.

    Every column has odd weight, so odd-weight kernel words cannot
    occur. Once weights two and four are excluded, equal distinct
    triple syndromes must be disjoint and witness weight six.
    Eight means exclusion through seven, not an LP-distance claim.
    """
    assert all(c.bit_count() & 1 for c in columns)
    if len(set(columns)) != len(columns):
        return 2

    pairs = set()
    for i, column in enumerate(columns):
        for j in range(i):
            syndrome = column ^ columns[j]
            if syndrome in pairs:
                return 4
            pairs.add(syndrome)
    del pairs

    triples = set()
    for i, column in enumerate(columns):
        for j in range(i):
            pair = column ^ columns[j]
            for h in range(j):
                syndrome = pair ^ columns[h]
                if syndrome in triples:
                    return 6
                triples.add(syndrome)
    return 8


def _q32_kernel_reading(rng, columns, trials, floor):
    """Actual component-kernel witnesses; high readings are not proofs."""
    best = len(columns) + 1
    order = list(range(len(columns)))
    for _ in range(trials):
        rng.shuffle(order)
        pivots, words = {}, []
        for c in order:
            syndrome, word = columns[c], 1 << c
            while syndrome:
                p = syndrome.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                words.append(word)
                best = min(best, word.bit_count())

        for i, word in enumerate(words):
            for other in words[:i]:
                trial = word ^ other
                if trial:
                    best = min(best, trial.bit_count())

        small = sorted(words, key=int.bit_count)[:12]
        for i, word in enumerate(small):
            for j in range(i):
                pair = word ^ small[j]
                for h in range(j):
                    trial = pair ^ small[h]
                    if trial:
                        best = min(best, trial.bit_count())

        # Greedy XOR descent preserves the exact kernel syndrome.
        for start in small[:4]:
            word = start
            for _ in range(4):
                improved, weight = word, word.bit_count()
                for other in words:
                    trial = word ^ other
                    if trial and trial.bit_count() < weight:
                        improved, weight = trial, trial.bit_count()
                if improved == word:
                    break
                word = improved
                best = min(best, weight)
        if best <= floor:
            break
    return best


def _q32_choose_component(rng, G, side, mode, tops):
    """Independent annealing restarts, followed by exact short screening."""
    import math
    pool = []
    seen = set()
    expected_rank = G.q if side == "A" else 2 * G.q

    for restart in range(10):
        grid = _q32_draw(rng, G, side, mode, tops)
        columns = _q32_component(G, side, grid)
        cost = _q32_cost(columns)
        local = (cost, _q32_copy(grid), columns)

        for step in range(48):
            trial = _q32_mutate(rng, G, grid, side, mode, tops)
            tc = _q32_component(G, side, trial)
            proposed = _q32_cost(tc)
            energy = (40 * cost[0] + cost[1]) / float(G.q)
            next_energy = (40 * proposed[0] + proposed[1]) / float(G.q)
            temperature = 1.8 * (1 - step / 48.0) + 0.08
            delta = next_energy - energy
            if delta <= 0 or rng.random() < math.exp(-delta / temperature):
                grid, columns, cost = trial, tc, proposed
                if cost < local[0]:
                    local = (cost, _q32_copy(grid), list(columns))

        key = _q32_key(local[1])
        if key not in seen:
            seen.add(key)
            pool.append(local)

    pool.sort(key=lambda item: item[0])
    screened = []
    for cost, grid, columns in pool:
        if gf2_rank(columns) != expected_rank:
            continue
        floor = _q32_floor7(columns)
        if floor <= 4:
            continue
        reading = _q32_kernel_reading(rng, columns, 10, floor)
        if reading > len(columns):
            continue
        screened.append({
            "grid": grid, "columns": columns, "floor": floor,
            "reading": reading, "cost": cost,
        })

    def quality(item):
        return (item["floor"], min(item["reading"], 24),
                -item["cost"][1], -item["cost"][0])

    screened.sort(key=quality, reverse=True)
    if not screened:
        raise ValueError("no full-rank component passed weight-four screening")

    finalists = screened[:4]
    for item in finalists:
        item["reading"] = min(
            item["reading"],
            _q32_kernel_reading(rng, item["columns"], 32, item["floor"]))
    finalists.sort(key=quality, reverse=True)
    return finalists


def _q32_basis(rows):
    basis = {}
    for word in rows:
        while word:
            p = word.bit_length() - 1
            old = basis.get(p)
            if old is None:
                basis[p] = word
                break
            word ^= old
    return basis


def _q32_outside(word, basis):
    while word:
        p = word.bit_length() - 1
        old = basis.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _q32_small_logical4(checks, stabilizers, width):
    """Exact assembled logical detection through weight four.

    One representative per syndrome suffices: any discarded
    representative differs from it by a stabilizer, or has already
    exposed a logical in the same weight range.
    """
    basis = _q32_basis(stabilizers)
    columns = _q32_columns(checks, width)
    bits = [1 << i for i in range(width)]
    singles = {}
    for i, syndrome in enumerate(columns):
        word = bits[i]
        if syndrome == 0 and _q32_outside(word, basis):
            return True
        previous = singles.get(syndrome)
        if previous is not None:
            if _q32_outside(word ^ previous, basis):
                return True
        else:
            singles[syndrome] = word

    pairs = {}
    for i, column in enumerate(columns):
        for j in range(i):
            syndrome = column ^ columns[j]
            word = bits[i] | bits[j]
            if syndrome == 0 and _q32_outside(word, basis):
                return True
            previous = singles.get(syndrome)
            if previous is not None and _q32_outside(word ^ previous, basis):
                return True
            previous = pairs.get(syndrome)
            if previous is not None:
                if _q32_outside(word ^ previous, basis):
                    return True
            else:
                pairs[syndrome] = word
    return False


def NEW_CTX(rng, G, idx):
    """Twenty architectures; optimize voltages, not relabeling orbits."""
    recipe = idx % 20
    mode = recipe % 4
    tops = _Q32_B_TOPS[recipe // 4]
    sa = [[3, 3, 3]]
    sb = [list(tops), [3 - t for t in tops]]
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)
    assert sum(tops) == 7
    assert all(0 <= t <= 3 for t in tops)

    acandidates = _q32_choose_component(rng, G, "A", mode, tops)
    bcandidates = _q32_choose_component(rng, G, "B", mode, tops)
    pairs = []
    for ai, a in enumerate(acandidates):
        for bi, b in enumerate(bcandidates):
            quality = (
                min(a["floor"], b["floor"]),
                min(a["reading"], b["reading"], 24),
                min(a["reading"], 24) + min(b["reading"], 24),
                -a["cost"][1] - b["cost"][1],
            )
            pairs.append((quality, ai, bi))
    pairs.sort(reverse=True)

    kb = G.q * (A_SHAPE[1] - A_SHAPE[0]) * (
        B_SHAPE[1] - B_SHAPE[0])
    for quality, ai, bi in pairs:
        A = acandidates[ai]["grid"]
        B = bcandidates[bi]["grid"]
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k != kb or not (325*n <= 1000*k < 375*n):
            continue
        if _q32_small_logical4(hz, hx, n):
            continue
        if _q32_small_logical4(hx, hz, n):
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified quaternion lift failed CSS check")
        assert n == 544 and k == 192
        assert max(row.bit_count() for row in hx + hz) <= W_MAX
        return {"idx": idx, "cells": {"A": A, "B": B},
                "component_screen": quality}

    raise ValueError("all component pairs failed assembled rank/logical screening")


def CELL(rng, G, side, i, j, ctx):
    """Return the complete selected cell without further random labeling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled supports with uniform labels do not reliably eliminate weight-5 sector-1 logicals on Z17 (2,5)x(2,6): all five measured parent slots have d30s=5.
# REFUTED: Cyclic difference packing and component screening through six remove the sector-1 bottleneck: the current five measured leaders still have weight-eight minima exclusively in sector 1, with k-kb=0.
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
