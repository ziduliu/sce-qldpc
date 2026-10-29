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
CONCEPT = ("For Frobenius C19 semidirect C3, order 57, shapes (1,3)x(1,3), noncommuting trinomials without component four-cycles and with reduced short-boundary exposure should improve worse-basis circuit memory over the order-58 dihedral parent; measured pL decides this hypothesis.")
GROUP = ("Jump to the untried C19:C3 group with action x->x^7; 7^3=1 modulo 19. Unlike the dihedral parent it has no involutions. Related order-39 Frobenius beds reached about 7.1 at lower rates, motivating this experiment without establishing a high-rate advantage. The fixed order gives n=570.")
SHAPES = ("Keep equal (1,3) sides for cell 0.400: each component has 171 variables and 57 checks, n=570 and guaranteed k>=228. Verified complete-code ranks require k=228. Parent weight-eight witnesses and collective failures span sector-1 Acol/Bcol blocks; Acol/Brow check enrichment is near 1.0, so screen both components rather than reinforce an unsupported hotspot.")
FILLING = ("Twenty canonical pairs of distinct rotation-step geometries prescribe independent voltage searches. All six cells are noncommuting trinomials with order-19 rotation and order-three complementary differences. Verified component lifts require full rank, zero Tanner four-cycles and no kernel of weight <=4. Shortlisted components are compared by anchored weight-six collision exposure, small triple boundaries and approximate check-graph spectral radius. Final verified lifts require k=228, CSS and no quantum logical of weight <=4. Both row-weight sums are 12 and six-round CNOT count is 24624. This targets the parent's 125/1750 Z failures in slot 0 and widespread logical flips; none of these screens is circuit fitness.")
FAMILY_NAME = "r040_Frob57_13x13_rotation_geometry_boundary_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 57

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C19 semidirect C3 requires q=57")
    # x^19=y^3=1, y*x*y^-1=x^7; 7^3=1 modulo 19.
    els = [(a, b) for b in range(3) for a in range(19)]

    def red(g):
        return (int(g[0]) % 19, int(g[1]) % 3)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + pow(7, b, 19) * c) % 19, (b + d) % 3)

    def inv(g):
        a, b = red(g)
        bi = (-b) % 3
        return ((-pow(7, bi, 19) * a) % 19, bi)

    def idx(g):
        a, b = red(g)
        return a + 19 * b

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


def _f_columns(rows, width):
    out = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            out[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return out


def _f_low4(rows, stabilizers, width):
    """Exact existence of a nontrivial kernel word of weight <=4."""
    pivots = {}
    for row in stabilizers:
        while row:
            p = row.bit_length() - 1
            if p in pivots:
                row ^= pivots[p]
            else:
                pivots[p] = row
                break

    def nontrivial(word):
        while word:
            p = word.bit_length() - 1
            if p not in pivots:
                return True
            word ^= pivots[p]
        return False

    cols = _f_columns(rows, width)
    seen = {0: 0}
    for i, syndrome in enumerate(cols):
        word = 1 << i
        if syndrome in seen:
            if nontrivial(word ^ seen[syndrome]):
                return True
        else:
            seen[syndrome] = word
    for i, x in enumerate(cols):
        for j in range(i):
            syndrome = x ^ cols[j]
            word = (1 << i) | (1 << j)
            if syndrome in seen:
                if nontrivial(word ^ seen[syndrome]):
                    return True
            else:
                seen[syndrome] = word
    return False


def _f_graph(cols, nr):
    """Exact four/six-cycle statistics when four-cycles are absent."""
    adjacency = [0] * nr
    multiplicity = {}
    for column in cols:
        neighbors = []
        while column:
            bit = column & -column
            neighbors.append(bit.bit_length() - 1)
            column ^= bit
        assert len(neighbors) == 3
        for a in range(3):
            for b in range(a):
                i, j = sorted((neighbors[a], neighbors[b]))
                multiplicity[i, j] = multiplicity.get((i, j), 0) + 1
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i
    four = sum(v * (v - 1) // 2 for v in multiplicity.values())
    if four:
        return four, None, adjacency
    triangles = 0
    for i in range(nr):
        higher = adjacency[i] & ~((1 << (i + 1)) - 1)
        while higher:
            bit = higher & -higher
            j = bit.bit_length() - 1
            triangles += (adjacency[i] & adjacency[j]
                          & ~((1 << (j + 1)) - 1)).bit_count()
            higher ^= bit
    return 0, triangles - len(cols), adjacency


def _f_radius(adjacency):
    """Power-iteration expansion heuristic, not a certified bound."""
    from math import sin, sqrt
    nr = len(adjacency)
    neighbors = []
    for word in adjacency:
        row = []
        while word:
            bit = word & -word
            row.append(bit.bit_length() - 1)
            word ^= bit
        neighbors.append(row)
    assert len({len(row) for row in neighbors}) == 1
    result = 0.0
    for seed in (1, 3, 7):
        vector = [sin((i + 1) * seed) for i in range(nr)]
        mean = sum(vector) / nr
        vector = [v - mean for v in vector]
        norm = sqrt(sum(v * v for v in vector))
        vector = [v / norm for v in vector]
        for step in range(24):
            image = [sum(vector[j] for j in row) for row in neighbors]
            mean = sum(image) / nr
            image = [v - mean for v in image]
            norm = sqrt(sum(v * v for v in image))
            if norm < 1e-12:
                break
            vector = [v / norm for v in image]
        result = max(result, norm)
    return round(result, 8)


def _f_exposure(cols, q):
    """Anchored six-word collisions and small triple boundaries.

    All columns have odd weight three, excluding odd-weight kernels.
    After weight<=4 exclusion, distinct equal-syndrome triples are
    disjoint. Counts below are anchored exposure, not an exact spectrum.
    """
    size = len(cols)
    anchors = tuple(range(0, size, q))
    aset = set(anchors)
    anchored = {}
    boundary = 0
    for a in anchors:
        for b in range(size):
            if b == a or (b in aset and b < a):
                continue
            ab = cols[a] ^ cols[b]
            for c in range(b + 1, size):
                if c == a or (c in aset and c < a):
                    continue
                syndrome = ab ^ cols[c]
                anchored[syndrome] = anchored.get(syndrome, 0) + 1
                weight = syndrome.bit_count()
                if weight <= 3:
                    boundary += (4 - weight) ** 2
    six = -sum(anchored.values())
    get = anchored.get
    for a in range(size - 2):
        x = cols[a]
        for b in range(a + 1, size - 1):
            ab = x ^ cols[b]
            for c in range(b + 1, size):
                six += get(ab ^ cols[c], 0)
    assert six >= 0
    return six, boundary


def NEW_CTX(rng, G, idx):
    """Optimize twenty geometry contracts; all search state is local."""
    from itertools import combinations, combinations_with_replacement

    q = G.q
    width = 3 * q
    mask = (1 << q) - 1
    supports = [[3, 3, 3]]
    assert _grid_weights(supports, supports) == 12
    assert _no_dead_lines(supports, supports)

    # Undirected rotation-step triples modulo multipliers and columns.
    # This removes obvious group-automorphism copies, not arbitrary
    # Tanner isomorphisms; the evaluator remains authoritative.
    representatives = set()
    for steps in combinations(range(1, 10), 3):
        images = [
            tuple(sorted(min((u * d) % 19, (-u * d) % 19)
                         for d in steps))
            for u in range(1, 19)
        ]
        representatives.add(min(images))
    book = sorted(representatives)
    contracts = list(combinations_with_replacement(range(len(book)), 2))
    assert len(contracts) >= 20
    design, batch = idx % 20, idx // 20
    ia, ib = contracts[design * len(contracts) // 20]
    cosets = ((1, 1, 1), (1, 1, 2), (1, 2, 2), (2, 1, 2))
    ac = cosets[(design // 5 + batch) % len(cosets)]
    bc = cosets[(design % 5 + 2 * batch) % len(cosets)]

    # Cache only singleton actions extracted from verified lifts.
    # Composing them is exact linearity, not a hand-written lift.
    actions = {"A": {}, "B": {}}
    unit = [[[G.e]]]
    for g in G.elements:
        hx, _, _ = lifted_matrices(G, [[[g]]], unit)
        _, hz, _ = lifted_matrices(G, unit, [[[g]]])
        actions["A"][G.idx(g)] = tuple(row & mask for row in hx)
        actions["B"][G.idx(g)] = tuple(row & mask for row in hz)

    def component(grid, side):
        rows = [0] * q
        for j, cell in enumerate(grid[0]):
            for g in cell:
                term = actions[side][G.idx(g)]
                for r, word in enumerate(term):
                    rows[r] ^= word << (j * q)
        return rows

    def select(side, geometry, quotient):
        parameters = [(a, b) for a in range(19) for b in range(19)]
        rng.shuffle(parameters)
        pool = []
        for trial, (a, b) in enumerate(parameters):
            offsets = (0, a, b)
            grid = [[
                [G.e, (d, 0), (t, h)]
                for d, t, h in zip(geometry, offsets, quotient)
            ]]
            # Cheap algebraic prefilter. Actual cycle checks below use
            # the component composed from verified singleton lifts.
            differences = set()
            repeated = False
            for cell in grid[0]:
                for g in cell:
                    for h in cell:
                        if g == h:
                            continue
                        value = G.idx(G.mul(g, G.inv(h)))
                        if value in differences:
                            repeated = True
                        differences.add(value)
            if repeated:
                continue
            rows = component(grid, side)
            if gf2_rank(rows) != q:
                continue
            cols = _f_columns(rows, width)
            four, six, adjacency = _f_graph(cols, q)
            if four:
                continue
            pool.append((six, trial, grid, rows, cols, adjacency))
            pool.sort(key=lambda entry: entry[:2])
            del pool[12:]
        if not pool:
            raise ValueError("no full-rank four-cycle-free component")

        refined = []
        for cycles, trial, grid, rows, cols, adjacency in pool:
            if _f_low4(rows, [], width):
                continue
            six, boundary = _f_exposure(cols, q)
            radius = _f_radius(adjacency)
            score = (six, boundary, radius, cycles)
            refined.append((score, trial, grid, rows))
            if len(refined) == 6:
                break
        refined.sort(key=lambda entry: entry[:2])
        if not refined:
            raise ValueError("component short-kernel screening failed")
        return refined[:3]

    aa = select("A", book[ia], ac)
    bb = select("B", book[ib], bc)
    options = []
    for ka, ta, A, ar in aa:
        for kb, tb, B, br in bb:
            joint = tuple(v for a, b in zip(ka, kb)
                          for v in (max(a, b), a + b))
            options.append((joint, ta, tb, A, B, ar, br))
    options.sort(key=lambda entry: entry[:3])

    for score, _, _, A, B, ar, br in options:
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 570 or k != 228:
            continue
        # Confirm that the screened components are precisely those
        # present in the final evaluator-convention matrices.
        actual_a = [
            sum(((row >> (3 * j * q)) & mask) << (j * q)
                for j in range(3))
            for row in hx[:q]
        ]
        actual_b = [row & ((1 << width) - 1) for row in hz[:q]]
        assert actual_a == ar and actual_b == br
        if _f_low4(hx, hz, n) or _f_low4(hz, hx, n):
            continue
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx + hz) == 12
        assert 6 * sum(row.bit_count() for row in hx + hz) == 24624
        return {"idx": idx, "cells": {"A": A, "B": B}, "k": k,
                "contract": (book[ia], book[ib], ac, bc),
                "component_screen": score}
    raise ValueError("no complete target-rate filling passed screening")


def CELL(rng, G, side, i, j, ctx):
    """Return the already screened complete cell, without resampling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Screen d2s=4 alone implies reliable circuit memory; the parent's measured slots still had worst-basis failures of 120-144/250 shots, with broad sector-1 weight-four witnesses.
# REFUTED: Balanced dihedral trinomials with d2s=8 suffice for rare circuit failures; all five measured parent codes reached failed-shot caps, including slot 0 Z=125/1750, with flips spanning all 232 logical qubits.
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
