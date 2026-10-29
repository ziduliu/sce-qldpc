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
CONCEPT = ("Dicyclic Dic5, order 20, shapes (2,5)x(2,5): excluding central-involution cell ratios and reducing small-boundary component triples should improve worse-basis circuit fitness beyond the cyclic parent's 5.5627.")
GROUP = ("Dic5 has presentation x^10=1, y^2=x^5, y*x*y^-1=x^-1. It is nonabelian, unlike parent C20, with centre of order two. Successful centre-two order-28 and order-44 beds motivate this untried order-20 dicyclic bed without isolating a group effect. Noncentral rotation ratios have orders five or ten; outside-coset ratios have order four. Order 20 is the largest lift allowed at these shapes.")
SHAPES = ("Keep equal (2,5) components, each with 40 checks and 100 variables: n=29*20=580 and kb=180 give guaranteed rate 9/29=0.310345 in target cell 0.300. Parent failures span many sector-1 Acol/Bcol blocks with mostly 1.1x enrichment, so there is no established isolated Acol/Brow hotspot. Preserve both component dimensions and test protection on both bases.")
FILLING = ("Twenty prescribed support pairs retain column sums 3-4 and row sums 7-8, hence both check-weight sums <=12. Replace two augmentation-deficient profiles with distinct full-augmentation-rank profiles. Cells exclude duplicates and central-involution ratios; additional harness slots use homogeneous, mixed, or alternating coset policies rather than orbit copies. Optimize verified component lifts for rank, connectivity, short dependencies, cycles and small-boundary triples. Require verified CSS and rate in [0.275,0.325), not exactly k=180. Parent equal-(n,k,d2s) measured slots scored 5.402-5.782 and all hit failure caps; these screens target collective decoder failures, not distance-only fitness.")
FAMILY_NAME = "r030_dic5_25x25_longratio_boundary_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 20

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Dic5 requires q=20")
    # (a,b) = x^a y^b; x^10=1, y^2=x^5, y*x*y^-1=x^-1.
    els = [(a, b) for b in range(2) for a in range(10)]

    def red(g):
        a, b = int(g[0]), int(g[1])
        return ((a + 5 * (b // 2)) % 10, b % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c) + 5 * b * d) % 10,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % 10, 0) if b == 0 else ((a + 5) % 10, 1)

    def idx(g):
        a, b = red(g)
        return a + 10 * b

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 5)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


# Columns are prescribed term-count pairs, not random support draws.
# Every profile has five columns of weight 3 or 4 and row sums <=8.
_SUPPORT_PROFILES = (
    ((2,1),(2,1),(1,2),(1,2),(1,2)),
    ((2,1),(2,1),(1,2),(1,2),(2,2)),
    ((2,1),(1,2),(1,2),(1,2),(3,1)),
    ((3,0),(2,1),(1,2),(1,2),(1,2)),
    ((3,0),(2,1),(2,1),(0,3),(0,3)),
    ((3,0),(0,3),(2,1),(1,2),(1,2)),
    ((3,0),(0,3),(2,1),(1,2),(2,2)),
    ((3,0),(0,3),(0,3),(3,1),(2,1)),
)
_PROFILE_PAIRS = (
    (0,0),(0,1),(0,2),(1,1),(1,2),(2,2),
    (0,3),(0,4),(1,3),(1,4),(2,3),(2,4),
    (3,3),(3,4),(4,4),(0,5),(1,5),(2,5),
    (0,6),(0,7),
)


def _support_profile(rng, number):
    columns = list(_SUPPORT_PROFILES[number])
    rng.shuffle(columns)
    swap = rng.randrange(2)
    return [[column[i ^ swap] for column in columns] for i in range(2)]


def _component_rows(G, side, grid):
    """Extract the component action from the verified construction.

    The opposite side is a zero 1x1 probe, not a submitted component.
    This preserves the evaluator's left/right conventions exactly.
    """
    if side == "A":
        rows = lifted_matrices(G, grid, [[[]]])[0]
    else:
        rows = lifted_matrices(G, [[[]]], grid)[1]
    mask = (1 << (len(grid[0]) * G.q)) - 1
    return [row & mask for row in rows]


def _cycle_counts(rows):
    """Exact component Tanner four- and six-cycle counts."""
    size = len(rows)
    intersections = [[0] * size for _ in range(size)]
    counts = [[0] * size for _ in range(size)]
    four = 0
    for i in range(size):
        for j in range(i + 1, size):
            common = rows[i] & rows[j]
            number = common.bit_count()
            intersections[i][j] = common
            counts[i][j] = number
            four += number * (number - 1) // 2
    six = 0
    for i in range(size):
        for j in range(i + 1, size):
            a = counts[i][j]
            if not a:
                continue
            for k in range(j + 1, size):
                b, c = counts[i][k], counts[j][k]
                if b and c:
                    common = (intersections[i][j] & rows[k]).bit_count()
                    # Count distinct variable choices, excluding stars.
                    six += a*b*c - common*(a+b+c) + 2*common
    return four, six


def _short_collision_counts(rows, columns):
    """Cheap component-codeword screen through weight four.

    Pair collisions with overlapping supports can overcount when
    weight-one/two words exist; those are already penalized first.
    """
    syndromes = [0] * columns
    for check, row in enumerate(rows):
        while row:
            bit = row & -row
            syndromes[bit.bit_length() - 1] |= 1 << check
            row ^= bit
    singles = {}
    one = two = 0
    for syndrome in syndromes:
        one += int(syndrome == 0)
        two += singles.get(syndrome, 0)
        singles[syndrome] = singles.get(syndrome, 0) + 1
    pairs = {}
    three = four = 0
    for i in range(columns):
        for j in range(i + 1, columns):
            syndrome = syndromes[i] ^ syndromes[j]
            three += singles.get(syndrome, 0)
            four += pairs.get(syndrome, 0)
            pairs[syndrome] = pairs.get(syndrome, 0) + 1
    return one, two, three, four


def _copy_cells(grid):
    return [[list(cell) for cell in row] for row in grid]


def _dic_cell(rng, G, weight, base_coset, mixed):
    """Distinct terms, with no relative central involution x^5."""
    if weight == 0:
        return []
    cosets = [base_coset] * weight
    if mixed and weight >= 2:
        cosets[-1] ^= 1
    cell = []
    central = (5, 0)
    for parity in cosets:
        choices = [
            g for g in G.elements
            if g[1] == parity
            and all(G._key(g) != G._key(h)
                    and G._key(G.mul(G.inv(h), g)) != central
                    for h in cell)
        ]
        # Each coset has ten elements; at most four are excluded here.
        if not choices:
            raise ValueError("dicyclic cell model exhausted")
        cell.append(rng.choice(choices))
    return cell


def _decoder_risk(rows, width):
    """Verified component connectivity and triple-boundary heuristics.

    Triple/pair and triple/triple collisions expose short dependencies.
    They are not quantum distance certificates or measured decoder tests.
    """
    columns = [0] * width
    for r, row in enumerate(rows):
        while row:
            low = row & -row
            columns[low.bit_length() - 1] |= 1 << r
            row ^= low

    parent = list(range(len(rows)))

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for col in columns:
        if not col:
            continue
        first = (col & -col).bit_length() - 1
        remaining = col & (col - 1)
        while remaining:
            low = remaining & -remaining
            r = low.bit_length() - 1
            parent[root(r)] = root(first)
            remaining ^= low
    components = len({root(r) for r in range(len(rows))})

    pairs = {
        columns[i] ^ columns[j]
        for i in range(width) for j in range(i)
    }
    triples = {}
    boundary = five = six = 0
    for i in range(width - 2):
        for j in range(i + 1, width - 1):
            pair = columns[i] ^ columns[j]
            for k in range(j + 1, width):
                syndrome = pair ^ columns[k]
                weight = syndrome.bit_count()
                if weight < 4:
                    boundary += (4 - weight) ** 2
                five += int(syndrome in pairs)
                old = triples.get(syndrome, 0)
                six += old
                triples[syndrome] = old + 1
    return components, boundary, five, six


def _design_component(rng, G, side, support, idx):
    """Independent constrained searches, screened on verified lifts."""
    import math

    nr, nc = len(support), len(support[0])
    positions = [(i, j) for i in range(nr) for j in range(nc)
                 if support[i][j]]
    overlap = [j for j in range(nc)
               if support[0][j] and support[1][j]]
    assert len(overlap) >= 2

    # A varying relative coset on shared columns avoids a balanced
    # index-two voltage pattern in the homogeneous-cell architecture.
    offset = int(side == "B")
    parity = [(j + idx + offset) % 2 for j in range(nc)]
    for t, j in enumerate(overlap):
        parity[j] = (t + idx + offset) % 2

    # Extra slots change the actual element architecture:
    # long rotation ratios; mixed order-four ratios; or a mixture.
    mode = (idx // len(_PROFILE_PAIRS)) % 3

    def draw_cell(i, j):
        weight = support[i][j]
        base = parity[j] if i == 0 else 0
        mixed = (mode == 1 or
                 (mode == 2 and (i + j + idx) % 2 == 0))
        return _dic_cell(rng, G, weight, base, mixed)

    augmentation = [
        sum((weight % 2) << j for j, weight in enumerate(row))
        for row in support
    ]
    unavoidable_deficit = nr - gf2_rank(augmentation)
    selected = None
    selected_quality = None
    cycle_weight = 200

    for restart in range(4):
        grid = [[draw_cell(i, j) for j in range(nc)]
                for i in range(nr)]
        rows = _component_rows(G, side, grid)
        cycles = _cycle_counts(rows)
        # Normalize orbit-sized changes so annealing can escape traps.
        score = (cycle_weight * cycles[0] + cycles[1]) / G.q
        local_best = _copy_cells(grid)
        local_score = score

        for step in range(220):
            i, j = rng.choice(positions)
            old = grid[i][j]
            grid[i][j] = draw_cell(i, j)
            trial_rows = _component_rows(G, side, grid)
            trial_cycles = _cycle_counts(trial_rows)
            trial_score = (cycle_weight * trial_cycles[0]
                           + trial_cycles[1]) / G.q
            temperature = 35.0 * (1.0 - step / 220.0) + 0.3
            delta = trial_score - score
            if delta <= 0 or rng.random() < math.exp(
                    -min(700.0, delta / temperature)):
                score = trial_score
                if score < local_score:
                    local_score = score
                    local_best = _copy_cells(grid)
            else:
                grid[i][j] = old

        rows = _component_rows(G, side, local_best)
        deficit = len(rows) - gf2_rank(rows)
        short = _short_collision_counts(rows, nc * G.q)
        four, six_cycles = _cycle_counts(rows)
        components, boundary, five, six = _decoder_risk(
            rows, nc * G.q)
        # Favor maximal attainable rank and connected protection, then
        # remove small dependencies before comparing trapping heuristics.
        quality = (
            max(0, deficit - unavoidable_deficit),
            components - 1,
        ) + short + (
            four, boundary, five, six, six_cycles, deficit,
        )
        if selected_quality is None or quality < selected_quality:
            selected_quality = quality
            selected = local_best
    return selected


def NEW_CTX(rng, G, idx):
    """Build a deliberate architecture and verify its measured rate."""
    pa, pb = _PROFILE_PAIRS[idx % len(_PROFILE_PAIRS)]
    sa = _support_profile(rng, pa)
    sb = _support_profile(rng, pb)
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert _no_dead_lines(sa, sb)
    assert min(sum(row[j] for row in sa) for j in range(5)) >= 3
    assert min(sum(row[j] for row in sb) for j in range(5)) >= 3

    for attempt in range(4):
        A = _design_component(rng, G, "A", sa, idx)
        B = _design_component(rng, G, "B", sb, idx)
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        # Permit small genuine rank deficits, but keep every submitted
        # code inside the requested cell rather than relying on its mean.
        if k > 0 and 11 * n <= 40 * k < 13 * n:
            assert n == 580
            assert css_ok(hx, hz)
            assert max(row.bit_count() for row in hx + hz) <= min(12, W_MAX)
            return {"idx": idx, "k": k,
                    "cells": {"A": A, "B": B}}
    raise ValueError("no optimized dicyclic filling in cell 0.300")


def CELL(rng, G, side, i, j, ctx):
    """Return the already-screened complete cell, without mutation."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: A d2s=6 screen does not ensure balanced circuit protection: parent slot 56 had 107/1250 Z failures but only 53/200000 X failures.
# REFUTED: Equal (n,k,d2s) fixes circuit reliability: current cyclic slots with (580,180,8) scored 5.402-5.782, with worst-basis failure caps reached after 28750-68250 shots.
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
