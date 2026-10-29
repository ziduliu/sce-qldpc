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
CONCEPT = ("AGL(1,5), order 20, shapes (2,5)x(2,5): excluding involutory cell ratios and reducing five-/six-term component dependencies should improve worse-basis circuit reliability over the cyclic parent's 5.5627 fitness.")
GROUP = ("Invertible affine maps over F5 form a nonabelian group with trivial centre, genuinely different from C20. The run's AGL(1,5) (2,4)x(2,6) bed scored 4.8286, which does not prove a group advantage; its symmetric (2,5)x(2,5) bed is untried. Order 20 retains n=580.")
SHAPES = ("Keep equal (2,5) components: each has 40 checks and 100 variables, and guaranteed k/n=9/29=0.310345 targets cell 0.300. The parent's failed shots spread across many Acol/Bcol and Acol/Brow blocks, mostly at 1.1x enrichment, so reinforce both components rather than infer a unique hotspot.")
FILLING = ("Twenty prescribed support pairs retain column sums 3-4, row sums <=8 and both CSS weight sums <=12. Doubletons have relative order four or five; trinomials have noncommuting relative elements. Optimize verified lifts for cycles, then refine by exact five-/six-term dependencies and sampled short-kernel multiplicity. Full CSS screening enforces the target rate and excludes logicals of weight <=4. Parent slot 0 reached 100 X failures in 68250 shots with 4896 logical flips; measured slots scored only 5.40-5.78 despite d2s=8. The term-count budget is unchanged; circuit pL remains the test.")
FAMILY_NAME = "r030_agl1_f5_25x25_longratio_dependency_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 20

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("AGL(1,5) requires q=20")
    # (u,v) acts as z -> u*z+v over F5; multiplication is composition.
    els = [(u, v) for u in range(1, 5) for v in range(5)]

    def red(g):
        u, v = int(g[0]) % 5, int(g[1]) % 5
        if u == 0:
            raise ValueError("affine slope must be nonzero")
        return (u, v)

    def mul(g, h):
        return ((g[0] * h[0]) % 5,
                (g[1] + g[0] * h[1]) % 5)

    def inv(g):
        ui = pow(g[0], 3, 5)
        return (ui, (-ui * g[1]) % 5)

    def idx(g):
        u, v = red(g)
        return 5 * (u - 1) + v

    return dict(elements=els, ident=(1, 0), mul=mul, inv=inv,
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
    ((3,0),(1,2),(1,2),(1,2),(1,2)),
    ((3,0),(0,3),(2,1),(1,2),(1,2)),
    ((3,0),(0,3),(2,1),(1,2),(2,2)),
    ((3,0),(1,2),(1,2),(1,2),(2,2)),
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


def _affine_long_cell(rng, G, count, policy, i, j):
    """Order-4/5 ratios; noncommuting relative elements in trinomials."""
    assert 0 <= count <= 3
    if count == 0:
        return []
    g = G.pick(rng)
    if count == 1:
        return [g]
    translations = [(1, v) for v in range(1, 5)]
    quartic = [(u, v) for u in (2, 3) for v in range(5)]
    if count == 2:
        if policy == 0:
            ratios = quartic
        elif policy == 1:
            ratios = translations
        elif policy == 2:
            ratios = translations if (i + j) % 2 else quartic
        else:
            ratios = translations + quartic
        u = rng.choice(ratios)
        return [g, G.mul(g, u)]

    # A nonidentity translation and a quartic element never commute.
    # Their relative ratio also has slope 2 or 3, hence order four.
    u, v = rng.choice(translations), rng.choice(quartic)
    assert G._key(G.mul(u, v)) != G._key(G.mul(v, u))
    return [g, G.mul(g, u), G.mul(g, v)]


def _dependency_columns(rows, width):
    """Syndromes from the verified component action."""
    columns = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return columns


def _five_six_dependencies(columns):
    """Exact word counts, conditional on no component words of weight <=4.

    Pair/triple collisions then have disjoint supports and count each
    weight-five word ten times. Triple/triple collisions count each
    weight-six word ten times. These are not quantum-distance claims.
    """
    pairs = {}
    width = len(columns)
    for i in range(width):
        for j in range(i):
            syndrome = columns[i] ^ columns[j]
            pairs[syndrome] = pairs.get(syndrome, 0) + 1
    triples = {}
    five = six = 0
    for i in range(width):
        for j in range(i):
            pair = columns[i] ^ columns[j]
            for h in range(j):
                syndrome = pair ^ columns[h]
                five += pairs.get(syndrome, 0)
                previous = triples.get(syndrome, 0)
                six += previous
                triples[syndrome] = previous + 1
    assert five % 10 == 0 and six % 10 == 0
    return five // 10, six // 10


def _sample_short_kernels(rng, columns, trials=48):
    """Verified fundamental kernel circuits; upper bounds only.

    Count distinct sampled words through weight eight rather than
    reward arbitrarily large, optimistic distance estimates.
    """
    order = list(range(len(columns)))
    bound = len(columns) + 1
    small = set()
    for _ in range(trials):
        rng.shuffle(order)
        pivots = {}
        for c in order:
            syndrome, word = columns[c], 1 << c
            while syndrome:
                p = syndrome.bit_length() - 1
                previous = pivots.get(p)
                if previous is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= previous[0]
                word ^= previous[1]
            if not syndrome and word:
                weight = word.bit_count()
                bound = min(bound, weight)
                if weight <= 8:
                    small.add(word)
    return bound, len(small)


def _affine_small_logical(checks, stabilizers, n):
    """Exact existence test for non-stabilizer logicals of weight <=4."""
    columns = _dependency_columns(checks, n)
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

    seen = {0: 0}
    bits = [1 << i for i in range(n)]
    for i, syndrome in enumerate(columns):
        previous = seen.get(syndrome)
        if previous is None:
            seen[syndrome] = bits[i]
        elif nontrivial(bits[i] ^ previous):
            return True
    for i in range(n):
        for j in range(i):
            syndrome = columns[i] ^ columns[j]
            word = bits[i] | bits[j]
            previous = seen.get(syndrome)
            if previous is None:
                seen[syndrome] = word
            elif nontrivial(word ^ previous):
                return True
    return False


def _design_component(rng, G, side, support, idx):
    """Constrained cycle optimization followed by dependency refinement."""
    import math
    positions = [(i, j) for i in range(2) for j in range(5)
                 if support[i][j]]
    # Twenty support-pair briefs; additional slots alter ratio policy.
    policy = (idx % 4 + idx // len(_PROFILE_PAIRS)) % 4
    cycle_weight = 200
    proposals = []

    for restart in range(2):
        grid = [[_affine_long_cell(rng, G, support[i][j],
                                  policy, i, j)
                 for j in range(5)] for i in range(2)]
        cycles = _cycle_counts(_component_rows(G, side, grid))
        score = cycle_weight * cycles[0] + cycles[1]
        local_best, local_score = _copy_cells(grid), score

        for step in range(96):
            i, j = rng.choice(positions)
            old = grid[i][j]
            grid[i][j] = _affine_long_cell(
                rng, G, support[i][j], policy, i, j)
            trial_cycles = _cycle_counts(_component_rows(G, side, grid))
            trial_score = cycle_weight * trial_cycles[0] + trial_cycles[1]
            temperature = 35.0 * (1.0 - step / 96.0) + 0.3
            delta = trial_score - score
            if delta <= 0 or rng.random() < math.exp(-delta / temperature):
                score = trial_score
                if score < local_score:
                    local_score = score
                    local_best = _copy_cells(grid)
            else:
                grid[i][j] = old
            if step in (31, 63, 95):
                proposals.append(_copy_cells(grid))
        proposals.append(local_best)

    screened = []
    seen = set()
    width = 5 * G.q
    for grid in proposals:
        signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                          for row in grid for cell in row)
        if signature in seen:
            continue
        seen.add(signature)
        rows = _component_rows(G, side, grid)
        short = _short_collision_counts(rows, width)
        deficit = len(rows) - gf2_rank(rows)
        cycles = _cycle_counts(rows)
        basic = (int(any(short)), deficit, short, cycles)
        screened.append((basic, grid, rows))
    screened.sort(key=lambda item: item[0])

    selected = None
    selected_quality = None
    for basic, grid, rows in screened[:5]:
        columns = _dependency_columns(rows, width)
        short_bad, deficit, short, cycles = basic
        if short_bad:
            five, six = 10**12, 10**12
        else:
            five, six = _five_six_dependencies(columns)
        bound, small = _sample_short_kernels(rng, columns)
        quality = (
            short_bad, deficit, five, six,
            int(bound < 8), small, -min(bound, 10),
            cycles[0], cycles[1], short,
        )
        if selected_quality is None or quality < selected_quality:
            selected_quality, selected = quality, grid
    return selected


def NEW_CTX(rng, G, idx):
    """Build complete affine fillings and screen the actual CSS pair."""
    pa, pb = _PROFILE_PAIRS[idx % len(_PROFILE_PAIRS)]
    sa = _support_profile(rng, pa)
    sb = _support_profile(rng, pb)
    # HX: A-row + B-column; HZ: B-row + A-column.
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert _no_dead_lines(sa, sb)

    for attempt in range(3):
        A = _design_component(rng, G, "A", sa, idx)
        B = _design_component(rng, G, "B", sb, idx + 1)
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        # Permit small rank deficits without voiding a target-band code.
        if not (k > 0 and 275 * n <= 1000 * k < 325 * n):
            continue
        if _affine_small_logical(hx, hz, n):
            continue
        if _affine_small_logical(hz, hx, n):
            continue
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx + hz) <= min(12, W_MAX)
        return {"idx": idx, "k": k, "cells": {"A": A, "B": B}}
    raise ValueError("no affine filling passed target-band logical screening")


def CELL(rng, G, side, i, j, ctx):
    """Return the screened full cell without subsequent relabeling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: A d2s=6 screen does not ensure balanced circuit protection: parent slot 56 had 107/1250 Z failures but only 53/200000 X failures.
# REFUTED: Balanced degree-3/4 cyclic components plus cycle screening suffice for ceiling-level reliability: the current parent measured s=5.40-5.78, reached the 100-failure caps in both bases, and exhibited widespread collective logical flips despite d2s=8.
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
