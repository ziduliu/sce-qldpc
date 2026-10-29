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
CONCEPT = ("For the affine matrix group AGL(1,7), order 42, A=(1,3), B=(2,4), minimum component degree three, noncommuting cell supports and short-dependency screening should improve worse-basis circuit fitness over the 0.300 frontier's 5.5627.")
GROUP = ("Use invertible affine maps over F7, represented as upper-triangular 2x2 matrices with lower row (0,1). This is genuinely different from the parent's order-48 group. AGL(1,7) has trivial centre and seven conjugacy classes; the order-42 centre-one bed reached 7.2 at lower rate, motivating but not proving this higher-rate transfer.")
SHAPES = ("Target cell 0.300: A=(1,3), B=(2,4), q=42 give n=588, kb=168 and rate bound 2/7=0.285714. Components have 42 checks/126 variables and 84 checks/168 variables. Gate every submitted code to [0.275,0.325). Parent failures were spread across Acol/Bcol blocks, generally at 1.0-1.4x enrichment, so strengthen both components rather than assume an isolated hotspot.")
FILLING = ("Twenty briefs cross five B support profiles with four relative-order policies. A cells are trinomials; every B column has three terms. HX weight is 12, HZ weight <=11, and six-round CNOT count is 25704 versus parent slot 5's 27936. Independently optimize verified component lifts for four-cycles, connectivity, rank deficits and short kernel dependencies, then screen complete CSS matrices. Noncommuting trinomials and prescribed binomial orders test trap suppression against gate noise; measured pL remains the objective.")
FAMILY_NAME = "r030_affineF7_13x24_noncommuting_dependency_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 42

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("AGL(1,7) requires order 42")
    # (a,b) encodes [[a,b],[0,1]], acting as x -> a*x+b.
    els = [(a, b) for a in range(1, 7) for b in range(7)]

    def red(g):
        return (int(g[0]) % 7, int(g[1]) % 7)

    def mul(g, h):
        return ((g[0] * h[0]) % 7,
                (g[0] * h[1] + g[1]) % 7)

    def inv(g):
        ai = pow(g[0], 5, 7)
        return (ai, (-ai * g[1]) % 7)

    def idx(g):
        a, b = red(g)
        return 7 * (a - 1) + b

    return dict(elements=els, ident=(1, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb)


def _design_four_cycles(rows):
    """Count four-cycles within one basis, not across X and Z."""
    seen = {}
    cycles = 0
    for row in rows:
        cols = []
        while row:
            bit = row & -row
            cols.append(bit.bit_length() - 1)
            row ^= bit
        for a in range(len(cols)):
            for b in range(a):
                pair = (cols[b], cols[a])
                previous = seen.get(pair, 0)
                cycles += previous
                seen[pair] = previous + 1
    return cycles


def _design_small_logical(checks, stabilizers, n):
    """Exact existence test for a logical of weight 1..4.

    Equal syndromes of supports of size <=2 exhaust this range.
    Reduce their difference against the opposite stabilizer row space.
    Inputs are exclusively the scaffold's verified lifted matrices.
    """
    pivots = {}
    for row in stabilizers:
        while row:
            p = row.bit_length() - 1
            if p in pivots:
                row ^= pivots[p]
            else:
                pivots[p] = row
                break

    def trivial(vector):
        while vector:
            p = vector.bit_length() - 1
            if p not in pivots:
                return False
            vector ^= pivots[p]
        return True

    syndromes = [0] * n
    for r, row in enumerate(checks):
        syndrome_bit = 1 << r
        while row:
            bit = row & -row
            syndromes[bit.bit_length() - 1] ^= syndrome_bit
            row ^= bit

    bits = [1 << i for i in range(n)]
    seen = {0: 0}
    for i, syndrome in enumerate(syndromes):
        previous = seen.get(syndrome)
        if previous is None:
            seen[syndrome] = bits[i]
        elif not trivial(previous ^ bits[i]):
            return True

    for i in range(n):
        si, vi = syndromes[i], bits[i]
        for j in range(i):
            syndrome = si ^ syndromes[j]
            vector = vi ^ bits[j]
            previous = seen.get(syndrome)
            if previous is None:
                seen[syndrome] = vector
            elif not trivial(previous ^ vector):
                return True
    return False


# Each column totals three. Profiles differ in row balance and zeros,
# not merely in a permutation or automorphism of their labels.
_AFFINE_B_PROFILES = (
    ((2, 1), (2, 1), (1, 2), (1, 2)),
    ((2, 1), (1, 2), (1, 2), (1, 2)),
    ((1, 2), (1, 2), (1, 2), (1, 2)),
    ((3, 0), (1, 2), (1, 2), (1, 2)),
    ((3, 0), (0, 3), (2, 1), (1, 2)),
)


def _affine_tables(G):
    """Local algebra tables; no cross-slot mutable state."""
    mt = [[G.idx(G.mul(g, h)) for h in G.elements]
          for g in G.elements]
    iv = [G.idx(G.inv(g)) for g in G.elements]
    identity = G.idx(G.e)
    orders = []
    for a in range(G.q):
        h = identity
        for t in range(1, G.q + 1):
            h = mt[h][a]
            if h == identity:
                orders.append(t)
                break
        else:
            raise ValueError("invalid finite-group order")
    relative = [[orders[mt[iv[a]][b]] for b in range(G.q)]
                for a in range(G.q)]
    return mt, iv, relative


def _affine_cell(rng, G, count, mode, i, j, tables):
    """Prescribed relative orders, with noncommuting trinomials."""
    if count == 0:
        return []
    if count == 1:
        return [G.pick(rng)]
    mt, iv, relative = tables
    logarithm = {1: 0, 3: 1, 2: 2, 6: 3, 4: 4, 5: 5}
    for _ in range(512):
        ids = rng.sample(range(G.q), count)
        if count == 2:
            wanted = (7, 6, 7 if (i + j) % 2 else 6, 3)[mode]
            if relative[ids[0]][ids[1]] != wanted:
                continue
        else:
            if any(relative[ids[a]][ids[b]] < 3
                   for a in range(count) for b in range(a)):
                continue
            exponents = {logarithm[G.elements[a][0]] for a in ids}
            if mode == 0 or (mode == 2 and j == 0):
                if len(exponents) != 2:
                    continue
                x, y = sorted(exponents)
                if (y - x) % 6 not in (1, 5):
                    continue
            elif mode == 1:
                if not any(exponents == {t, (t + 1) % 6, (t + 2) % 6}
                           for t in range(6)):
                    continue
            elif mode == 2:
                if len(exponents) != 2:
                    continue
                x, y = sorted(exponents)
                if (y - x) % 6 not in (2, 4):
                    continue
            u = mt[iv[ids[0]]][ids[1]]
            v = mt[iv[ids[0]]][ids[2]]
            if mt[u][v] == mt[v][u]:
                continue
        return [G.elements[a] for a in ids]
    raise ValueError("could not realize affine cell policy")


def _affine_copy(grid):
    return [[list(cell) for cell in row] for row in grid]


def _affine_component_rows(G, side, grid):
    """Probe the scaffold's exact component action, never a private lift."""
    if side == "A":
        rows = lifted_matrices(G, grid, [[[]]])[0]
    else:
        rows = lifted_matrices(G, [[[]]], grid)[1]
    mask = (1 << (len(grid[0]) * G.q)) - 1
    return [row & mask for row in rows]


def _affine_columns(rows, width):
    columns = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return columns


def _affine_component_screen(rows, width):
    """Exact exclusion test through kernel weight four, plus connectivity."""
    columns = _affine_columns(rows, width)
    bad = int(0 in columns or len(set(columns)) != width)
    singles = set(columns)
    pairs = set()
    if not bad:
        for i, a in enumerate(columns):
            for b in columns[:i]:
                syndrome = a ^ b
                if syndrome in singles or syndrome in pairs:
                    bad = 1
                    break
                pairs.add(syndrome)
            if bad:
                break

    parent = list(range(len(rows)))

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for column in columns:
        if not column:
            continue
        first = (column & -column).bit_length() - 1
        remaining = column & (column - 1)
        while remaining:
            bit = remaining & -remaining
            a, b = root(first), root(bit.bit_length() - 1)
            if a != b:
                parent[b] = a
            remaining ^= bit
    components = len({root(a) for a in range(len(rows))})
    return (bad, components - 1, len(rows) - gf2_rank(rows),
            _design_four_cycles(rows)), columns


def _affine_six_exposure(columns):
    """Triple-syndrome collisions after excluding kernel weights <=4.

    All component columns have odd weight three, so odd-weight kernel
    words are absent. With the <=4 screen passed, colliding triples
    are disjoint and measure six-term dependency multiplicity.
    """
    triples = {}
    collisions = 0
    for i in range(len(columns)):
        for j in range(i):
            ab = columns[i] ^ columns[j]
            for h in range(j):
                syndrome = ab ^ columns[h]
                previous = triples.get(syndrome, 0)
                collisions += previous
                triples[syndrome] = previous + 1
    return collisions


def _affine_component_pool(rng, G, side, support, mode, tables):
    """Multistart cycle refinement, then dependency-based finalist ranking."""
    import math
    positions = [(i, j) for i, row in enumerate(support)
                 for j, count in enumerate(row) if count]
    proposals = []

    for restart in range(3):
        grid = [[_affine_cell(rng, G, count, mode, i, j, tables)
                 for j, count in enumerate(row)]
                for i, row in enumerate(support)]
        rows = _affine_component_rows(G, side, grid)
        score = _design_four_cycles(rows)
        local_best = _affine_copy(grid)
        local_score = score

        for step in range(72):
            i, j = rng.choice(positions)
            old = grid[i][j]
            grid[i][j] = _affine_cell(
                rng, G, support[i][j], mode, i, j, tables)
            trial_rows = _affine_component_rows(G, side, grid)
            trial_score = _design_four_cycles(trial_rows)
            delta = trial_score - score
            temperature = 8.0 * (1.0 - step / 72.0) + 0.2
            if delta <= 0 or rng.random() < math.exp(-delta / temperature):
                score = trial_score
                if score < local_score:
                    local_score = score
                    local_best = _affine_copy(grid)
            else:
                grid[i][j] = old
            if step in (23, 47, 71):
                proposals.append(_affine_copy(grid))
        proposals.append(local_best)

    screened = []
    seen = set()
    width = len(support[0]) * G.q
    for grid in proposals:
        signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                          for row in grid for cell in row)
        if signature in seen:
            continue
        seen.add(signature)
        rows = _affine_component_rows(G, side, grid)
        quality, columns = _affine_component_screen(rows, width)
        screened.append((quality, grid, columns))
    screened.sort(key=lambda item: item[0])

    finalists = []
    for quality, grid, columns in screened[:3]:
        exposure = (10**12 if quality[0]
                    else _affine_six_exposure(columns))
        # First avoid small kernels and disconnected components.
        # Rank deficit and cycles shortlist; six-word exposure refines it.
        key = quality[:2] + (exposure, quality[2], quality[3])
        finalists.append((key, grid))
    finalists.sort(key=lambda item: item[0])
    return finalists[:2]


def NEW_CTX(rng, G, idx):
    """Twenty support/order briefs; independently optimize their labels."""
    design = idx % 20
    mode = design % 4
    profile = _AFFINE_B_PROFILES[design // 4]
    sa = [[3, 3, 3]]
    sb = [[column[i] for column in profile] for i in range(2)]
    wx = max(sum(row) for row in sa) + max(
        sum(row[j] for row in sb) for j in range(4))
    wz = max(sum(row) for row in sb) + max(
        sum(row[j] for row in sa) for j in range(3))
    assert wx == 12 and wz <= 11
    assert max(wx, wz) <= W_MAX
    assert _no_dead_lines(sa, sb)

    tables = _affine_tables(G)
    best = None
    for batch in range(2):
        apool = _affine_component_pool(
            rng, G, "A", sa, mode, tables)
        bpool = _affine_component_pool(
            rng, G, "B", sb, mode, tables)
        for akey, A in apool:
            for bkey, B in bpool:
                hx, hz, n = lifted_matrices(G, A, B)
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if not (k > 0 and 275 * n <= 1000 * k < 325 * n):
                    continue
                sx = _design_small_logical(hx, hz, n)
                sz = _design_small_logical(hz, hx, n)
                cx = _design_four_cycles(hx)
                cz = _design_four_cycles(hz)
                key = (
                    int(sx) + int(sz),
                    akey[0] + bkey[0],
                    akey[1] + bkey[1],
                    max(akey[2], bkey[2]),
                    akey[2] + bkey[2],
                    max(cx / len(hx), cz / len(hz)),
                    cx + cz,
                    k - 168,
                )
                if best is None or key < best[0]:
                    best = (key, A, B, hx, hz)
        if best is not None and best[0][0] == 0:
            break

    if best is None:
        raise ValueError("no affine filling passed the target-rate screen")
    key, A, B, hx, hz = best
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) <= min(12, W_MAX)
    assert 6 * sum(row.bit_count() for row in hx + hz) == 25704
    return {"idx": idx, "design": design, "screen": key,
            "cells": {"A": A, "B": B}}


def CELL(rng, G, side, i, j, ctx):
    """Return distinct elements of the selected complete filling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Distance-only ordering is not circuit-score ordering: parent slot 58 had d2s=9, s=5.872, while slot 12 had d2s=8, s=6.018.
# REFUTED: Lower ASC depth alone guarantees better circuit fitness: current slot 42 had depth 14 and s=6.189, while slot 5 had depth 15 and s=6.657.
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
