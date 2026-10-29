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
CONCEPT = ("Cyclic Z17, order 17, shapes (3,5)x(3,5): three-row components with controlled trivial-character rank deficits should reduce the parent's persistent X-memory failures while retaining measured k=74..84 in cell 0.150.")
GROUP = ("Keep the metacyclic construction rule with m=1 and s=1; change only r from 23 to 17 because the new factor 34 would give n=782 at q=23. Order 17 gives n=578. The existing Z17 (3,5)x(3,5) bed scored only 2.7, so this tests designed supports and labels rather than assuming a shape advantage.")
SHAPES = ("Equal (3,5) components have 85 variables and 51 checks each. The guaranteed bound is k>=68, rate 2/17=0.117647, below the target cell. Support-parity ranks one/two deliberately introduce trivial-character deficits; full evaluator-convention ranks must give 74<=k<=84, rates 0.1280..0.1453. Parent measured codes had 5..11 X failures per 200000 shots affecting many logicals, with different Acol/Bcol enrichments, motivating distributed reinforcement.")
FILLING = ("Twenty explicit support pairs combine distributed doubletons, trinomials and a zero/trinomial incidence variant; additional harness slots use unused support pairs, not saved-code relabelings. Component rows weigh <=7, columns weigh 3..5, and both CSS check weights are <=12. Optimize verified component lifts, test component kernel words through weight six exactly, then sample fundamental words and pair XORs with equal budgets on both sides. The first twenty architectures use 27744..34272 six-round CNOTs. Exact CSS ranks gate the target cell; structural screens and sampled witnesses do not certify circuit pL or ASC schedulability.")
FAMILY_NAME = "r015_m1r17s1_35x35_trivialrank_balanced_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 17

def custom_ops(q):
    # Same metacyclic construction rule; factor 34 requires q<=17.
    # Z_1 x| Z_17 is cyclic, with the unchanged trivial action.
    if not valid_q(q):
        raise ValueError("this shape bed requires order 17")
    m, r, s = 1, 17, 1
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (3, 5)               # (ra, ca)
B_SHAPE = (3, 5)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Optimize fresh labels for a controlled-rank support architecture."""
    import math

    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    q = G.q
    assert (ra, ca, rb, cb, q) == (3, 5, 3, 5, 17)

    # Edits to the dense monomial 3x5 support. Every profile has
    # augmentation rank one or two, row weight <=7 and column weight
    # 3..5. Repeated parity rows do NOT impose repeated group-ring rows:
    # all element labels are independently optimized.
    edits = (
        (),
        ((0, 0, 2),),
        ((0, 0, 2), (1, 0, 2)),
        ((0, 0, 2), (0, 1, 2)),
        ((0, 0, 2), (0, 1, 2), (1, 0, 2), (1, 1, 2)),
        ((0, 0, 2), (1, 0, 2), (2, 1, 2)),
        ((0, 0, 2), (1, 0, 2), (2, 1, 2), (2, 2, 2)),
        ((0, 0, 3),),
        ((0, 0, 3), (1, 1, 3)),
        ((0, 0, 3), (1, 1, 3), (2, 2, 3)),
        ((0, 0, 3), (1, 1, 2)),
        ((0, 0, 3), (1, 1, 2), (2, 1, 2)),
        ((0, 0, 3), (1, 1, 3), (2, 2, 2)),
        ((0, 0, 0), (1, 0, 3)),
    )

    # Explicit unordered pairs avoid deliberately submitting A/B swaps.
    # These are count/incidence briefs, not transformations of one code.
    designs = [
        (2, 3), (2, 4), (2, 5), (2, 6),
        (3, 4), (3, 5), (3, 6),
        (4, 4), (4, 5), (4, 6),
        (5, 6), (6, 6),
        (7, 8), (7, 9), (8, 8), (8, 9), (9, 9),
        (10, 11), (11, 12), (12, 13),
    ]
    used = set(designs)
    designs.extend(
        (a, b) for a in range(len(edits))
        for b in range(a, len(edits)) if (a, b) not in used
    )
    if idx >= len(designs):
        raise ValueError("slot budget exceeds unused support architectures")
    pa, pb = designs[idx]

    def support(profile):
        grid = [[1] * 5 for _ in range(3)]
        for i, j, weight in edits[profile]:
            grid[i][j] = weight
        parity = [
            sum((weight & 1) << j for j, weight in enumerate(row))
            for row in grid
        ]
        assert 1 <= gf2_rank(parity) <= 2
        assert max(map(sum, grid)) <= 7
        assert all(3 <= sum(row[j] for row in grid) <= 5
                   for j in range(5))
        return grid

    sa, sb = support(pa), support(pb)
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert _no_dead_lines(sa, sb)
    assert all(0 <= weight <= CELL_MAX
               for grid in (sa, sb) for row in grid for weight in row)

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def draw(counts):
        return [[rng.sample(G.elements, w) for w in row]
                for row in counts]

    component_cache = {}

    def component_rows(A, B):
        # Zero companions are probes, never submitted slot matrices.
        # Project components only from the verified scaffold lift.
        # Cache the unchanged side during single-label mutations.
        result = []
        for side, grid, width in (("A", A, ca), ("B", B, cb)):
            fingerprint = tuple(
                tuple(sorted(G.idx(g) for g in cell))
                for row in grid for cell in row
            )
            old = component_cache.get(side)
            if old is None or old[0] != fingerprint:
                if side == "A":
                    rows = lifted_matrices(G, grid, [[[]]])[0]
                else:
                    rows = lifted_matrices(G, [[[]]], grid)[1]
                mask = (1 << (width * q)) - 1
                rows = [row & mask for row in rows]
                component_cache[side] = (fingerprint, rows)
            result.append(component_cache[side][1])
        return result[0], result[1]

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

    def column_syndromes(rows, nc):
        columns = [0] * nc
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                columns[bit.bit_length() - 1] |= tag
                row ^= bit
        return columns

    def short_kernel(rows, nc):
        """Exact existence test for a nonzero component word <=6.

        Every subset of size <=3 is visited once. Equal syndromes of
        distinct subsets give a nonzero kernel word of weight <=6.
        This tests components, not quantum logicals or circuit faults.
        """
        columns = column_syndromes(rows, nc)
        seen = {0}
        for syndrome in columns:
            if syndrome in seen:
                return True
            seen.add(syndrome)
        for i in range(nc):
            for j in range(i):
                syndrome = columns[i] ^ columns[j]
                if syndrome in seen:
                    return True
                seen.add(syndrome)
        for i in range(nc):
            for j in range(i):
                pair = columns[i] ^ columns[j]
                for h in range(j):
                    syndrome = pair ^ columns[h]
                    if syndrome in seen:
                        return True
                    seen.add(syndrome)
        return False

    def kernel_scan(rows, nc, seed, trials):
        """Sample actual fundamental kernel words and their pair XORs.

        Returned minima are upper bounds only. Pair XORs improve the
        witness search without treating a missed word as a certificate.
        """
        srng = random.Random(seed)
        columns = column_syndromes(rows, nc)
        order = list(range(nc))
        best = nc + 1
        small = set()

        def record(word):
            nonlocal best
            if not word:
                return
            weight = word.bit_count()
            best = min(best, weight)
            if weight <= 14:
                small.add(word)

        for _ in range(trials):
            srng.shuffle(order)
            pivots = {}
            basis = []
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
                    basis.append(word)
                    record(word)
            for i, word in enumerate(basis):
                for other in basis[:i]:
                    record(word ^ other)
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
        # The parity-rank model motivates 74/78/84 but never substitutes
        # for the exact ranks of the complete evaluator-convention lift.
        k = lifted_k(G, A, B)
        if not 74 <= k <= 84:
            continue
        bad_a = short_kernel(local[1], ca * q)
        bad_b = short_kernel(local[2], cb * q)
        da, na = kernel_scan(local[1], ca * q,
                             271828 + idx * 1009, 128)
        db, nb = kernel_scan(local[2], cb * q,
                             314159 + idx * 1009, 128)
        # Equal-length components receive equal search budgets.
        # Saturate sampled protection rewards, then reduce the number
        # of found short words and short cycles. None is measured pL.
        quality = (int(not bad_a and not bad_b),
                   -int(bad_a) - int(bad_b),
                   min(da, db, 14),
                   -(na + nb),
                   -local[3],
                   -local[4])
        if winner_quality is None or quality > winner_quality:
            winner_quality = quality
            winner = {"A": clone(A), "B": clone(B)}

    if winner is None:
        raise ValueError("no optimized filling retained exact k=74..84")
    hx, hz, n = lifted_matrices(G, winner["A"], winner["B"])
    k = n - gf2_rank(hx) - gf2_rank(hz)
    assert css_ok(hx, hz)
    assert n == 578 and 74 <= k <= 84
    assert 8 * k >= n and 40 * k < 7 * n
    assert max(row.bit_count() for row in hx + hz) <= min(12, W_MAX)
    return {"idx": idx, "architecture": (pa, pb), "k": k,
            "screen": winner_quality, "cells": winner}


def CELL(rng, G, side, i, j, ctx):
    """Return the selected full cell without further random draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Stage-1 settlement is not a reliability certificate: parent slot 4 settled 1498/1500 X shots at stage 1 yet failed 102/1500.
# REFUTED: Check weight <=12 and exclusion of weight-six logicals certify zero circuit failures: all five current measured codes still had 5..11 X failures in 200000 shots, despite minimum found Z-type witness weight 12.
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
