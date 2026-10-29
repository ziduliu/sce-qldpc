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
CONCEPT = ("For the affine matrix group AGL(1,7), order 42, A=(1,2), B=(2,6), uniform degree-three sector-one protection and independently optimized components should raise the 0.300 frontier's measured worst-basis fitness above 5.8.")
GROUP = ("Jump to affine matrices over F7: elements act as z -> a*z+b, a!=0. This order-42 group is not isomorphic to the order-48 parent. Its centre-one, seven-class fingerprint agrees with the order-42 beds scoring 7.2 at (2,3)x(2,3) and 4.6 at (1,3)x(2,4); those readings motivate, but do not establish, performance at this untried shape bed.")
SHAPES = ("Target 0.300 with A=(1,2), B=(2,6): n=588, kb=168, guaranteed rate 2/7=0.285714, and lift order 42. Components have 84 variables/42 checks and 252 variables/84 checks. Parent sector-one witnesses spanned Acol/Bcol blocks; failure enrichments shifted among Bcol0,2,3,4 and were mostly weak. Reinforce both components rather than infer a causal hotspot or transferable logical-qubit index.")
FILLING = ("Twenty architectural briefs cross four B supports with five relative-element policies; further harness batches alter the A quotient-support brief and binomial-order schedule. A cells have three terms; every B row has nine and every B column three. Thus HX rows weigh 9, HZ rows 12, and six-round CNOT count is 25704, versus parent slot5's 27936 and slot10's 29952. Verified component lifts undergo multistart short-cycle optimization, connectivity/rank screening, exact small-kernel tests and capped sampled-kernel/multiplicity preferences. Final verified ranks retain [0.275,0.325), with small-logical exclusion preferred. Reduced gate exposure and decoder ambiguity, not distance alone, are the hypotheses.")
FAMILY_NAME = "r030_affineF7_q42_12x26_biregular_orders_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 42

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("AGL(1,7) requires order 42")
    # Matrix [[a,b],[0,1]] over F7, with a nonzero.
    # Multiplication is composition of affine transformations.
    els = [(a, b) for a in range(1, 7) for b in range(7)]

    def red(g):
        a, b = int(g[0]) % 7, int(g[1]) % 7
        if a == 0:
            raise ValueError("singular affine matrix")
        return (a, b)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a*c) % 7, (a*d+b) % 7)

    def inv(g):
        a, b = red(g)
        ai = pow(a, 5, 7)
        return (ai, (-ai*b) % 7)

    def idx(g):
        a, b = red(g)
        return 7*(a-1)+b

    return dict(elements=els, ident=(1, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (2, 6)               # (rb, cb)


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


def _af_copy(grid):
    return [[list(cell) for cell in row] for row in grid]


def _af_cell(rng, G, side, i, j, count, policy, batch):
    """Affine relative-support brief; translations are not saved orbit copies."""
    if count == 0:
        return []
    anchor = G.pick(rng)
    if count == 1:
        return [anchor]
    if count == 2:
        if policy < 2:
            order = 7
        elif policy == 2:
            order = (7, 3)[(j+batch) % 2]
        elif policy == 3:
            order = (7, 6)[(j+batch) % 2]
        else:
            order = (7, 3, 6)[(j+batch) % 3]
        if order == 7:
            relative = (1, rng.randrange(1, 7))
        else:
            multiplier = rng.choice((2, 4) if order == 3 else (3, 5))
            relative = (multiplier, rng.randrange(7))
        return [anchor, G.mul(anchor, relative)]

    if count != 3:
        raise ValueError("unsupported affine support weight")
    book = (
        ((0, 0, 1), (0, 0, 1)),
        ((0, 1, 2), (0, 1, 2)),
        ((0, 0, 1), (0, 1, 2)),
        ((0, 1, 2), (0, 1, 2)),
        ((0, 1, 3), (0, 1, 2)),
    )
    phases = book[policy][j % 2 if side == "A" else (i+j) % 2]
    if side == "A":
        if batch % 3 == 1 and j == 0:
            phases = (0, 1, 3)
        elif batch % 3 == 2 and j == 1:
            phases = (0, 2, 3)

    relatives = [(1, 0)]
    for phase in phases[1:]:
        multiplier = pow(3, phase, 7)
        choices = [(multiplier, b) for b in range(7)
                   if (multiplier, b) not in relatives]
        relatives.append(rng.choice(choices))
    cell = [G.mul(anchor, h) for h in relatives]
    assert len(set(cell)) == count
    return cell


def _af_grid(rng, G, side, support, policy, batch):
    return [[_af_cell(rng, G, side, i, j, count, policy, batch)
             for j, count in enumerate(row)]
            for i, row in enumerate(support)]


def _af_rows(G, side, grid):
    """Obtain the component solely through the verified construction."""
    if side == "A":
        rows = lifted_matrices(G, grid, [[[]]])[0]
    else:
        rows = lifted_matrices(G, [[[]]], grid)[1]
    width = len(grid[0]) * G.q
    mask = (1 << width) - 1
    return [row & mask for row in rows], width


def _af_cycles(rows, width):
    """Four-cycles, a six-cycle proxy, and check-graph components.

    Triangle-minus-stars is exact for six-cycles when four-cycles
    are absent; otherwise use it only as an optimization proxy.
    """
    size = len(rows)
    adjacency = [0] * size
    degrees = [0] * width
    four = 0
    for i, row in enumerate(rows):
        bits = row
        while bits:
            bit = bits & -bits
            degrees[bit.bit_length()-1] += 1
            bits ^= bit
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            four += overlap*(overlap-1)//2
            if overlap:
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i

    triangles = 0
    for i in range(size):
        later = adjacency[i] >> (i+1)
        while later:
            bit = later & -later
            j = i+1+bit.bit_length()-1
            triangles += (
                (adjacency[i] & adjacency[j]) >> (j+1)
            ).bit_count()
            later ^= bit
    stars = sum(d*(d-1)*(d-2)//6 for d in degrees)

    unseen = (1 << size)-1
    parts = 0
    while unseen:
        todo = unseen & -unseen
        unseen ^= todo
        parts += 1
        while todo:
            bit = todo & -todo
            todo ^= bit
            neighbors = adjacency[bit.bit_length()-1] & unseen
            unseen ^= neighbors
            todo |= neighbors
    return four, max(0, triangles-stars), parts


def _af_small_kernel(rows, width, limit):
    """Exact kernel existence test through weight four or six."""
    columns = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            columns[bit.bit_length()-1] |= 1 << r
            row ^= bit
    seen = {0}
    for column in columns:
        if column in seen:
            return True
        seen.add(column)
    for i, column in enumerate(columns):
        for j in range(i):
            syndrome = column ^ columns[j]
            if syndrome in seen:
                return True
            seen.add(syndrome)
    if limit == 6:
        for i, column in enumerate(columns):
            for j in range(i):
                pair = column ^ columns[j]
                for h in range(j):
                    syndrome = pair ^ columns[h]
                    if syndrome in seen:
                        return True
                    seen.add(syndrome)
    return False


def _af_kernel_scan(rng, rows, width, trials):
    """Sample actual kernel words and short-word multiplicity.

    The minimum is an upper bound, never a distance certificate or
    a circuit-failure estimate. All sampled words use verified rows.
    """
    minimum = width
    small = set()

    def inspect(word):
        weight = word.bit_count()
        if weight <= 12:
            small.add(word)
        return weight

    for _ in range(trials):
        mat = list(rows)
        order = list(range(width))
        rng.shuffle(order)
        pivots = []
        rank = 0
        for c in order:
            bit = 1 << c
            found = next((r for r in range(rank, len(mat))
                          if mat[r] & bit), None)
            if found is None:
                continue
            mat[rank], mat[found] = mat[found], mat[rank]
            pivot = mat[rank]
            for r in range(len(mat)):
                if r != rank and mat[r] & bit:
                    mat[r] ^= pivot
            pivots.append(c)
            rank += 1
            if rank == len(mat):
                break
        occupied = set(pivots)
        basis = []
        for c in order:
            if c in occupied:
                continue
            bit = 1 << c
            word = bit
            for r, p in enumerate(pivots):
                if mat[r] & bit:
                    word ^= 1 << p
            basis.append(word)
            minimum = min(minimum, inspect(word))
        for i, word in enumerate(basis):
            for j in range(i):
                minimum = min(minimum, inspect(word ^ basis[j]))
        if len(basis) >= 3:
            for shot in range(256):
                a, b, c = rng.sample(range(len(basis)), 3)
                minimum = min(
                    minimum, inspect(basis[a] ^ basis[b] ^ basis[c]))
        if minimum <= 4:
            break
    return minimum, len(small)


def _af_select(rng, G, side, support, policy, batch):
    """Optimize independent fillings inside one prescribed architecture."""
    import math

    positions = [(i, j) for i, row in enumerate(support)
                 for j, count in enumerate(row) if count]
    pool = []

    def analyze(grid):
        rows, width = _af_rows(G, side, grid)
        four, six, parts = _af_cycles(rows, width)
        energy = (100*four + six/5.0)/G.q + 30*(parts-1)
        return energy, rows, width, four, six, parts

    def retain(grid, result):
        energy, rows, width, four, six, parts = result
        deficit = len(rows)-gf2_rank(rows)
        key = (parts-1, deficit, four, six)
        pool.append((key, _af_copy(grid), rows, width))

    for restart in range(3):
        grid = _af_grid(rng, G, side, support, policy, batch)
        current = analyze(grid)
        retain(grid, current)
        local_grid, local = _af_copy(grid), current
        for step in range(200):
            i, j = rng.choice(positions)
            previous = grid[i][j]
            grid[i][j] = _af_cell(
                rng, G, side, i, j, support[i][j], policy, batch)
            proposed = analyze(grid)
            delta = proposed[0]-current[0]
            temperature = 20.0*(0.975**step)+0.15
            if delta <= 0 or rng.random() < math.exp(
                    -min(700.0, delta/temperature)):
                current = proposed
                if current[0] < local[0]:
                    local_grid, local = _af_copy(grid), current
            else:
                grid[i][j] = previous
        retain(local_grid, local)

    pool.sort(key=lambda item: item[0])
    finalists, seen = [], set()
    for cost, grid, rows, width in pool:
        signature = tuple(tuple(tuple(sorted(cell)) for cell in row)
                          for row in grid)
        if signature in seen:
            continue
        seen.add(signature)
        bad = _af_small_kernel(rows, width, 6 if side == "A" else 4)
        distance, multiplicity = _af_kernel_scan(
            rng, rows, width, 24 if side == "A" else 16)
        finalists.append({
            "grid": grid, "cost": cost,
            "bad": int(bad or distance <= 6),
            "distance": distance, "multiplicity": multiplicity,
        })
        if len(finalists) == 3:
            break
    return finalists


def NEW_CTX(rng, G, idx):
    """Four support architectures times five algebraic labeling briefs."""
    # Every listed column has degree three and every row sums to nine.
    # The number of zero cells is 0, 1, 2 or 3, respectively.
    profiles = (
        ((1, 2), (1, 2), (1, 2), (2, 1), (2, 1), (2, 1)),
        ((0, 3), (2, 1), (2, 1), (2, 1), (2, 1), (1, 2)),
        ((0, 3), (3, 0), (1, 2), (1, 2), (2, 1), (2, 1)),
        ((0, 3), (0, 3), (3, 0), (2, 1), (2, 1), (2, 1)),
    )
    case, batch = idx % 20, idx // 20
    architecture, policy = case // 5, case % 5
    sa = [[3, 3]]
    sb = [[column[i] for column in profiles[architecture]]
          for i in range(2)]
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert all(sum(row) == 9 for row in sb)
    assert all(sum(row[j] for row in sb) == 3 for j in range(6))
    # Explicitly check both CSS budgets.
    assert sum(sa[0]) + 3 == 9
    assert 9 + max(sa[0]) == 12

    apool = _af_select(rng, G, "A", sa, policy, batch)
    bpool = _af_select(rng, G, "B", sb, policy, batch)
    options = []
    kb = 4*G.q
    for a in apool:
        for b in bpool:
            A, B = a["grid"], b["grid"]
            hx, hz, n = lifted_matrices(G, A, B)
            k = n-gf2_rank(hx)-gf2_rank(hz)
            if not (275*n <= 1000*k < 325*n):
                continue
            ac, bc = a["cost"], b["cost"]
            da, db = min(a["distance"], 12), min(b["distance"], 12)
            key = (
                a["bad"]+b["bad"], ac[0]+bc[0], k-kb,
                ac[2]+bc[2], -min(da, db),
                a["multiplicity"]+b["multiplicity"],
                abs(da-db), ac[3]+bc[3],
            )
            options.append((key, A, B, hx, hz, n, k))
    options.sort(key=lambda item: item[0])

    fallback = None
    for key, A, B, hx, hz, n, k in options:
        assert n == 588
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx+hz) <= min(12, W_MAX)
        assert 6*sum(row.bit_count() for row in hx+hz) == 25704
        candidate = {
            "idx": idx, "case": case, "batch": batch, "k": k,
            "screen": key, "cells": {"A": A, "B": B},
        }
        if fallback is None:
            fallback = candidate
        if _design_small_logical(hx, hz, n):
            continue
        if _design_small_logical(hz, hx, n):
            continue
        return candidate
    if fallback is not None:
        return fallback

    # Preserve the slot's prescribed support and element policy in rescue.
    for attempt in range(64):
        A = _af_grid(rng, G, "A", sa, policy, batch)
        B = _af_grid(rng, G, "B", sb, policy, batch)
        hx, hz, n = lifted_matrices(G, A, B)
        k = n-gf2_rank(hx)-gf2_rank(hz)
        if 275*n <= 1000*k < 325*n:
            assert css_ok(hx, hz)
            assert max(row.bit_count() for row in hx+hz) <= min(12, W_MAX)
            return {"idx": idx, "case": case, "batch": batch, "k": k,
                    "cells": {"A": A, "B": B}}
    raise ValueError("no affine filling retained target cell 0.300")


def CELL(rng, G, side, i, j, ctx):
    """Return distinct elements of the selected complete filling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Distance-only ordering is not circuit-score ordering: parent slot 58 had d2s=9, s=5.872, while slot 12 had d2s=8, s=6.018.
# REFUTED: Low ASC depth alone guarantees basis-balanced reliability: current-parent slot42 had depth14 in both bases but 11/200000 Z failures versus 91/200000 X failures.
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
