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
CONCEPT = ("Dic4, order 16, shapes (3,5)x(3,5): full-rank augmentation support cards and screened order-eight binomials should improve the parent's circuit fitness 7.1162 by approaching the k=64 measurement ceiling in cell 0.100.")
GROUP = ("Keep the dicyclic construction rule, resizing Dic11 to Dic4 because the new size factor 34 excludes order 44. Order 16 is the largest dicyclic lift below q=18. Four parent measured codes had zero failures in both bases; this supports preserving the family, not asserting an advantage for its smaller member. The cyclic order-17 equal-3x5 bed reached 7.2, but does not isolate a group effect.")
SHAPES = ("Equal (3,5) components have 80 variables and 48 checks each. n=544 and kb=64 give guaranteed rate 2/17=0.117647 in the open-ended 0.100 cell. Require verified component ranks 48 and full-CSS k=64. Parent slot34 had two Z failures, 38 collective logical flips and Acol1/Bcol2 enrichment 2.0x; four measured siblings had none, so the evidence does not identify a persistent block defect.")
FILLING = ("Twenty canonical monomial/doubleton support-card pairs have augmentation rank three on both sides, component row totals <=7 and column totals <=5. Check both HX and HZ budgets explicitly against twelve, prioritizing lower-weight cards. Doubleton ratios have order eight; no archived labeling is reused. Optimize fresh voltages using helper-derived component actions, short-cycle heuristics, exact kernel-word<=4 exclusion, weight<=6 collision warnings, triple boundaries and capped sampled witnesses. Full verified matrices enforce k=64, CSS and absence of quantum logicals<=4. The intended gain is reliable memory with reduced extraction exposure, not distance as fitness; the zero-failure ceiling at k=64 still limits the achievable score.")
FAMILY_NAME = "r010_dic4_35x35_aug33_order8_lowexposure_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 16

def custom_ops(q):
    # Same dicyclic rule, now Dic4:
    # x^8=1, y^2=x^4, y*x*y^-1=x^-1.
    # Encoding (a,b) means x^a*y^b, with b in {0,1}.
    p = 4
    m = 2 * p
    assert q == 4 * p
    els = [(a, b) for b in range(2) for a in range(m)]

    def red(g):
        return (g[0] % m, g[1] % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c) + p*b*d) % m,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % m, 0) if b == 0 else ((a + p) % m, 1)

    def idx(g):
        a, b = red(g)
        return m*b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (3, 5)               # (ra, ca)
B_SHAPE = (3, 5)               # (rb, cb)


def _dv_columns(rows, n):
    """Column syndromes of an already verified lifted matrix."""
    cols = [0] * n
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return cols


def _dv_pivots(rows):
    piv = {}
    for row in rows:
        while row:
            p = row.bit_length() - 1
            if p in piv:
                row ^= piv[p]
            else:
                piv[p] = row
                break
    return piv


def _dv_remainder(word, piv):
    while word:
        p = word.bit_length() - 1
        if p not in piv:
            return word
        word ^= piv[p]
    return 0


def _dv_low4(rows, stabilizers, n):
    """Exact existence test for a nontrivial kernel word of weight <=4."""
    cols = _dv_columns(rows, n)
    piv = _dv_pivots(stabilizers)
    seen = {0: 0}
    for i, syndrome in enumerate(cols):
        word = 1 << i
        if syndrome in seen:
            if _dv_remainder(word ^ seen[syndrome], piv):
                return True
        else:
            seen[syndrome] = word
    for i in range(n):
        si = cols[i]
        wi = 1 << i
        for j in range(i):
            syndrome = si ^ cols[j]
            word = wi | (1 << j)
            if syndrome in seen:
                if _dv_remainder(word ^ seen[syndrome], piv):
                    return True
            else:
                seen[syndrome] = word
    return False


def _dv_components(hx, hz, q):
    """Project both components from the verified full lift."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    arows = []
    for i in range(ra):
        for r in range(q):
            row = hx[(i * cb) * q + r]
            packed = 0
            for j in range(ca):
                packed |= ((row >> (j * cb * q)) & mask) << (j * q)
            arows.append(packed)
    bmask = (1 << (cb * q)) - 1
    brows = [hz[a * q + r] & bmask
             for a in range(rb) for r in range(q)]
    return arows, brows


def _dv_cycles(rows, n):
    """Four-cycle count and a row-intersection triangle heuristic."""
    nr = len(rows)
    adjacency = [0] * nr
    c4 = 0
    for i in range(nr):
        for j in range(i):
            common = (rows[i] & rows[j]).bit_count()
            c4 += common * (common - 1) // 2
            if common:
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i
    triangles = 0
    for i in range(nr):
        lower = adjacency[i] & ((1 << i) - 1)
        while lower:
            bit = lower & -lower
            j = bit.bit_length() - 1
            triangles += (adjacency[i] & adjacency[j]
                          & ((1 << j) - 1)).bit_count()
            lower ^= bit
    stars = 0
    for col in _dv_columns(rows, n):
        degree = col.bit_count()
        stars += degree * (degree - 1) * (degree - 2) // 6
    # Exact six-cycle count when intersections are simple; heuristic otherwise.
    return c4, max(0, triangles - stars)


def _dv_kernel_estimate(rng, rows, n, trials):
    """Actual sampled kernel witnesses: upper bounds, never certificates."""
    cols = _dv_columns(rows, n)
    order = list(range(n))
    best = n + 1
    for _ in range(trials):
        rng.shuffle(order)
        piv, basis = {}, []
        for c in order:
            syndrome, word = cols[c], 1 << c
            while syndrome:
                p = syndrome.bit_length() - 1
                if p in piv:
                    s, w = piv[p]
                    syndrome ^= s
                    word ^= w
                else:
                    piv[p] = (syndrome, word)
                    break
            if not syndrome and word:
                basis.append(word)
                best = min(best, word.bit_count())
        # Every XOR remains in the kernel of the verified component.
        for sweep in range(2):
            rng.shuffle(basis)
            for i, word in enumerate(basis):
                weight = word.bit_count()
                for other in basis[:i]:
                    proposal = word ^ other
                    if proposal:
                        pw = proposal.bit_count()
                        best = min(best, pw)
                        if pw < weight:
                            word, weight = proposal, pw
                basis[i] = word
        if best <= 6:
            break
    return best


def NEW_CTX(rng, G, idx):
    """Twenty support contracts, then independent component optimization."""
    from itertools import combinations_with_replacement, permutations

    q = G.q
    assert q == 16
    component_n = 5 * q
    row_permutations = list(permutations(range(3)))

    def support(pattern):
        return [[1 + ((mask >> i) & 1) for mask in pattern]
                for i in range(3)]

    # Canonical count cards under component row/column permutations.
    # This is support canonicalization, not a claim of complete Tanner
    # nonisomorphism; final voltage assignments are independently searched.
    cards = set()
    for pattern in combinations_with_replacement(range(7), 5):
        extras = [sum((mask >> i) & 1 for mask in pattern)
                  for i in range(3)]
        if max(extras) > 2:
            continue
        aug = [sum((1 ^ ((mask >> i) & 1)) << j
                   for j, mask in enumerate(pattern))
               for i in range(3)]
        if gf2_rank(aug) != 3:
            continue
        canonical = min(
            tuple(sorted(
                sum(((mask >> p[i]) & 1) << i for i in range(3))
                for mask in pattern))
            for p in row_permutations)
        cards.add(canonical)

    book = sorted(cards, key=lambda p: (
        sum(mask.bit_count() for mask in p), p))
    contracts = []
    for a in range(len(book)):
        for b in range(a, len(book)):
            aa, bb = support(book[a]), support(book[b])
            weight = _grid_weights(aa, bb)
            if weight <= min(12, W_MAX):
                total = sum(map(sum, aa)) + sum(map(sum, bb))
                contracts.append((weight, total, a, b))
    contracts.sort()
    assert len(contracts) >= 20
    weight, total, pa, pb = contracts[idx % 20]
    sa, sb = support(book[pa]), support(book[pb])
    mode = (idx // 20) % 3

    # Explicitly check both CSS budgets for this contract.
    wx = max(map(sum, sa)) + max(map(sum, zip(*sb)))
    wz = max(map(sum, sb)) + max(map(sum, zip(*sa)))
    assert max(wx, wz) == weight <= W_MAX
    assert _no_dead_lines(sa, sb)
    for counts in (sa, sb):
        aug = [sum((count & 1) << j
                   for j, count in enumerate(row)) for row in counts]
        assert gf2_rank(aug) == 3

    # Cache actions only by asking the verified helper. No separately
    # implemented left/right lift participates in screening.
    mask = (1 << q) - 1
    actions = [{}, {}]
    for g in G.elements:
        hx, _, _ = lifted_matrices(G, [[[g]]], [[[]]])
        _, hz, _ = lifted_matrices(G, [[[]]], [[[g]]])
        actions[0][G._key(g)] = tuple(row & mask for row in hx)
        actions[1][G._key(g)] = tuple(row & mask for row in hz)

    def component_rows(grid, side):
        rows = []
        for row in grid:
            out = [0] * q
            for j, cell in enumerate(row):
                for g in cell:
                    block = actions[side][G._key(g)]
                    for t, bits in enumerate(block):
                        out[t] ^= bits << (j * q)
            rows.extend(out)
        return rows

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def cell(count, i, j, side):
        g = G.pick(rng)
        if count == 1:
            return [g]
        # All rotation displacements below have order eight.
        # Extra harness batches change equality geometry, not a saved
        # labeling by a group automorphism or qubit permutation.
        if mode == 0:
            delta = 1
        elif mode == 1:
            delta = 3 if (i + j + side) % 2 else 1
        else:
            delta = rng.choice((1, 3))
        return [g, G.mul(g, (delta, 0))]

    def draw(counts, side):
        return [[cell(count, i, j, side)
                 for j, count in enumerate(row)]
                for i, row in enumerate(counts)]

    def boundary_screen(rows):
        """Exact <=6 kernel collision warning and triple boundaries.

        A warning is not silently repaired and is not a pL estimate.
        Component words <=4 are excluded separately.
        """
        cols = _dv_columns(rows, component_n)
        seen = {0}
        for value in cols:
            if value in seen:
                return True, 0
            seen.add(value)
        for i, value in enumerate(cols):
            for j in range(i):
                syndrome = value ^ cols[j]
                if syndrome in seen:
                    return True, 0
                seen.add(syndrome)
        bad, boundary = False, 0
        for i, value in enumerate(cols):
            for j in range(i):
                pair = value ^ cols[j]
                for h in range(j):
                    syndrome = pair ^ cols[h]
                    if syndrome in seen:
                        bad = True
                    seen.add(syndrome)
                    size = syndrome.bit_count()
                    if size < 5:
                        boundary += (5 - size) ** 2
        return bad, boundary

    def optimize(counts, side):
        pool, seen = [], set()
        for trial in range(128):
            grid = draw(counts, side)
            signature = tuple(tuple(sorted(G.idx(g) for g in c))
                              for row in grid for c in row)
            if signature in seen:
                continue
            seen.add(signature)
            rows = component_rows(grid, side)
            coarse = _dv_cycles(rows, component_n)
            pool.append((coarse, trial, grid, rows))
            pool.sort(key=lambda item: item[:2])
            del pool[8:]

        finalists = []
        for coarse, trial, grid, rows in pool:
            # Search inter-cell holonomy and binomial geometry within
            # the indexed count contract, not just independent draws.
            for step in range(64):
                proposal = clone(grid)
                i, j = rng.randrange(3), rng.randrange(5)
                proposal[i][j] = cell(counts[i][j], i, j, side)
                proposed_rows = component_rows(proposal, side)
                proposed = _dv_cycles(proposed_rows, component_n)
                if proposed <= coarse:
                    grid, rows, coarse = proposal, proposed_rows, proposed

            # The local-group-algebra augmentation argument motivates
            # full row rank; only verified-action ranks accept it.
            if gf2_rank(rows) != 3 * q:
                continue
            if _dv_low4(rows, [], component_n):
                continue
            bad6, boundary = boundary_screen(rows)
            distance = _dv_kernel_estimate(
                rng, rows, component_n, 48)
            key = (int(bad6), int(distance < 12),
                   boundary, coarse, -min(distance, 16), trial)
            finalists.append((key, grid, rows, distance))
        finalists.sort(key=lambda item: item[0])
        return finalists[:4]

    for batch in range(3):
        apool, bpool = optimize(sa, 0), optimize(sb, 1)
        if not apool or not bpool:
            continue
        options = []
        for a in apool:
            for b in bpool:
                ak, A, ar, da = a
                bk, B, br, db = b
                key = (
                    max(ak[0], bk[0]), ak[0] + bk[0],
                    int(min(da, db) < 12),
                    max(ak[2], bk[2]), ak[2] + bk[2],
                    max(ak[3][0], bk[3][0]),
                    ak[3][0] + bk[3][0],
                    ak[3][1] + bk[3][1],
                    -min(da, db, 16), abs(da - db),
                    ak[-1], bk[-1])
                options.append((key, A, B, ar, br))
        options.sort(key=lambda item: item[0])

        for key, A, B, ar, br in options:
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if n != 544 or k != 64 or 8 * k >= n:
                continue
            # Verify cached actions against this exact submitted lift.
            actual_a, actual_b = _dv_components(hx, hz, q)
            assert ar == actual_a and br == actual_b
            if _dv_low4(hx, hz, n) or _dv_low4(hz, hx, n):
                continue
            assert css_ok(hx, hz)
            assert max(row.bit_count() for row in hx + hz) == weight
            assert 6 * sum(row.bit_count() for row in hx + hz) == (
                6 * q * 8 * total)
            return {"idx": idx, "cells": {"A": A, "B": B},
                    "design": idx % 20, "mode": mode, "k": k,
                    "support_cards": (book[pa], book[pb]),
                    "screen": key}

    raise ValueError("no full-rank Dic4 contract passed short-logical screening")


def CELL(rng, G, side, i, j, ctx):
    """Return the precomputed distinct elements; no additional randomness."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal uniform D7 labels on (6,2)x(6,2) do not suffice for reliable memory: measured s=3.30-3.59, abundant weight-four sector-two logicals, and roughly 10-45 percent failed-shot rates.
# REFUTED: d2s=26 guarantees ceiling-level Dic11 circuit performance: parent slot34 had two Z failures, 38 collective logical flips and s=6.864, while slot26 with the same d2s had zero failures in both bases.
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
