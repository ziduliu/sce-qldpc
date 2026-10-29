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
CONCEPT = ("S3 x C7, order 42, shapes (1,3)x(1,3): equal degree-three components and reduced connected four-variable absorbing-set exposure should improve worse-basis circuit fitness beyond the approximately 4.9 frontier in cell 0.400.")
GROUP = ("Retain the exact permutation direct-product group and order under L2. The parent scored 5.797 on (1,2)x(2,6), while the related centre-seven order-42 bed scored about 4.6 on (1,3)x(2,4). These readings motivate testing balanced components on this untried bed, not a demonstrated group advantage.")
SHAPES = ("Target 0.400 with equal (1,3) components, each having 126 variables and 42 checks. At q=42, n=420 and kb=168 give guaranteed rate 0.400; verified full ranks retain [0.375,0.425). Both CSS bases have 126 checks. Parent Z failures were diffuse across Acol/Bcol data and Acol/Brow check blocks, mostly 1.0-1.2x enrichment, so the shape change addresses basis imbalance rather than one alleged hotspot.")
FILLING = ("All six cells are distinct-term trinomials: component rows weigh nine and columns three, both CSS check weights are twelve, and six-round CNOT count is 18144 versus the parent's 25704. Twenty initial contracts prescribe different unordered pairs of triangle-order architectures; further slots use additional contracts, not saved-label transformations. Fresh labels undergo verified component-cycle optimization, connectivity and kernel<=4 screening, capped sampled kernel/multiplicity screening, connected absorbing-four exposure and anchored small-boundary comparison. Complete lifted ranks, short quantum logicals and CSS decide acceptance. Parent slot2 had 101/38500 Z failures versus 37/200000 X failures despite d2s=10; structural screens are hypotheses about circuit pL, not its measurement.")
FAMILY_NAME = "r040_S3xC7_13x13_trianglecards_absorbing_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 42

def custom_ops(q):
    from itertools import permutations
    if q != 42:
        raise ValueError("S3 x C7 requires order 42")
    # (p0,p1,p2,c): a permutation of three points and an additive C7 label.
    perms = list(permutations(range(3)))
    els = [p + (c,) for c in range(7) for p in perms]

    def red(g):
        return tuple(int(x) for x in g[:3]) + (int(g[3]) % 7,)

    def mul(g, h):
        return (g[h[0]], g[h[1]], g[h[2]], (g[3] + h[3]) % 7)

    def inv(g):
        p = tuple(g[:3])
        return tuple(p.index(i) for i in range(3)) + ((-g[3]) % 7,)

    def idx(g):
        g = red(g)
        return 6 * g[3] + perms.index(g[:3])

    return dict(elements=els, ident=(0, 1, 2, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


def _relative_order(G, g):
    """Exact order under the validated group law."""
    value = G.e
    for order in range(1, G.q + 1):
        value = G.mul(value, g)
        if G._key(value) == G._key(G.e):
            return order
    raise ValueError("element order exceeds the group order")


def NEW_CTX(rng, G, idx):
    """Balanced shapes with deliberate triangle-order architecture pairs."""
    import math
    from itertools import combinations

    q = G.q
    width = 3 * q
    sa = sb = [[3, 3, 3]]
    assert q == 42
    assert A_SHAPE == B_SHAPE == (1, 3)
    assert _grid_weights(sa, sb) == 12
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)

    # Types distinguish the entire unordered relative-order spectrum.
    # Types 0..2 are noncommuting; 3 and 4 are controlled commuting
    # cells. Every component card retains a noncommuting cell.
    specifications = (
        ((14, 14, 21), False),
        ((3, 14, 14), False),
        ((2, 14, 21), False),
        ((7, 14, 14), True),
        ((7, 21, 21), True),
    )
    orders = {G.idx(g): _relative_order(G, g) for g in G.elements}
    triangles = [[] for _ in specifications]
    nonidentity = [g for g in G.elements if G._key(g) != G._key(G.e)]
    for u, v in combinations(nonidentity, 2):
        relative = G.mul(G.inv(u), v)
        spectrum = tuple(sorted((orders[G.idx(u)], orders[G.idx(v)],
                                 orders[G.idx(relative)])))
        commuting = G._key(G.mul(u, v)) == G._key(G.mul(v, u))
        for t, (wanted, commute) in enumerate(specifications):
            if spectrum == wanted and commuting == commute:
                triangles[t].append([G.e, u, v])
    assert all(triangles)

    architectures = (
        (0,0,0), (0,0,1), (0,1,1), (1,1,1), (0,0,2),
        (0,0,3), (0,1,3), (0,0,4), (0,1,4), (0,3,4),
        (0,2,2), (1,1,2), (0,3,3), (0,4,4), (1,3,4),
    )
    first_twenty = (
        (0,0), (0,1), (0,2), (1,1), (1,2), (2,2),
        (0,3), (1,3), (2,3), (3,3),
        (0,4), (1,4), (0,5), (1,5), (0,6), (1,6),
        (0,7), (1,7), (0,8), (0,9),
    )
    contracts = list(first_twenty)
    contracts.extend((a, b) for a in range(len(architectures))
                     for b in range(a, len(architectures))
                     if (a, b) not in first_twenty)
    if idx >= len(contracts):
        raise ValueError("unused triangle architecture contracts exhausted")
    pa, pb = contracts[idx]
    cards = {"A": architectures[pa], "B": architectures[pb]}

    def draw_grid(side):
        # Whole-cell anchors on a single-row component are gauges.
        # Search internal supports rather than spend proposals on anchors.
        return [[list(rng.choice(triangles[t])) for t in cards[side]]]

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def component(side, grid):
        # Probe only through the scaffold's verified left/right lifts.
        if side == "A":
            rows = lifted_matrices(G, grid, [[[]]])[0]
        else:
            rows = lifted_matrices(G, [[[]]], grid)[1]
        mask = (1 << width) - 1
        return [row & mask for row in rows]

    def columns(rows, ncols):
        result = [0] * ncols
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                result[bit.bit_length() - 1] |= tag
                row ^= bit
        return result

    def connected(cols, nchecks):
        parent = list(range(nchecks))

        def root(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for syndrome in cols:
            if not syndrome:
                return False
            first = syndrome & -syndrome
            a = first.bit_length() - 1
            syndrome ^= first
            while syndrome:
                bit = syndrome & -syndrome
                b = bit.bit_length() - 1
                parent[root(b)] = root(a)
                syndrome ^= bit
        return len({root(r) for r in range(nchecks)}) == 1

    def four(rows):
        return sum(v * (v - 1) // 2
                   for i, row in enumerate(rows)
                   for other in rows[:i]
                   for v in [(row & other).bit_count()])

    def six(rows):
        """Exact Tanner six-cycles, including repeated intersections."""
        total = 0
        nr = len(rows)
        overlap = [[0] * nr for _ in range(nr)]
        for i in range(nr):
            for j in range(i):
                overlap[i][j] = overlap[j][i] = (
                    rows[i] & rows[j]).bit_count()
        for i in range(nr):
            for j in range(i):
                x = overlap[i][j]
                if not x:
                    continue
                common = rows[i] & rows[j]
                for h in range(j):
                    y, z = overlap[j][h], overlap[i][h]
                    if y and z:
                        triple = (common & rows[h]).bit_count()
                        total += (x * y * z -
                                  triple * (x + y + z) + 2 * triple)
        return total

    def boundary_screen(cols):
        """Exact <=4 exclusion and orbit-anchored triple exposure.

        Anchoring avoids storing millions of triple syndromes for B.
        Pair/triple matches warn of actual short words; the boundary
        statistic is not a decoder or circuit-failure certificate.
        """
        seen = {0}
        for value in cols:
            if value in seen:
                return None
            seen.add(value)
        for i, x in enumerate(cols):
            for j in range(i):
                value = x ^ cols[j]
                if value in seen:
                    return None
                seen.add(value)
        boundary = 0
        for anchor in range(0, len(cols), q):
            for i, x in enumerate(cols):
                if i == anchor:
                    continue
                pair = cols[anchor] ^ x
                for j in range(i):
                    if j == anchor:
                        continue
                    value = pair ^ cols[j]
                    if value in seen:
                        boundary += 200
                    weight = value.bit_count()
                    if weight <= 4:
                        boundary += (5 - weight) ** 2
        return boundary

    def sampled_kernel(cols, side):
        """Verified witnesses and sampled multiplicity, not a certificate."""
        srng = random.Random(420031 + 1009 * idx + int(side == "B"))
        order = list(range(len(cols)))
        best = len(cols) + 1
        short = set()
        for _ in range(48):
            srng.shuffle(order)
            pivots = {}
            basis = []
            for c in order:
                syndrome, word = cols[c], 1 << c
                while syndrome:
                    p = syndrome.bit_length() - 1
                    if p not in pivots:
                        pivots[p] = syndrome, word
                        break
                    old_s, old_w = pivots[p]
                    syndrome ^= old_s
                    word ^= old_w
                if not syndrome:
                    candidates = [word]
                    candidates.extend(word ^ other for other in basis)
                    for witness in candidates:
                        weight = witness.bit_count()
                        best = min(best, weight)
                        if weight <= 10 and len(short) < 8192:
                            short.add(witness)
                    basis.append(word)
            if best <= 4:
                break
        return (best,
                sum(w.bit_count() <= 6 for w in short),
                sum(w.bit_count() <= 8 for w in short),
                len(short))

    def short_logical(checks, stabilizers, n):
        """Exact non-stabilizer logical existence through weight four."""
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

        cols = columns(checks, n)
        seen = {0: 0}
        for i, syndrome in enumerate(cols):
            word = 1 << i
            previous = seen.get(syndrome)
            if previous is None:
                seen[syndrome] = word
            elif nontrivial(word ^ previous):
                return True
        for i, x in enumerate(cols):
            for j in range(i):
                syndrome = x ^ cols[j]
                word = (1 << i) | (1 << j)
                previous = seen.get(syndrome)
                if previous is None:
                    seen[syndrome] = word
                elif nontrivial(word ^ previous):
                    return True
        return False

    def absorbing_exposure(cols, side):
        """Sample connected four-variable supports on the verified graph.

        Counts are a common-stream risk surrogate, not exhaustive
        absorbing-set counts or ASC-certified fault probabilities.
        """
        neighbors = [0] * q
        for c, syndrome in enumerate(cols):
            bits = syndrome
            while bits:
                bit = bits & -bits
                neighbors[bit.bit_length() - 1] |= 1 << c
                bits ^= bit
        adjacency = []
        for c, syndrome in enumerate(cols):
            union = 0
            bits = syndrome
            while bits:
                bit = bits & -bits
                union |= neighbors[bit.bit_length() - 1]
                bits ^= bit
            adjacency.append(union & ~(1 << c))

        srng = random.Random(420073 + 1013 * idx + int(side == "B"))
        examined = set()
        severe = proper = relaxed = 0
        loads = [0] * 3
        for trial in range(12000):
            chosen = [srng.randrange(width)]
            occupied = 1 << chosen[0]
            frontier = adjacency[chosen[0]]
            for _ in range(3):
                available = frontier & ~occupied
                if not available:
                    break
                position = srng.randrange(available.bit_count())
                for _ in range(position):
                    available &= available - 1
                bit = available & -available
                c = bit.bit_length() - 1
                chosen.append(c)
                occupied |= bit
                frontier |= adjacency[c]
            if len(chosen) != 4 or occupied in examined:
                continue
            examined.add(occupied)
            syndrome = 0
            for c in chosen:
                syndrome ^= cols[c]
            boundary = syndrome.bit_count()
            if not (0 < boundary <= 4):
                continue
            unsatisfied = [(syndrome & cols[c]).bit_count()
                           for c in chosen]
            if max(unsatisfied) <= 1:
                proper += 1
                severe += int(boundary <= 2)
                for block in {c // q for c in chosen}:
                    loads[block] += 1
            if max(unsatisfied) <= 2:
                relaxed += 1
        return severe, proper, max(loads), relaxed

    def select_component(side):
        pool = []
        seen = set()
        serial = 0

        def analyze(grid):
            rows = component(side, grid)
            f, s = four(rows), six(rows)
            return (80.0 * f + s / 4.0) / q, rows, f, s

        def retain(grid, result):
            nonlocal serial
            signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                              for row in grid for cell in row)
            if signature in seen:
                return
            seen.add(signature)
            serial += 1
            energy, rows, f, s = result
            pool.append(((f, s), serial, clone(grid), rows))
            pool.sort(key=lambda item: item[:2])
            del pool[24:]

        for restart in range(4):
            grid = draw_grid(side)
            current = analyze(grid)
            retain(grid, current)
            for step in range(220):
                j = rng.randrange(3)
                previous = grid[0][j]
                grid[0][j] = list(rng.choice(triangles[cards[side][j]]))
                proposed = analyze(grid)
                delta = proposed[0] - current[0]
                temperature = 18.0 * (0.98 ** step) + 0.12
                if delta <= 0 or rng.random() < math.exp(
                        -min(700.0, delta / temperature)):
                    current = proposed
                    retain(grid, current)
                else:
                    grid[0][j] = previous

        finalists = []
        for coarse, serial, grid, rows in pool:
            cols = columns(rows, width)
            if not connected(cols, q):
                continue
            boundary = boundary_screen(cols)
            if boundary is None:
                continue
            words = sampled_kernel(cols, side)
            traps = absorbing_exposure(cols, side)
            # Stop rewarding optimistic sampled protection beyond ten.
            # Connected decoder-risk structures precede cycle tie-breaks.
            key = (int(words[0] < 8), traps, boundary,
                   words[1], words[2], -min(words[0], 10),
                   words[3], coarse, serial)
            finalists.append((key, grid, words, traps, boundary, coarse))
            if len(finalists) >= 8:
                break
        finalists.sort(key=lambda item: item[0])
        return finalists[:4]

    for attempt in range(3):
        apool = select_component("A")
        bpool = select_component("B")
        options = []
        for _, A, aw, at, ab, ac in apool:
            for _, B, bw, bt, bb, bc in bpool:
                hx, hz, n = lifted_matrices(G, A, B)
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if not (n == 420 and
                        375 * n <= 1000 * k < 425 * n):
                    continue
                trap_key = tuple(value
                                 for a, b in zip(at, bt)
                                 for value in (max(a, b), a + b))
                key = (int(min(aw[0], bw[0]) < 8),) + trap_key + (
                    max(ab, bb), ab + bb,
                    max(aw[1], bw[1]), aw[1] + bw[1],
                    max(aw[2], bw[2]), aw[2] + bw[2],
                    -min(aw[0], bw[0], 10),
                    abs(min(aw[0], 10) - min(bw[0], 10)),
                    max(ac[0], bc[0]), ac[0] + bc[0],
                    ac[1] + bc[1],
                )
                options.append((key, A, B, hx, hz, n, k))
        options.sort(key=lambda item: item[0])

        for key, A, B, hx, hz, n, k in options:
            if short_logical(hx, hz, n) or short_logical(hz, hx, n):
                continue
            assert css_ok(hx, hz)
            assert max(row.bit_count() for row in hx + hz) == 12
            assert max(row.bit_count() for row in hx + hz) <= W_MAX
            assert 6 * sum(row.bit_count() for row in hx + hz) == 18144
            return {"idx": idx, "cells": {"A": A, "B": B},
                    "contract": (pa, pb), "k": k, "screen": key}

    raise ValueError("no balanced S3 x C7 triangle contract passed cell 0.400")


def CELL(rng, G, side, i, j, ctx):
    """Emit the screened full cell, including deliberate zero blocks."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: fewer CNOTs alone imply a better memory score; parent slot 21 had 25200 CNOTs and s=5.427, while slot 39 had 29232 CNOTs and s=6.122.
# REFUTED: Larger d2s plus repeated-neighbor optimization guarantees reliable memory in this order-39 bed: slot57 had d2s=24 and s=5.436 with 100/142000 X failures, while slot59 had d2s=23 and zero failures in both bases, s=7.103.
# REFUTED: k=59, check weight ten and d2s=18 certify ceiling-level memory: current slot4 had six Z failures and s=6.734, whereas slot14 with the same reported parameters had zero failures and s=7.283.
# REFUTED: Noncommuting A trinomials, long-order B differences and short-boundary screening ensure basis-balanced memory on S3 x C7 (1,2)x(2,6): slot2 had 101/38500 Z failures versus 37/200000 X failures despite d2s=10.
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
