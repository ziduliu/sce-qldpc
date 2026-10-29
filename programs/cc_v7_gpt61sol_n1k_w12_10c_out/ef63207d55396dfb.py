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
CONCEPT = ("On cyclic Z37, order 37, shapes (2,6)x(2,6), globally separated relative voltage gaps and verified cofactor-kernel selection should yield five measured distance readings >=12 instead of the parent's ten; this is a falsifiable selection hypothesis, not a distance certificate.")
GROUP = ("Frozen metacyclic Z37:Z1 with trivial action, q=37. This bed improved from fitness 14.468 to 40.0, but all five measured minima are ten. Its commutative group algebra permits explicit cofactor kernel families; target those families rather than infer depth from short-cycle counts.")
SHAPES = ("Frozen A=(2,6), B=(2,6): n=1480, k>=592, guaranteed rate 0.400 targets cell 0.400; q=37 is the largest allowed lift. Equal components have length 222 and 74 checks each. Parent measured minima occupy sector-1 Acol/Bcol blocks throughout both components, with none in Arow/Brow; all measured k-kb values are zero.")
FILLING = ("Keep degree-three columns and nine-term rows, hence w<=12. Remove the two-pure-column profile: its pure triple and two same-orientation split columns supply a cofactor word of weight at most ten. Twenty cards vary the remaining support pairs, four cycle objectives, and doubled-gap exclusions. Signed within-cell gaps are distinct across BOTH component rows. Optimize short cofactor words and binomial multiples computed from verified component columns, then select finalists using shifted cofactor cancellations and stronger randomized circuit cancellations. Exact lifted ranks retain the 0.400 band; valid fallbacks remain available, and no screen certifies quantum distance.")
HOW_AB = FILLING
FAMILY_NAME = "r040_m37r1s1_26x26_globalgaps_cofactor12"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 37

def custom_ops(q):
    # metacyclic Z_37 x| Z_1: y x y^-1 = x^1 (s^r = 1 mod m);
    # rolled, not designed - see make_rate_seeds.py
    m, r, s = 37, 1, 1
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 6)               # (ra, ca)
B_SHAPE = (2, 6)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Construct one of twenty globally constrained cubic design cards."""
    import itertools
    import math

    q = G.q
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    card = idx % 20
    deck = card // 4
    mode = (card % 4 + idx // 20) % 4
    profiles = (
        [(2, 1)] * 3 + [(1, 2)] * 3,
        [(3, 0), (2, 1)] + [(1, 2)] * 4,
    )
    # Decks 3/4 additionally exclude doubled binomial gap relations.
    # No card is obtained by relabeling another card's final filling.
    profile_pairs = ((0, 0), (0, 1), (1, 1), (0, 0), (0, 1))
    pa, pb = profile_pairs[deck]
    mask = (1 << q) - 1

    def copy_cells(cells):
        return [[list(cell) for cell in row] for row in cells]

    def signature(A, B):
        return tuple(tuple(sorted(G.idx(g) for g in cell))
                     for grid in (A, B) for row in grid for cell in row)

    def make_side(profile):
        # Voltage exclusions construct a four-cycle-free component.
        pattern = list(profiles[profile])
        rng.shuffle(pattern)
        if rng.randrange(2):
            pattern = [(b, a) for a, b in pattern]
        cells = [[[] for _ in pattern] for _ in range(2)]
        within = [set(), set()]
        cross = set()
        binomial_gaps = set()
        for j, counts in enumerate(pattern):
            for _ in range(1000):
                column = [rng.sample(G.elements, count) for count in counts]
                local = [set(), set()]
                legal = True
                for i in range(2):
                    for a, x in enumerate(column[i]):
                        for y in column[i][:a]:
                            d = G.mul(G.inv(x), y)
                            key = min(G.idx(d), G.idx(G.inv(d)))
                            # The parent only separated gaps within one
                            # row. Cross-row equality can collapse a
                            # binomial product in a mixed cofactor.
                            if (key in within[0] or key in within[1]
                                    or key in local[0] or key in local[1]):
                                legal = False
                            local[i].add(key)
                            if len(column[i]) == 2 and deck >= 3:
                                doubled = G.mul(d, d)
                                dk = min(G.idx(doubled),
                                         G.idx(G.inv(doubled)))
                                if dk in binomial_gaps:
                                    legal = False
                                for old_gap in binomial_gaps:
                                    old_element = G.elements[old_gap]
                                    twice = G.mul(old_element, old_element)
                                    tk = min(G.idx(twice),
                                             G.idx(G.inv(twice)))
                                    if tk == key:
                                        legal = False
                between = set()
                for x in column[0]:
                    for y in column[1]:
                        key = G.idx(G.mul(G.inv(x), y))
                        if key in cross or key in between:
                            legal = False
                        between.add(key)
                if legal:
                    for i in range(2):
                        cells[i][j] = column[i]
                        within[i].update(local[i])
                        if len(column[i]) == 2:
                            binomial_gaps.update(local[i])
                    cross.update(between)
                    break
            else:
                return None
        return cells

    def components(hx, hz):
        # Both components are extracted from the VERIFIED LP matrices.
        ar = []
        for i in range(ra):
            for t in range(q):
                source = hx[(i * cb) * q + t]
                row = 0
                for j in range(ca):
                    row |= ((source >> (j * cb * q)) & mask) << (j * q)
                ar.append(row)
        bmask = (1 << (cb * q)) - 1
        br = [hz[a * q + t] & bmask
              for a in range(rb) for t in range(q)]

        answer = []
        for rows, width in ((ar, ca * q), (br, cb * q)):
            cols = [0] * width
            for r, row in enumerate(rows):
                while row:
                    low = row & -row
                    cols[low.bit_length() - 1] |= 1 << r
                    row ^= low
            answer.append(cols)
        return answer

    def rotate(poly, shift):
        shift %= q
        if shift == 0:
            return poly
        return ((poly << shift) & mask) | (poly >> (q - shift))

    def multiply(a, b):
        """Cyclic F2 polynomial product, using verified column signatures."""
        answer = 0
        while b:
            low = b & -b
            answer ^= rotate(a, low.bit_length() - 1)
            b ^= low
        return answer

    def translate(word, shift):
        answer = 0
        for j in range(ca):
            answer |= rotate((word >> (j * q)) & mask, shift) << (j * q)
        return answer

    def cofactor_words(cols):
        """Generate and verify three-block component kernel words.

        At lift coordinate zero, each verified syndrome gives the two
        cyclic polynomials. Thus both left/right conventions are taken
        from lifted_matrices, not independently reconstructed.
        """
        polys = [(cols[j * q] & mask, cols[j * q] >> q)
                 for j in range(ca)]
        minors = {}
        for a, b in itertools.combinations(range(ca), 2):
            minors[a, b] = (
                multiply(polys[a][0], polys[b][1]) ^
                multiply(polys[b][0], polys[a][1]))
        words = []
        for a, b, c in itertools.combinations(range(ca), 3):
            word = (minors[b, c] << (a * q) |
                    minors[a, c] << (b * q) |
                    minors[a, b] << (c * q))
            if not word:
                continue
            syndrome = 0
            remaining = word
            while remaining:
                low = remaining & -remaining
                syndrome ^= cols[low.bit_length() - 1]
                remaining ^= low
            assert syndrome == 0, "cofactor disagrees with verified lift"
            words.append(word)
        return words

    def algebraic_penalty(cols):
        """Penalize witnessed component words below the target twelve."""
        minimum = len(cols)
        short = 0
        for word in cofactor_words(cols):
            weight = word.bit_count()
            minimum = min(minimum, weight)
            short += int(weight < 12)
            # Shifts d and -d yield translated copies of the same word.
            for shift in range(1, (q + 1) // 2):
                doubled = word ^ translate(word, shift)
                if doubled:
                    weight = doubled.bit_count()
                    minimum = min(minimum, weight)
                    short += int(weight < 12)
        return 25000 * max(0, 12 - minimum) ** 2 + 250 * short / q

    def statistics(cols):
        h = 2 * q
        adj = [0] * h
        incidence = [[] for _ in range(h)]
        four = 0
        for v, syndrome in enumerate(cols):
            checks = []
            while syndrome:
                low = syndrome & -syndrome
                u = low.bit_length() - 1
                checks.append(u)
                incidence[u].append(v)
                syndrome ^= low
            assert len(checks) == 3
            for a, u in enumerate(checks):
                for vcheck in checks[:a]:
                    if adj[u] & (1 << vcheck):
                        four += 1
                    adj[u] |= 1 << vcheck
                    adj[vcheck] |= 1 << u

        if four:
            return (four, 0, 0, 0), incidence

        triangle_sum = 0
        projected_four = 0
        local_triangles = []
        for u in range(h):
            local = 0
            for v in range(u + 1, h):
                common = (adj[u] & adj[v]).bit_count()
                projected_four += common * (common - 1) // 2
                if adj[u] & (1 << v):
                    triangle_sum += common
            neighbors = adj[u]
            while neighbors:
                low = neighbors & -neighbors
                v = low.bit_length() - 1
                local += (adj[u] & adj[v]).bit_count()
                neighbors ^= low
            # Remove triangles contributed by individual variables.
            local_triangles.append(local // 2 - len(incidence[u]))
        six = triangle_sum // 3 - len(cols)
        # This is a check-projection statistic, not an exact Tanner C8 count.
        projected_four //= 2
        return (0, six, projected_four, max(local_triangles)), incidence

    def objective(stats):
        four, six, projected, peak = stats
        if four:
            return 10000000 + 100000 * four
        if mode == 0:
            return float(six)
        if mode == 1:
            return six + projected / 32.0
        if mode == 2:
            return six + 4.0 * peak
        return six + projected / 8.0

    def inspect(A, B):
        hx, hz, n = lifted_matrices(G, A, B)
        cols = components(hx, hz)
        stats = [statistics(c)[0] for c in cols]
        cost = sum(objective(s) for s in stats)
        if not any(s[0] for s in stats):
            cost += sum(algebraic_penalty(c) for c in cols)
        return cost, stats, cols, hx, hz, n

    def tiny_kernel(cols):
        """Exact detection of component words of weight <=4."""
        singles = set(cols)
        if 0 in singles or len(singles) != len(cols):
            return True
        pairs = set()
        for a, x in enumerate(cols):
            for y in cols[:a]:
                syndrome = x ^ y
                if syndrome in singles or syndrome in pairs:
                    return True
                pairs.add(syndrome)
        return False

    def weight_six(cols):
        """Exact weight-six screen for degree three with no four-cycles.

        Each variable in a weight-six word has at least three distinct
        adjacent variables in that word. There are two disjoint adjacent
        pairs, allowing a partition into two triples, each containing an
        adjacent pair. Enumerating those triples therefore suffices.
        Weight <=4 words must have been excluded first.
        """
        nc = len(cols)
        _, incidence = statistics(cols)
        adj = [set() for _ in cols]
        for variables in incidence:
            for v in variables:
                adj[v].update(variables)
                adj[v].discard(v)
        seen = {}
        for a in range(nc):
            for b in adj[a]:
                if b <= a:
                    continue
                pair = cols[a] ^ cols[b]
                for c in range(nc):
                    if c == a or c == b:
                        continue
                    lo, hi = min(a, c), max(b, c)
                    mid = a + b + c - lo - hi
                    tag = lo + nc * mid + nc * nc * hi
                    syndrome = pair ^ cols[c]
                    previous = seen.get(syndrome)
                    if previous is not None and previous != tag:
                        return True
                    seen[syndrome] = tag
        return False

    def elimination_reading(cols):
        """Verified component-kernel upper reading, never a lower bound."""
        words = cofactor_words(cols)
        best = min((word.bit_count() for word in words), default=len(cols))

        # Cofactors and binomial multiples alone need not contain the
        # minimum. Search cancellations between their translated orbits.
        translated = [
            [translate(word, shift) for shift in range(q)]
            for word in words
        ]
        for a, word in enumerate(words):
            for b in range(a, len(words)):
                for other in translated[b]:
                    candidate = word ^ other
                    if candidate:
                        best = min(best, candidate.bit_count())
        if best <= 10:
            return best

        order = list(range(len(cols)))
        for _ in range(40):
            rng.shuffle(order)
            pivots = {}
            basis = []
            for v in order:
                syndrome, word = cols[v], 1 << v
                while syndrome:
                    p = syndrome.bit_length() - 1
                    previous = pivots.get(p)
                    if previous is None:
                        pivots[p] = (syndrome, word)
                        break
                    syndrome ^= previous[0]
                    word ^= previous[1]
                if not syndrome:
                    basis.append(word)
                    best = min(best, word.bit_count())

            # Fundamental circuits can be long even when a cancellation
            # is short. Every tested XOR is an actual component word.
            light = sorted(basis, key=int.bit_count)[:40]
            for word in light:
                for other in basis:
                    candidate = word ^ other
                    if candidate:
                        best = min(best, candidate.bit_count())

            # Also cancel fundamental circuits against algebraic words.
            for word in light[:12]:
                for orbit in translated:
                    for other in orbit:
                        candidate = word ^ other
                        if candidate:
                            best = min(best, candidate.bit_count())
            if best <= 10:
                return best
        return best

    keepers = []

    def remember(A, B, record):
        if any(s[0] for s in record[1]):
            return
        key = signature(A, B)
        if any(entry[1] == key for entry in keepers):
            return
        keepers.append((record[0], key, copy_cells(A), copy_cells(B), record))
        keepers.sort(key=lambda entry: entry[0])
        del keepers[8:]

    # Independent voltages, not automorphism or permutation orbit copies.
    starts = []
    for _ in range(12):
        A, B = make_side(pa), make_side(pb)
        if A is None or B is None:
            continue
        record = inspect(A, B)
        starts.append((record[0], A, B, record))
        remember(A, B, record)
    if not starts:
        raise ValueError("could not construct the degree-three profile")

    _, A, B, current = min(starts, key=lambda entry: entry[0])
    for step in range(240):
        grid = A if rng.randrange(2) == 0 else B
        positions = [(i, j) for i, row in enumerate(grid)
                     for j, cell in enumerate(row) if cell]
        i, j = rng.choice(positions)
        cell = grid[i][j]
        t = rng.randrange(len(cell))
        old = cell[t]
        new = G.pick(rng)
        while any(G._key(new) == G._key(g) for g in cell):
            new = G.pick(rng)
        cell[t] = new
        proposed = inspect(A, B)
        legal = not any(s[0] for s in proposed[1])
        temperature = q * (4.0 * (1.0 - step / 240.0) ** 2 + 0.05)
        accept = legal and (
            proposed[0] <= current[0] or
            rng.random() < math.exp((current[0] - proposed[0]) / temperature))
        if accept:
            current = proposed
            remember(A, B, current)
        else:
            cell[t] = old

    finalists = []
    for cost, key, A, B, record in keepers:
        cols = record[2]
        if any(tiny_kernel(c) for c in cols):
            continue
        readings = [elimination_reading(c) for c in cols]
        finalists.append((min(readings), sum(readings), -cost, A, B, record))
    finalists.sort(key=lambda entry: entry[:3], reverse=True)

    fallback = None
    selected = None
    for reading, total, negative_cost, A, B, record in finalists:
        _, stats, cols, hx, hz, n = record
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (k > 0 and 0.375 <= k / n < 0.425):
            continue
        candidate = (A, B, record, k)
        if fallback is None:
            fallback = candidate
        if reading >= 12 and not any(weight_six(c) for c in cols):
            selected = candidate
            break

    # Preserve a valid slot if every finalist still contains a short word.
    if selected is None:
        selected = fallback
    if selected is None:
        for cost, key, A, B, record in keepers:
            hx, hz, n = record[3:]
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if k > 0 and 0.375 <= k / n < 0.425:
                selected = (A, B, record, k)
                break
    if selected is None:
        raise ValueError("no optimized filling in rate cell 0.400")

    A, B, record, k = selected
    hx, hz, n = record[3:]
    assert css_ok(hx, hz)
    sa = [[len(cell) for cell in row] for row in A]
    sb = [[len(cell) for cell in row] for row in B]
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)
    return {"idx": idx, "card": card, "cells": {"A": A, "B": B},
            "screen_k": k, "cycle_cost": record[0]}


def CELL(rng, G, side, i, j, ctx):
    """Return the optimized full cell without further randomization."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal positive-cell rolling with uniform Z37 labels reliably avoids short sector-1 logicals; the parent instead has many d2s=2 slots and all five measured distances equal six, with minimum witnesses confined to sector 1.
# REFUTED: Cubic four-cycle-free lifting with weight-six screening eliminates the weight-ten bottleneck on this C37 bed; all five measured current-parent minima are ten, with thousands of verified sector-1 witnesses.
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
