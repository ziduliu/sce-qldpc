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
CONCEPT = ("Cyclic C19, order 19, shapes (2,5)x(2,5): degree-three/four components with reduced short-word ambiguity should improve worst-basis circuit reliability in frontier cell 0.300; circuit measurements must test this.")
GROUP = ("Jump from nonabelian C7 semidirect C3 to prime-order C19, an untried bed in BEDS. The related C20 (2,5)x(2,5) bed reached 5.8. Since ord_19(2)=18, C19 has one nontrivial binary cyclotomic factor, motivating a prime-order rank experiment without claiming a circuit advantage.")
SHAPES = ("Equal (2,5) components have 38 checks and 95 variables each. Order 19 gives n=551 and kb=171, guaranteed rate 9/29=0.31034 in cell 0.300; verified ranks retain k=171. Order 20 is the shape's maximum, but this experiment fixes the fresh prime-order lift. Parent failures touched multiple Acol/Bcol and Arow/Brow blocks with mostly mild enrichment, rather than identifying a unique repair block.")
FILLING = ("Twenty cards cross four support-pair architectures with five canonical internal-label geometries; extra harness slots extend the geometry book. Columns have degree three or four and component rows total at most eight, so both CSS weight sums are <=12. Fresh offsets are screened using verified lifts, connectivity, exact ranks, component words through six, triple boundaries and anchored weight-seven/eight ambiguity; actual quantum logicals through four are rejected. Sampled kernel witnesses are only upper bounds. This targets collective failures and decoding ambiguity, not distance as fitness: parent slot12 had 100/129750 Z failures, and slot18 failed ASC scheduling.")
FAMILY_NAME = "r030_C19_25x25_prime_ambiguitybook_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 19

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C19 requires q=19")
    els = [(a,) for a in range(q)]

    def red(g):
        return (int(g[0]) % q,)

    def mul(g, h):
        return ((int(g[0]) + int(h[0])) % q,)

    def inv(g):
        return ((-int(g[0])) % q,)

    def idx(g):
        return red(g)[0]

    return dict(elements=els, ident=(0,), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 5)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


def _f21_columns(hx, hz, q):
    """Project component columns from the VERIFIED lifted matrices.

    A: sector 1 of X checks at B-column zero.
    B: sector 1 of Z checks at A-column zero.
    No independent left/right-lift implementation is used.
    """
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    ac = [0] * (ca * q)
    bc = [0] * (cb * q)
    amask = sum(((1 << q) - 1) << (j * cb * q)
                for j in range(ca))
    bmask = (1 << (cb * q)) - 1
    for i in range(ra):
        for r in range(q):
            bits = hx[(i * cb) * q + r] & amask
            check = 1 << (i * q + r)
            while bits:
                low = bits & -bits
                c = low.bit_length() - 1
                ac[(c // (cb * q)) * q + c % q] |= check
                bits ^= low
    for a in range(rb):
        for r in range(q):
            bits = hz[a * q + r] & bmask
            check = 1 << (a * q + r)
            while bits:
                low = bits & -bits
                bc[low.bit_length() - 1] |= check
                bits ^= low
    return ac, bc


def _f21_stats(cols, nchecks):
    """Exact short-word counts, Tanner four-cycles and connectivity."""
    parent = list(range(nchecks))

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    checkpairs = {}
    c4 = 0
    for col in cols:
        rows = []
        bits = col
        while bits:
            low = bits & -bits
            rows.append(low.bit_length() - 1)
            bits ^= low
        if rows:
            for r in rows[1:]:
                parent[root(r)] = root(rows[0])
        for t, r in enumerate(rows):
            for s in rows[t + 1:]:
                pair = (r, s)
                old = checkpairs.get(pair, 0)
                c4 += old
                checkpairs[pair] = old + 1

    colset = set(cols)
    repeated = len(cols) - len(colset)
    pairseen = {}
    words3 = words4 = 0
    for i, x in enumerate(cols):
        for y in cols[i + 1:]:
            syndrome = x ^ y
            words3 += int(syndrome in colset)
            old = pairseen.get(syndrome, 0)
            words4 += old
            pairseen[syndrome] = old + 1
    components = len({root(r) for r in range(nchecks)})
    return (repeated, words3, words4), c4, components


def _f21_traps(cols):
    """Small boundaries of three-variable sets; repeated triple syndromes.

    The first statistic is a trapping-set heuristic. The second also
    detects weight-six component words when shorter words are absent.
    Neither statistic is asserted to predict circuit pL exactly.
    """
    boundary = repeated = 0
    seen = {}
    count = len(cols)
    for i in range(count - 2):
        for j in range(i + 1, count - 1):
            pair = cols[i] ^ cols[j]
            for k in range(j + 1, count):
                syndrome = pair ^ cols[k]
                w = syndrome.bit_count()
                if w < 4:
                    boundary += (4 - w) ** 2
                old = seen.get(syndrome, 0)
                repeated += old
                seen[syndrome] = old + 1
    return boundary, repeated


def NEW_CTX(rng, G, idx):
    """Optimize fresh offsets inside a prescribed support/geometry card."""
    q = G.q
    profiles = (
        ((1, 1, 2, 2, 2), (2, 2, 1, 1, 1)),
        ((1, 1, 2, 2, 2), (2, 2, 1, 1, 2)),
        ((0, 3, 1, 2, 1), (3, 0, 2, 1, 2)),
    )
    pairs = ((0, 0), (0, 1), (1, 1), (0, 2))
    architecture = (idx // 5) % len(pairs)
    pa, pb = pairs[architecture]
    sa, sb = profiles[pa], profiles[pb]
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert max(sum(row) for row in sa) + max(
        sum(row[j] for row in sb) for j in range(5)) <= 12
    assert max(sum(row) for row in sb) + max(
        sum(row[j] for row in sa) for j in range(5)) <= 12

    def normalized(values):
        if not values:
            return ()
        return min(tuple(sorted((v - origin) % q for v in values))
                   for origin in values)

    def geometry_key(grids):
        """Remove obvious automorphism, cell-gauge and permutation copies.

        This is an internal-label invariant, not a complete classifier
        of arbitrary Tanner-graph isomorphisms.
        """
        images = []
        for u in range(1, q):
            sides = []
            for grid in grids:
                columns = [
                    tuple(normalized([u * v % q for v in grid[i][j]])
                          for i in range(2))
                    for j in range(5)
                ]
                direct = tuple(sorted(columns))
                swapped = tuple(sorted((b, a) for a, b in columns))
                sides.append(min(direct, swapped))
            images.append(tuple(sorted(sides)))
        return min(images)

    geometry_rng = random.Random(190031 + 1009 * architecture)
    geometry_number = idx % 5 + 5 * (idx // 20)
    book = []
    seen = set()
    for proposal in range(20000):
        grids = []
        for supports in (sa, sb):
            sites = [(i, j) for i in range(2) for j in range(5)
                     if supports[i][j] == 2]
            # Distinct undirected binomial differences prevent repeated
            # internal binomial directions within a component.
            directions = geometry_rng.sample(range(1, (q + 1) // 2),
                                               len(sites))
            steps = dict(zip(sites, directions))
            grid = []
            for i in range(2):
                row = []
                for j in range(5):
                    count = supports[i][j]
                    if count == 0:
                        cell = ()
                    elif count == 1:
                        cell = (0,)
                    elif count == 2:
                        cell = (0, steps[i, j])
                    else:
                        cell = (0,) + tuple(
                            geometry_rng.sample(range(1, q), count - 1))
                    row.append(cell)
                grid.append(row)
            grids.append(grid)
        key = geometry_key(grids)
        if key in seen:
            continue
        seen.add(key)
        book.append(grids)
        if len(book) > geometry_number:
            break
    if len(book) <= geometry_number:
        raise ValueError("insufficient fresh internal-label geometries")
    ga, gb = book[geometry_number]

    def draw(geometry):
        # These independent cell offsets change inter-cell holonomies.
        # No candidate is obtained by relabeling a saved complete code.
        grid = []
        for row in geometry:
            out = []
            for values in row:
                offset = rng.randrange(q)
                out.append([((v + offset) % q,) for v in values])
            grid.append(out)
        return grid

    def kernel_upper(cols, trials=64):
        """Verified kernel witnesses from randomized fundamental circuits.

        Combining two sampled basis words also gives genuine kernel
        words. A large result is not a lower-bound distance certificate.
        """
        order = list(range(len(cols)))
        best = len(cols) + 1
        for _ in range(trials):
            rng.shuffle(order)
            pivots = {}
            basis = []
            for c in order:
                syndrome, word = cols[c], 1 << c
                while syndrome:
                    p = syndrome.bit_length() - 1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = (syndrome, word)
                        break
                    syndrome ^= old[0]
                    word ^= old[1]
                if syndrome == 0 and word:
                    best = min(best, word.bit_count())
                    for previous in basis:
                        best = min(best, (word ^ previous).bit_count())
                    basis.append(word)
            if best <= 6:
                break
        return best

    def low4(checks, stabilizers, n):
        """Exact quantum-logical existence test through weight four."""
        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = row
                    break
                row ^= old

        def nontrivial(word):
            while word:
                p = word.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    return True
                word ^= old
            return False

        columns = [0] * n
        for r, row in enumerate(checks):
            tag = 1 << r
            while row:
                bit = row & -row
                columns[bit.bit_length() - 1] ^= tag
                row ^= bit
        bits = [1 << c for c in range(n)]
        seen = {0: 0}
        for c, syndrome in enumerate(columns):
            old = seen.get(syndrome)
            if old is None:
                seen[syndrome] = bits[c]
            elif nontrivial(bits[c] ^ old):
                return True
        for c in range(n):
            for d in range(c):
                syndrome = columns[c] ^ columns[d]
                word = bits[c] ^ bits[d]
                old = seen.get(syndrome)
                if old is None:
                    seen[syndrome] = word
                elif nontrivial(word ^ old):
                    return True
        return False

    def small_boundary(cols):
        """Exact existence screen through weight six plus triple boundaries.

        If different supports of size <=3 have equal syndromes, their
        symmetric difference is a nonzero component kernel word <=6.
        A passing result is exact only within this short-word window.
        """
        seen = {0}
        for value in cols:
            if value in seen:
                return True, 0
            seen.add(value)
        for i, x in enumerate(cols):
            for j in range(i):
                value = x ^ cols[j]
                if value in seen:
                    return True, 0
                seen.add(value)
        boundary = 0
        for i, x in enumerate(cols):
            for j in range(i):
                pair = x ^ cols[j]
                for h in range(j):
                    value = pair ^ cols[h]
                    if value in seen:
                        return True, boundary
                    seen.add(value)
                    weight = value.bit_count()
                    if weight < 6:
                        boundary += (6 - weight) ** 2
        return False, boundary

    def anchored_exposure(cols):
        """Translation-anchored weight-seven/eight collision exposure.

        Called only after exclusion of component words through six.
        Equal syndromes of distinct four-subsets then have disjoint
        supports and witness weight eight. Four/three collisions witness
        weight seven. Counts are anchored exposures, not exact spectra
        and not a model of ASC ordering or circuit-level pL.
        """
        size = len(cols)
        anchors = tuple(range(0, size, q))
        anchor_set = set(anchors)
        anchored = {}
        for a in anchors:
            smaller = {h for h in anchors if h < a}
            allowed = [h for h in range(size)
                       if h != a and h not in smaller]
            length = len(allowed)
            for bi in range(length - 2):
                b = allowed[bi]
                ab = cols[a] ^ cols[b]
                for ci in range(bi + 1, length - 1):
                    c = allowed[ci]
                    abc = ab ^ cols[c]
                    for di in range(ci + 1, length):
                        d = allowed[di]
                        syndrome = abc ^ cols[d]
                        anchored[syndrome] = anchored.get(syndrome, 0) + 1

        lookup = anchored.get
        seven = eight = 0
        for a in range(size - 2):
            x = cols[a]
            for b in range(a + 1, size - 1):
                ab = x ^ cols[b]
                for c in range(b + 1, size):
                    seven += lookup(ab ^ cols[c], 0)

        for a in range(size - 3):
            x = cols[a]
            for b in range(a + 1, size - 2):
                ab = x ^ cols[b]
                for c in range(b + 1, size - 1):
                    abc = ab ^ cols[c]
                    for d in range(c + 1, size):
                        eight += lookup(abc ^ cols[d], 0)
        # Each anchored support was compared once with itself.
        eight -= sum(anchored.values())
        assert eight >= 0
        return seven, eight

    def component_rank(cols):
        rows = [0] * (2 * q)
        for c, syndrome in enumerate(cols):
            while syndrome:
                bit = syndrome & -syndrome
                rows[bit.bit_length() - 1] |= 1 << c
                syndrome ^= bit
        return gf2_rank(rows)

    for batch in range(2):
        shortlist = []
        for trial in range(96):
            A, B = draw(ga), draw(gb)
            hx, hz, n = lifted_matrices(G, A, B)
            ac, bc = _f21_columns(hx, hz, q)
            ast = _f21_stats(ac, 2 * q)
            bst = _f21_stats(bc, 2 * q)
            if ast[2] != 1 or bst[2] != 1:
                continue
            if component_rank(ac) != 2 * q:
                continue
            if component_rank(bc) != 2 * q:
                continue
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if n != 551 or k != 171:
                continue
            bad = max(ast[0], bst[0])
            totals = tuple(x + y for x, y in zip(ast[0], bst[0]))
            cycles = (max(ast[1], bst[1]), ast[1] + bst[1])
            base = (bad, totals, cycles)
            item = (base, trial, k, A, B, ac, bc, hx, hz, n)
            shortlist.append(item)
            shortlist.sort(key=lambda entry: (entry[0], entry[1]))
            del shortlist[8:]

        finalists = []
        for item in shortlist:
            base, trial, k, A, B, ac, bc, hx, hz, n = item
            bad_a, ba = small_boundary(ac)
            bad_b, bb = small_boundary(bc)
            da, db = kernel_upper(ac, 40), kernel_upper(bc, 40)
            distance = min(da, db)
            key = (int(bad_a) + int(bad_b),
                   int(distance < 8), max(ba, bb), ba + bb,
                   base[2], -min(distance, 10), abs(da - db), trial)
            finalists.append((key, item, not (bad_a or bad_b)))
        finalists.sort(key=lambda entry: entry[0])

        refined = []
        # Refine two alternatives, not an expensive full-spectrum search
        # on every proposal. Bad short-word candidates remain fallback
        # designs, but never outrank a passing short-word candidate.
        for key, item, safe in finalists[:2]:
            _, trial, _, _, _, ac, bc, _, _, _ = item
            if safe:
                ea, eb = anchored_exposure(ac), anchored_exposure(bc)
                exposure = (max(ea[0], eb[0]), ea[0] + eb[0],
                            max(ea[1], eb[1]), ea[1] + eb[1])
            else:
                exposure = (10**12,) * 4
            refined_key = (key[0], key[1], exposure, key[2:], trial)
            refined.append((refined_key, item))
        refined.sort(key=lambda entry: entry[0])

        for key, item in refined:
            _, _, k, A, B, _, _, hx, hz, n = item
            if low4(hx, hz, n) or low4(hz, hx, n):
                continue
            assert css_ok(hx, hz)
            assert max(row.bit_count() for row in hx + hz) <= W_MAX
            return {"idx": idx, "architecture": architecture,
                    "geometry": geometry_number, "k": k,
                    "screen": key, "cells": {"A": A, "B": B}}

    raise ValueError("No connected full-rank C19 design passed screening")


def CELL(rng, G, side, i, j, ctx):
    """Emit the complete screened prime-order cell without redraws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: The parent's large q=58 single-row uniform-label seed reliably suppresses collective circuit failures; measured worst-basis s=3.17-3.28, widespread logical flips and many sector-1 weight-four witnesses contradict this.
# REFUTED: Increasing d2s necessarily improves this bed's circuit score: zero-failure slot 16 had d2s=21,k=23,s=6.874, below zero-failure slot 28 with d2s=18,k=30,s=6.989; their sample ceilings limited both.
# REFUTED: Rank-one C3 quotient singleton/triple fillings reliably retain the earlier order-21 bed's zero-failure memory while raising k: current slot12 had 100/129750 Z failures and s=5.584, slot15 had 37/200000 X failures, and slot18 produced no ASC schedule.
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
