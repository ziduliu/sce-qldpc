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
CONCEPT = ("For cyclic Z13 (metacyclic Z13:Z1, s=1), order 13, shapes (2,5)x(3,8), balanced full-augmentation-rank B redistributions and reduced exact weight-eight component multiplicity should raise worse-basis circuit fitness above 4.9989.")
GROUP = ("Frozen L1 group and order: this exact Z13 bed scores 4.9989. Its size factor 46 permits at most q=13, giving n=598. Keep the group byte-for-byte to test filling rather than infer a group advantage from other beds.")
SHAPES = ("Frozen (2,5)x(3,8) targets cell 0.350: kb=195 and guaranteed rate 15/46=0.326087; verified ranks retain 195<=k<=201. A has 26 checks on 65 variables and B has 39 checks on 104. Parent failures affect all logical qubits, with only diffuse 1.1-1.2x Acol/Bcol enrichment; slot 1 implicates Acol1/Bcol0 and Zchk:Acol2/Brow2 mildly, not a proven isolated hotspot.")
FILLING = ("Ten balanced B architectures redistribute exactly 24 terms: row totals eight, column totals three, and augmentation rank three. Cross with two A support architectures for twenty designs; larger harness budgets extend to six A architectures, including one degree-four column. CSS weights are eleven or twelve and six-round CNOT budgets are 25974 or 26832. Verified lifts screen short kernels and four-cycles; checked cyclic syndrome orbits count weight-eight component words exactly when shorter even kernels are absent. Finalists minimize that multiplicity, then small triple-boundary exposure. Target thousands of weight-eight witnesses and slot 1's 103/7500 X versus 101/33000 Z failures; no structural screen substitutes for circuit pL.")
FAMILY_NAME = "r035_m13r1s1_25x38_balancedB_exact8_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 13

def custom_ops(q):
    # Same cyclic metacyclic construction; factor 46 requires q<=13.
    # Z_13 x| Z_1: y x y^-1 = x^1.
    if q != 13:
        raise ValueError("this bed requires order 13")
    m, r, s = 13, 1, 1
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 5)               # (ra, ca)
B_SHAPE = (3, 8)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """Balanced support contracts followed by verified exact-eight screening."""
    q = G.q
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE

    # First twenty slots cross the first two A architectures with ten
    # distinct B architectures. Additional slots change A's support,
    # rather than manufacture permutation or automorphism copies.
    a_profiles = (
        ((2, 2, 1, 1, 1), (1, 1, 2, 2, 2)),
        ((0, 2, 2, 2, 1), (3, 1, 1, 1, 2)),
        ((3, 0, 2, 1, 1), (0, 3, 1, 2, 2)),
        ((2, 2, 2, 1, 1), (2, 1, 1, 2, 2)),
        ((1, 2, 2, 2, 1), (3, 1, 1, 1, 2)),
        ((0, 2, 2, 2, 2), (4, 1, 1, 1, 1)),
    )
    if idx >= 10 * len(a_profiles):
        raise ValueError("balanced support book exhausted")
    pa = a_profiles[idx // 10]

    # Each directed triangle and reciprocal pair has equal row totals.
    # Pad with (1,1,1) columns to obtain row totals exactly eight.
    # The ten cards differ in their weighted-column architectures;
    # column permutations are not counted as additional designs.
    triangle = ((2, 1, 0), (0, 2, 1), (1, 0, 2))
    reciprocal = (
        ((2, 1, 0), (0, 1, 2)),
        ((2, 0, 1), (0, 2, 1)),
        ((1, 2, 0), (1, 0, 2)),
    )
    b_cards = (
        (0, (1, 1, 0)),
        (1, (0, 0, 0)),
        (0, (2, 1, 0)),
        (0, (1, 1, 1)),
        (1, (1, 0, 0)),
        (2, (0, 0, 0)),
        (1, (2, 0, 0)),
        (1, (1, 1, 0)),
        (2, (1, 0, 0)),
        (0, (2, 1, 1)),
    )
    triangles, pair_counts = b_cards[idx % 10]
    columns = list(triangle) * triangles
    for pair, number in zip(reciprocal, pair_counts):
        columns.extend(list(pair) * number)
    columns.extend([(1, 1, 1)] * (cb - len(columns)))
    assert len(columns) == cb
    columns.sort()
    sb = [[column[i] for column in columns] for i in range(rb)]
    counts = {"A": [list(row) for row in pa], "B": sb}

    assert tuple(map(sum, sb)) == (8, 8, 8)
    assert all(sum(column) == 3 for column in columns)
    augmentation = [
        sum((t & 1) << j for j, t in enumerate(row)) for row in sb
    ]
    assert gf2_rank(augmentation) == rb
    # HX <=8+3=11; HZ <=8+4=12. Every line remains live.
    assert _grid_weights(counts["A"], counts["B"]) <= W_MAX
    assert _no_dead_lines(counts["A"], counts["B"])

    dummy_a = [[[G.e] for _ in range(ca)] for _ in range(ra)]
    dummy_b = [[[G.e] for _ in range(cb)] for _ in range(rb)]
    mask = (1 << q) - 1

    def draw(side):
        return [[rng.sample(G.elements, t) for t in row]
                for row in counts[side]]

    def component_columns(side, grid):
        # Project actual helper-built matrices; do not substitute a lift.
        if side == "A":
            hx, _, _ = lifted_matrices(G, grid, dummy_b)
            cols = [0] * (ca * q)
            for i in range(ra):
                for r in range(q):
                    row = hx[(i * cb) * q + r]  # B-column zero
                    check_bit = 1 << (i * q + r)
                    for j in range(ca):
                        bits = (row >> (j * cb * q)) & mask
                        while bits:
                            bit = bits & -bits
                            cols[j * q + bit.bit_length() - 1] ^= check_bit
                            bits ^= bit
        else:
            _, hz, _ = lifted_matrices(G, dummy_a, grid)
            cols = [0] * (cb * q)
            sector_mask = (1 << (cb * q)) - 1
            for a in range(rb):
                for r in range(q):
                    bits = hz[a * q + r] & sector_mask  # A-column zero
                    check_bit = 1 << (a * q + r)
                    while bits:
                        bit = bits & -bits
                        cols[bit.bit_length() - 1] ^= check_bit
                        bits ^= bit
        expected = [sum(row[j] for row in counts[side])
                    for j in range(len(counts[side][0]))]
        assert all(c.bit_count() == expected[j // q]
                   for j, c in enumerate(cols))
        return cols

    def cycle_data(cols):
        """Exact four-cycle count for either component and mixed degrees."""
        pairs = {}
        adj = [0] * max(c.bit_length() for c in cols)
        conflicts = 0
        for col in cols:
            nodes = []
            bits = col
            while bits:
                bit = bits & -bits
                nodes.append(bit.bit_length() - 1)
                bits ^= bit
            for a, u in enumerate(nodes):
                for v in nodes[a + 1:]:
                    key = (1 << u) | (1 << v)
                    old = pairs.get(key, 0)
                    conflicts += old
                    pairs[key] = old + 1
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
        return conflicts, adj

    def kernel_metric(cols, adj):
        """Short syndrome ambiguities, including odd-weight B witnesses.

        Counts are collision proxies if shorter words exist. They are
        not quantum distances and do not distinguish stabilizers.
        """
        singles = {}
        for syn in cols:
            singles[syn] = singles.get(syn, 0) + 1
        n2 = sum(v * (v - 1) // 2 for v in singles.values())
        pair_hist = {}
        n3 = n4 = 0
        nc = len(cols)
        for a in range(nc - 1):
            for b in range(a + 1, nc):
                syn = cols[a] ^ cols[b]
                n3 += singles.get(syn, 0)
                old = pair_hist.get(syn, 0)
                n4 += old
                pair_hist[syn] = old + 1

        triangles = 0
        for u in range(len(adj)):
            bits = adj[u] & ~((1 << (u + 1)) - 1)
            while bits:
                bit = bits & -bits
                v = bit.bit_length() - 1
                triangles += (
                    adj[u] & adj[v] & ~((1 << (v + 1)) - 1)
                ).bit_count()
                bits ^= bit
        stars = sum(c.bit_count() * (c.bit_count() - 1)
                    * (c.bit_count() - 2) // 6 for c in cols)
        # Exact Tanner six-cycles here because callers require no C4.
        n6cycles = triangles - stars
        if n2 or n3 or n4:
            return (n2, n3, n4, 10**9, 10**9, n6cycles)

        triple_hist = {}
        n5 = n6 = 0
        for a in range(nc - 2):
            for b in range(a + 1, nc - 1):
                ab = cols[a] ^ cols[b]
                for c in range(b + 1, nc):
                    syn = ab ^ cols[c]
                    n5 += pair_hist.get(syn, 0)
                    old = triple_hist.get(syn, 0)
                    n6 += old
                    triple_hist[syn] = old + 1
        return (0, 0, 0, n5, n6, n6cycles)

    def rows_from_columns(cols):
        """Transpose the columns extracted from the verified lift."""
        rows = [0] * max(c.bit_length() for c in cols)
        for j, col in enumerate(cols):
            while col:
                bit = col & -col
                rows[bit.bit_length() - 1] |= 1 << j
                col ^= bit
        return rows

    def kernel_probe(cols, trials):
        """Random systematic bases give verified component witnesses only."""
        source = rows_from_columns(cols)
        nc = len(cols)
        minimum = nc + 1
        small = set()
        for _ in range(trials):
            rows = list(source)
            order = list(range(nc))
            rng.shuffle(order)
            pivots = []
            rank = 0
            for c in order:
                bit = 1 << c
                hit = next((t for t in range(rank, len(rows))
                            if rows[t] & bit), None)
                if hit is None:
                    continue
                rows[rank], rows[hit] = rows[hit], rows[rank]
                pivot = rows[rank]
                for t in range(len(rows)):
                    if t != rank and rows[t] & bit:
                        rows[t] ^= pivot
                pivots.append(c)
                rank += 1
                if rank == len(rows):
                    break
            occupied = set(pivots)
            for c in order:
                if c in occupied:
                    continue
                word = 1 << c
                for p, row in zip(pivots, rows[:rank]):
                    if (row >> c) & 1:
                        word |= 1 << p
                weight = word.bit_count()
                minimum = min(minimum, weight)
                if weight <= 10:
                    small.add(word)
            if minimum <= 6:
                break
        return minimum, len(small)

    # These are relabelings of already verified syndromes, not another
    # implementation of the lift. orbit_eight explicitly checks their
    # action against every extracted component column.
    rotations = [
        [((v << t) & mask) | (v >> (q - t))
         for v in range(mask + 1)]
        for t in range(q)
    ]
    least_shift = [
        min(range(q), key=lambda t: rotations[t][v])
        for v in range(mask + 1)
    ]

    def translated(value, shift, blocks):
        return sum(
            rotations[shift][(value >> (r * q)) & mask] << (r * q)
            for r in range(blocks)
        )

    def orbit_eight(cols, side):
        """Exact component weight-eight count, assuming no kernels <=6.

        Enumerate one representative of every four-support translation
        orbit. Nonconstant syndromes have free orbits because q=13 is
        prime. Equal canonical syndromes give thirteen pair collisions;
        each weight-eight word has 35 unordered four/four partitions.
        """
        nc = len(cols)
        blocks = len(counts[side])
        assert q == 13
        for j in range(nc // q):
            seed = cols[j * q]
            assert all(
                cols[j * q + t] == translated(seed, t, blocks)
                for t in range(q)
            ), "syndrome translation disagrees with verified component"

        bits = [1 << j for j in range(nc)]
        histogram = {}
        orbit_collisions = 0
        for anchor in range(0, nc, q):
            first_block = anchor // q
            for a in range(anchor + 1, nc - 2):
                sa = cols[anchor] ^ cols[a]
                for b in range(a + 1, nc - 1):
                    sab = sa ^ cols[b]
                    for c in range(b + 1, nc):
                        vertices = (anchor, a, b, c)
                        word = bits[anchor] | bits[a] | bits[b] | bits[c]

                        # Every orbit has a representative with lift zero
                        # in its first occupied block. Choose one of those
                        # representatives canonically, without gauge copies.
                        canonical = True
                        for v in vertices[1:]:
                            if v // q != first_block:
                                break
                            shift = v % q
                            image = sum(
                                bits[(u // q) * q + (u % q - shift) % q]
                                for u in vertices
                            )
                            if image < word:
                                canonical = False
                                break
                        if not canonical:
                            continue

                        syndrome = sab ^ cols[c]
                        shift = None
                        for r in range(blocks - 1, -1, -1):
                            part = (syndrome >> (r * q)) & mask
                            if part not in (0, mask):
                                shift = least_shift[part]
                                break
                        # An invariant four-support syndrome would imply
                        # a shorter even kernel through overlapping shifts.
                        # Return a conservative screen penalty if encountered.
                        if shift is None:
                            return 10**12
                        key = translated(syndrome, shift, blocks)
                        previous = histogram.get(key, 0)
                        orbit_collisions += previous
                        histogram[key] = previous + 1

        assert orbit_collisions % 35 == 0
        return q * (orbit_collisions // 35)

    def boundary_exposure(cols):
        """Worst-block anchored triple-boundary exposure, a BP surrogate.

        Each component column block supplies one lift-zero anchor.
        This is not a circuit-failure measurement or logical-distance
        claim; the parent maps motivate balancing blocks globally.
        """
        costs = []
        for anchor in range(0, len(cols), q):
            total = 0
            for a in range(len(cols)):
                if a == anchor:
                    continue
                pair = cols[anchor] ^ cols[a]
                for b in range(a):
                    if b == anchor:
                        continue
                    weight = (pair ^ cols[b]).bit_count()
                    if weight <= 4:
                        total += (5 - weight) ** 2
            costs.append(total)
        return max(costs), sum(costs)

    def signature(grid):
        return tuple(tuple(sorted(G.idx(g) for g in cell))
                     for row in grid for cell in row)

    def optimize(side):
        sites = [(i, j) for i, row in enumerate(counts[side])
                 for j, t in enumerate(row) if t]
        number_rows = len(counts[side])
        augmentation = [
            sum((t % 2) << j for j, t in enumerate(row))
            for row in counts[side]
        ]
        # For q=13, ord_13(2)=12. Deficits at the trivial character
        # are unavoidable; penalize additional measured rank loss.
        allowed_deficit = number_rows - gf2_rank(augmentation)

        def rank_penalty(columns):
            deficit = number_rows * q - gf2_rank(
                rows_from_columns(columns))
            return max(0, deficit - allowed_deficit)

        current = draw(side)
        cols = component_columns(side, current)
        cost, adj = cycle_data(cols)
        fallback, fallback_cost = current, cost
        shortlist = []
        tested = set()

        for step in range(2200):
            if cost == 0:
                sig = signature(current)
                if sig not in tested:
                    tested.add(sig)
                    metric = ((rank_penalty(cols),)
                              + kernel_metric(cols, adj))
                    shortlist.append((metric, sig, current, cols))
                    shortlist.sort(key=lambda item: (item[0], item[1]))
                    shortlist = shortlist[:4]
                    if (len(tested) >= 20
                            and shortlist[0][0][:6] == (0, 0, 0, 0, 0, 0)):
                        break

            i, j = sites[rng.randrange(len(sites))]
            trial = [list(row) for row in current]
            trial[i][j] = rng.sample(G.elements, counts[side][i][j])
            trial_cols = component_columns(side, trial)
            trial_cost, trial_adj = cycle_data(trial_cols)
            if trial_cost < fallback_cost:
                fallback, fallback_cost = trial, trial_cost

            # Neutral moves change voltage geometry, not just gauges.
            if trial_cost <= cost or rng.random() < 0.015:
                current, cols = trial, trial_cols
                cost, adj = trial_cost, trial_adj
            if step % 240 == 239 and cost:
                current = draw(side)
                cols = component_columns(side, current)
                cost, adj = cycle_data(cols)
                if cost < fallback_cost:
                    fallback, fallback_cost = current, cost

        if not shortlist:
            return fallback

        best = None
        for metric, sig, grid, columns in shortlist:
            # The collision formula requires absence of shorter even
            # kernels. kernel_metric supplies exact zero tests here.
            # Keep short ambiguities and measured excess rank loss first.
            eight = (orbit_eight(columns, side)
                     if not any(metric[1:6]) else 10**12)
            exposure = boundary_exposure(columns)
            quality = metric[:-1] + (eight,) + exposure + (metric[-1],)
            if best is None or quality < best[0]:
                best = quality, grid
        return best[1]

    # Retry label optimization, never alter an illegal submitted slot.
    # Every B contract now has full augmentation rank. Retain the
    # target band using actual CSS ranks, not a quotient prediction.
    for attempt in range(3):
        cells = {"A": optimize("A"), "B": optimize("B")}
        hx, hz, n = lifted_matrices(G, cells["A"], cells["B"])
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (195 <= k <= 201):
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified CSS commutation failed")
        assert n == 598
        assert max(row.bit_count() for row in hx + hz) <= W_MAX
        return {"idx": idx, "cells": cells, "k": k}
    raise ValueError("no verified filling in the target rate interval")


def CELL(rng, G, side, i, j, ctx):
    """Return the precomputed full cell without changing shared state."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Equal (n,k,d2s) determines circuit performance: parent Z17 slots with (578,204,5) scored 3.691-4.347, and slot 50 was strongly Z-limited despite its weight-seven Z-type witnesses.
# REFUTED: Weight-eight witnesses in both bases imply balanced memory reliability: current parent slot 20 had 100/8250 Z failures versus 100/50250 X failures, with only diffuse block enrichment.
# REFUTED: All five degree-three A partitions retain weight-eight protection on this Z13 bed: slots 12-19 had d2s=4, whereas slots 0-3 had d2s=8.
# REFUTED: Extra B doubletons and d2s=8 suffice for balanced memory: slot 1 failed 103/7500 X versus 101/33000 Z shots, with worse-basis s=4.851; this does not isolate term count from labels.
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
