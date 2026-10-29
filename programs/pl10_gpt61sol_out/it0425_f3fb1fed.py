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
CONCEPT = ("C8 x C2, order 16, shapes (2,5)x(2,6): full-augmentation-rank, degree-balanced components and reduced extraction exposure should exceed the cell-0.350 incumbent fitness 5.1107. Fewer measured circuit failures, not larger sampled distance, decide this hypothesis.")
GROUP = ("Jump from C13 to the genuinely different abelian group C8 x C2 of order 16. This bed is absent from BEDS; nearby order-16 (2,5)x(2,6) beds score about 4.6-4.7, so no group advantage is assumed. The size factor 34 admits this lift at n=544. Full augmentation rank over this characteristic-two group algebra motivates retaining the rank-count rate, but actual helper-built CSS ranks decide.")
SHAPES = ("Target cell 0.350 with (2,5)x(2,6): n=544, kb=192, guaranteed rate 6/17=0.352941, and require measured k=192. A has 80 variables/32 checks; B has 96 variables/32 checks, versus the parent's 65/26 and 104/39 components. Parent failures touched all 195 logicals, with diffuse 1.1-1.3x Acol/Bcol and mild Acol/Brow enrichment; logical #138 reached 56 flips in one X experiment. These observations motivate balanced global protection, not a claimed isolated block repair.")
FILLING = ("Twenty briefs cross four eighteen-term B support architectures with five canonical relative-support briefs; additional harness batches consume further non-orbit ratio cards. A has fifteen terms and degree-three columns. B rows total nine, with column degrees two through four. HX<=8+4=12 and HZ<=9+3=12. Six-round CNOTs are 23616 instead of 25974. Relative supports exclude involutory differences and repeated internal directions within each row, and generate the full group rather than a cyclic subgroup. Singleton actions come only from verified lifts. Component cycle, short-kernel and capped witness screens retain alternatives; complete CSS ranks, stabilizer-aware weight-four classification and two-interval noisy-syndrome BP with independent holdouts select the pairing. This attenuated benchmark omits ASC ordering, six-round history and relay/LSD; it is not measured pL. Target the parent's five failed-shot caps and X failure fractions near 0.6-0.9 percent, not a distance-only improvement.")
FAMILY_NAME = "r035_C8xC2_q16_25x26_balanced_holdout_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 16

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C8 x C2 requires order 16")
    els = [(a, b) for b in range(2) for a in range(8)]

    def red(g):
        return (int(g[0]) % 8, int(g[1]) % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + c) % 8, (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % 8, (-b) % 2)

    def idx(g):
        a, b = red(g)
        return a + 8*b

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 5)               # (ra, ca)
B_SHAPE = (2, 6)               # (rb, cb)


def _retired_C13_exact8_ctx(rng, G, idx):
    """Inactive historical C13 screen; never called on the new group."""
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


def NEW_CTX(rng, G, idx):
    """Fresh relative-support cards and verified full-pair holdout screening."""
    import math
    from itertools import combinations

    q = G.q
    assert q == 16
    architecture, marker = divmod(idx % 20, 5)
    batch = idx // 20
    sa = [[2, 2, 1, 1, 1], [1, 1, 2, 2, 2]]
    profiles = (
        ((2,1), (2,1), (2,1), (1,2), (1,2), (1,2)),
        ((1,1), (2,2), (2,1), (2,1), (1,2), (1,2)),
        ((1,1), (1,1), (2,2), (2,2), (2,1), (1,2)),
        ((3,1), (1,2), (1,2), (2,1), (1,2), (1,1)),
    )
    profile = tuple(sorted(profiles[architecture]))
    sb = [[column[i] for column in profile] for i in range(2)]
    counts = {"A": sa, "B": sb}
    assert tuple(map(sum, sb)) == (9, 9)
    assert sum(map(sum, sa)) == 15
    assert sum(map(sum, sb)) == 18
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)
    for grid in counts.values():
        assert gf2_rank([
            sum((t & 1) << j for j, t in enumerate(row))
            for row in grid
        ]) == 2

    def direction(g):
        return min(G.idx(g), G.idx(G.inv(g)))

    def order(g):
        power = G.e
        for t in range(1, q + 1):
            power = G.mul(power, g)
            if power == G.e:
                return t
        raise ValueError("validated group has no element order")

    orders = [order(g) for g in G.elements]

    def internal(cell):
        return [
            direction(G.mul(G.inv(a), b))
            for a, b in combinations(cell, 2)
        ]

    def admissible(cell):
        ds = internal(cell)
        return (len(set(ds)) == len(ds)
                and all(orders[d] > 2 for d in ds))

    directions = [
        g for g in G.elements
        if orders[G.idx(g)] > 2 and G.idx(g) == direction(g)
    ]
    triples = [
        [G.e, u, v]
        for u, v in combinations(G.elements[1:], 2)
        if admissible([G.e, u, v])
    ]
    assert directions and triples

    def normal(cell):
        if not cell:
            return ()
        return min(
            tuple(sorted(G.idx(G.mul(G.inv(origin), g))
                         for g in cell))
            for origin in cell
        )

    def geometry_key(grids):
        """Quotient actual group automorphisms and row/column images.

        This canonicalizes ratio cards, not arbitrary full Tanner graphs.
        No completed code is transformed to manufacture another slot.
        """
        images = []
        for u in (1, 3, 5, 7):
            for c in (0, 4):
                for w in (0, 1):
                    sides = []
                    for grid in grids:
                        columns = []
                        for j in range(len(grid[0])):
                            cells = []
                            for i in range(2):
                                image = [
                                    ((u*g[0] + c*g[1]) % 8,
                                     (w*g[0] + g[1]) % 2)
                                    for g in grid[i][j]
                                ]
                                cells.append(normal(image))
                            columns.append(tuple(cells))
                        direct = tuple(sorted(columns))
                        swapped = tuple(sorted((b, a)
                                               for a, b in columns))
                        sides.append(min(direct, swapped))
                    images.append(tuple(sides))
        return min(images)

    def generates_group(grid):
        generators = [
            G.mul(G.inv(cell[0]), g)
            for row in grid for cell in row if cell
            for g in cell[1:]
        ]
        reached = {G.e}
        frontier = [G.e]
        while frontier:
            a = frontier.pop()
            for g in generators:
                h = G.mul(a, g)
                if h not in reached:
                    reached.add(h)
                    frontier.append(h)
        return len(reached) == q

    # Order-four doubleton counts change the relative-support brief,
    # rather than submitting new cell-origin draws of one geometry.
    briefs = ((0, 0), (0, 2), (1, 1), (2, 2), (2, 3))
    catalogue_rng = random.Random(816256031 + 1009 * architecture)
    wanted = 5 * batch + marker
    book, seen = [], set()

    def template(side):
        grid = []
        for row in counts[side]:
            cells = [[] for _ in row]
            occupied = set()
            for j in sorted(range(len(row)), key=lambda j: -row[j]):
                t = row[j]
                if t == 1:
                    candidates = [[G.e]]
                elif t == 2:
                    candidates = [
                        [G.e, d] for d in directions
                        if direction(d) not in occupied
                    ]
                elif t == 3:
                    candidates = [
                        cell for cell in triples
                        if not (set(internal(cell)) & occupied)
                    ]
                else:
                    candidates = [[]] if t == 0 else []
                if not candidates:
                    return None
                cells[j] = list(catalogue_rng.choice(candidates))
                occupied.update(internal(cells[j]))
            grid.append(cells)
        return grid

    for proposal in range(40000):
        grids = [template("A"), template("B")]
        if any(grid is None for grid in grids):
            continue
        charged = tuple(
            sum(len(cell) == 2
                and orders[direction(
                    G.mul(G.inv(cell[0]), cell[1]))] == 4
                for row in grid for cell in row)
            for grid in grids
        )
        if charged != briefs[len(book) % 5]:
            continue
        if not all(generates_group(grid) for grid in grids):
            continue
        key = geometry_key(grids)
        if key in seen:
            continue
        seen.add(key)
        book.append(grids)
        if len(book) > wanted:
            geometry = {"A": grids[0], "B": grids[1]}
            break
    else:
        raise ValueError("fresh canonical relative-support book exhausted")

    def transpose(rows, width):
        columns = [0] * width
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                columns[bit.bit_length() - 1] ^= 1 << r
                row ^= bit
        return columns

    def verified_component(side, grid):
        rows = (lifted_matrices(G, grid, [[[]]])[0]
                if side == "A" else
                lifted_matrices(G, [[[]]], grid)[1])
        keep = (1 << (len(grid[0]) * q)) - 1
        return [row & keep for row in rows]

    # Cache only actions returned by the verified construction helper.
    # Composition below is checked again on each selected component.
    actions = {}
    for side in ("A", "B"):
        actions[side] = [
            transpose(verified_component(side, [[[g]]]), q)
            for g in G.elements
        ]

    def component_columns(side, grid):
        columns = [0] * (len(grid[0]) * q)
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                for g in cell:
                    block = actions[side][G.idx(g)]
                    for t, value in enumerate(block):
                        columns[j*q+t] ^= value << (i*q)
        return columns

    def draw_cell(side, i, j):
        origin = G.pick(rng)
        return [G.mul(origin, g) for g in geometry[side][i][j]]

    def draw(side):
        return [[draw_cell(side, i, j) for j in range(len(row))]
                for i, row in enumerate(counts[side])]

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def signature(grid):
        return tuple(tuple(sorted(G.idx(g) for g in cell))
                     for row in grid for cell in row)

    def cycles(columns):
        pairs = {}
        adjacency = [0] * (2*q)
        four = 0
        stars = 0
        for column in columns:
            nodes = []
            while column:
                bit = column & -column
                nodes.append(bit.bit_length() - 1)
                column ^= bit
            d = len(nodes)
            stars += d * (d-1) * (d-2) // 6
            for a, b in combinations(nodes, 2):
                key = (a, b)
                old = pairs.get(key, 0)
                four += old
                pairs[key] = old + 1
                adjacency[a] |= 1 << b
                adjacency[b] |= 1 << a
        triangles = 0
        for a in range(len(adjacency)):
            bits = adjacency[a] & ~((1 << (a+1)) - 1)
            while bits:
                bit = bits & -bits
                b = bit.bit_length() - 1
                triangles += (
                    adjacency[a] & adjacency[b]
                    & ~((1 << (b+1)) - 1)
                ).bit_count()
                bits ^= bit
        # Exact at C4=0; otherwise only an annealing heuristic.
        return four, max(0, triangles - stars)

    def kernel_statistics(columns):
        """Collision counts and small boundaries, not quantum distances."""
        singles = {}
        one = two = three = four = 0
        for value in columns:
            one += int(value == 0)
            two += singles.get(value, 0)
            singles[value] = singles.get(value, 0) + 1
        pairs = {}
        for i, value in enumerate(columns):
            for other in columns[:i]:
                syndrome = value ^ other
                three += singles.get(syndrome, 0)
                four += pairs.get(syndrome, 0)
                pairs[syndrome] = pairs.get(syndrome, 0) + 1
        short = (one, two, three, four)
        if any(short):
            return short + (10**12, 10**12, 10**12)
        triples_hist = {}
        five = six = boundary = 0
        for a, b, c in combinations(range(len(columns)), 3):
            syndrome = columns[a] ^ columns[b] ^ columns[c]
            five += pairs.get(syndrome, 0)
            six += triples_hist.get(syndrome, 0)
            triples_hist[syndrome] = triples_hist.get(syndrome, 0) + 1
            weight = syndrome.bit_count()
            if weight <= 4:
                boundary += (5-weight) ** 2
        return short + (five, six, boundary)

    def sampled_words(columns):
        """Actual fundamental/pair-XOR witnesses; minimum is an upper bound."""
        permutation = list(range(len(columns)))
        minimum = len(columns) + 1
        small = set()

        def offer(word):
            nonlocal minimum
            if word:
                weight = word.bit_count()
                minimum = min(minimum, weight)
                if weight <= 8 and len(small) < 4096:
                    small.add(word)

        for trial in range(32):
            rng.shuffle(permutation)
            pivots, basis = {}, []
            for c in permutation:
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
                    offer(word)
            for sweep in range(2):
                rng.shuffle(basis)
                for i, word in enumerate(basis):
                    weight = word.bit_count()
                    for other in basis[:i]:
                        proposal = word ^ other
                        offer(proposal)
                        pw = proposal.bit_count()
                        if proposal and pw < weight:
                            word, weight = proposal, pw
                    basis[i] = word
            if minimum <= 4:
                break
        return minimum, len(small)

    def portfolio(side):
        pool, tested = [], set()
        positions = [
            (i, j) for i, row in enumerate(counts[side])
            for j, t in enumerate(row) if t
        ]

        def retain(grid, columns, stats):
            sig = signature(grid)
            if sig in tested:
                return
            tested.add(sig)
            pool.append((stats, rng.random(), clone(grid), list(columns)))
            pool.sort(key=lambda item: item[:2])
            del pool[8:]

        for restart in range(4):
            grid = draw(side)
            columns = component_columns(side, grid)
            stats = cycles(columns)
            energy = (60*stats[0] + stats[1]/4) / q
            retain(grid, columns, stats)
            local_min = energy
            for step in range(420):
                proposal = clone(grid)
                i, j = rng.choice(positions)
                proposal[i][j] = draw_cell(side, i, j)
                pc = component_columns(side, proposal)
                ps = cycles(pc)
                pe = (60*ps[0] + ps[1]/4) / q
                delta = pe - energy
                temperature = 12 * (0.987 ** step) + 0.10
                if delta <= 0 or rng.random() < math.exp(
                        -min(700.0, delta / temperature)):
                    grid, columns, stats, energy = proposal, pc, ps, pe
                improved = energy < local_min
                local_min = min(local_min, energy)
                if improved or step % 24 == 0:
                    retain(grid, columns, stats)
            retain(grid, columns, stats)

        finalists = []
        for stats, tie, grid, columns in pool:
            rows = transpose(columns, 2*q)
            # Here transpose(columns, 2*q) returns the component bit rows.
            deficit = 2*q - gf2_rank(rows)
            spectrum = kernel_statistics(columns)
            minimum, multiplicity = sampled_words(columns)
            guard = ((deficit,) + spectrum[:4]
                     + (int(spectrum[4] > 0), int(spectrum[5] > 0)))
            key = guard + (
                max(0, 8-minimum), multiplicity,
                spectrum[-1], stats[0], stats[1], tie,
            )
            finalists.append({
                "grid": grid, "key": key, "guard": guard,
                "cycles": stats, "minimum": minimum,
            })
        finalists.sort(key=lambda item: item["key"])
        return finalists[:3]

    def span(rows):
        pivots = {}
        for row in rows:
            while row:
                p = row.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = row
                    break
                row ^= old
        return pivots

    def nontrivial(word, pivots):
        while word:
            p = word.bit_length() - 1
            old = pivots.get(p)
            if old is None:
                return True
            word ^= old
        return False

    def short_logical(hx, hz, n):
        """Exact detection of non-stabilizer logicals through weight four."""
        for checks, stabilizers in ((hx, hz), (hz, hx)):
            columns = transpose(checks, n)
            pivots = span(stabilizers)
            seen_syndromes = {0: 0}
            for i, syndrome in enumerate(columns):
                word = 1 << i
                old = seen_syndromes.get(syndrome)
                if old is None:
                    seen_syndromes[syndrome] = word
                elif nontrivial(word ^ old, pivots):
                    return True
            for i in range(n):
                for j in range(i):
                    syndrome = columns[i] ^ columns[j]
                    word = (1 << i) | (1 << j)
                    old = seen_syndromes.get(syndrome)
                    if old is None:
                        seen_syndromes[syndrome] = word
                    elif nontrivial(word ^ old, pivots):
                        return True
        return False

    def full_four_cycles(rows):
        return sum(d * (d-1) // 2
                   for i, row in enumerate(rows)
                   for other in rows[:i]
                   for d in [(row & other).bit_count()])

    def noisy_probe(checks, stabilizers, n, seed, shots):
        """Two attenuated fault intervals, NOT an ASC circuit experiment.

        Equations are H*e0+m=s0 and H*(e0+e1)=sfinal.
        The intermediate syndrome is noisy and the final one perfect.
        Random opposite-check suffixes approximate correlated faults.
        Valid stabilizer-equivalent residuals count as successes.
        """
        import numpy as np

        m = len(checks)
        width = 2*n + m
        rows = [
            row | (1 << (2*n+r))
            for r, row in enumerate(checks)
        ] + [row | (row << n) for row in checks]
        columns = transpose(rows, width)
        edge_variables = []
        cn = [[] for _ in rows]
        vn = [[] for _ in range(width)]
        for v, column in enumerate(columns):
            while column:
                bit = column & -column
                c = bit.bit_length() - 1
                edge = len(edge_variables)
                edge_variables.append(v)
                cn[c].append(edge)
                vn[v].append(edge)
                column ^= bit
        ne = len(edge_variables)
        ev = np.asarray(edge_variables, dtype=np.int64)
        ce = np.full((len(rows), max(map(len, cn))), ne, dtype=np.int64)
        ve = np.full((width, max(map(len, vn))), ne, dtype=np.int64)
        for c, edges in enumerate(cn):
            ce[c, :len(edges)] = edges
        for v, edges in enumerate(vn):
            ve[v, :len(edges)] = edges
        real = ce != ne
        check_variables = np.asarray(
            edge_variables + [width], dtype=np.int64)[ce]
        pivots = span(stabilizers)
        channel = np.random.default_rng(seed)
        intensity = np.asarray((0.25, 0.45))[np.arange(shots) % 2]
        degree = np.asarray([
            c.bit_count() for c in transpose(checks + stabilizers, n)
        ], dtype=float)
        single_mu = 3 * 0.002 * (2.0/3.0) * degree
        data_p = 0.5 * (
            1.0 - np.exp(-2 * intensity[:, None] * single_mu))
        errors = np.zeros((shots, width+1), dtype=bool)
        errors[:, :n] = channel.random((shots, n)) < data_p
        errors[:, n:2*n] = channel.random((shots, n)) < data_p

        hook_load = np.zeros(n, dtype=float)
        for row in stabilizers:
            sites = []
            while row:
                bit = row & -row
                sites.append(bit.bit_length() - 1)
                row ^= bit
            if len(sites) < 2:
                continue
            hook_load[sites] += len(sites) + 1
            ordering = np.asarray(sites, dtype=np.int64)
            channel.shuffle(ordering)
            event_p = intensity * 3 * 0.002 * (len(sites)+1) / 3
            for interval in range(2):
                events = np.flatnonzero(channel.random(shots) < event_p)
                for shot in events:
                    cut = int(channel.integers(1, len(sites)))
                    errors[shot, interval*n + ordering[cut:]] ^= True

        measurement_mu = 3 * 0.002 * (2.0/3.0) * np.asarray([
            row.bit_count()+2 for row in checks
        ], dtype=float)
        measurement_p = 0.5 * (
            1.0 - np.exp(-2 * intensity[:, None] * measurement_mu))
        errors[:, 2*n:width] = (
            channel.random((shots, m)) < measurement_p)
        marginal_mu = single_mu + 3 * 0.002 * hook_load / 6
        marginal_p = np.clip(0.5 * (
            1.0 - np.exp(-2 * intensity[:, None] * marginal_mu)),
            0.003, 0.18)
        probabilities = np.concatenate(
            (marginal_p, marginal_p,
             np.clip(measurement_p, 0.003, 0.18)), axis=1)
        prior = np.log((1-probabilities) / probabilities)
        syndrome = np.logical_xor.reduce(
            errors[:, check_variables], axis=2)
        signs = 1.0 - 2.0 * syndrome.astype(float)
        messages = np.empty((shots, ne+1), dtype=float)
        messages[:, :ne] = prior[:, ev]
        messages[:, ne] = 30.0
        outgoing = np.zeros_like(messages)
        active = np.ones(shots, dtype=bool)
        ones = np.ones((shots, len(rows), 1), dtype=float)
        wrong = work = 0
        early = None

        for iteration in range(40):
            work += int(active.sum())
            values = np.tanh(np.clip(messages[:, ce], -18, 18) / 2)
            values[:, ~real] = 1.0
            left = np.concatenate(
                (ones, np.cumprod(values, axis=2)[:, :, :-1]), axis=2)
            suffix = np.cumprod(values[:, :, ::-1], axis=2)[:, :, ::-1]
            right = np.concatenate((suffix[:, :, 1:], ones), axis=2)
            update = 2 * np.arctanh(np.clip(
                signs[:, :, None] * left * right,
                -0.999999, 0.999999))
            outgoing[:, ce[real]] = update[:, real]
            posterior = prior + outgoing[:, ve].sum(axis=2)
            estimate = np.concatenate(
                (posterior < 0, np.zeros((shots, 1), dtype=bool)), axis=1)
            matched = np.logical_xor.reduce(
                estimate[:, check_variables], axis=2)
            settled = active & np.all(matched == syndrome, axis=1)
            for shot in np.flatnonzero(settled):
                residual = (
                    errors[shot, :n] ^ errors[shot, n:2*n]
                    ^ estimate[shot, :n] ^ estimate[shot, n:2*n])
                word = sum(1 << int(v) for v in np.flatnonzero(residual))
                wrong += int(nontrivial(word, pivots))
            active[settled] = False
            if iteration == 11:
                early = int(active.sum())
            if not active.any():
                break
            damping = 0.25 if iteration < 12 else 0.50
            proposal = posterior[:, ev] - outgoing[:, :ne]
            messages[:, :ne] = (
                damping * messages[:, :ne] + (1-damping) * proposal)

        remaining = int(active.sum())
        if early is None:
            early = remaining
        return wrong + remaining, remaining, early, work

    def probe_key(x, z):
        return (max(x[0], z[0]), x[0]+z[0],
                max(x[1], z[1]), x[1]+z[1],
                max(x[2], z[2]), x[3]+z[3])

    apool, bpool = portfolio("A"), portfolio("B")
    prepared = []
    for a in apool:
        for b in bpool:
            A, B = a["grid"], b["grid"]
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if n != 544 or k != 192:
                continue
            guard = tuple(max(x, y)
                          for x, y in zip(a["guard"], b["guard"]))
            fx, fz = full_four_cycles(hx), full_four_cycles(hz)
            coarse = guard + (
                max(0, 8-a["minimum"]) + max(0, 8-b["minimum"]),
                max(fx/len(hx), fz/len(hz)), fx+fz,
                a["key"], b["key"],
            )
            prepared.append((coarse, len(prepared), A, B, hx, hz))
    if not prepared:
        raise ValueError("relative-support card missed verified k=192")
    prepared.sort(key=lambda item: item[:2])

    coarse_results = []
    seed = 816544031 + 1009*idx
    for entry in prepared[:4]:
        coarse, number, A, B, hx, hz = entry
        bad = int(short_logical(hx, hz, 544))
        x = noisy_probe(hx, hz, 544, seed, 48)
        z = noisy_probe(hz, hx, 544, seed+1, 48)
        key = (bad,) + coarse[:7] + probe_key(x, z) + coarse[7:]
        coarse_results.append((key, number, entry, bad, x, z))
    coarse_results.sort(key=lambda item: item[:2])

    winner = None
    for _, number, entry, bad, cx, cz in coarse_results[:2]:
        coarse, _, A, B, hx, hz = entry
        x = noisy_probe(hx, hz, 544, seed+1000003, 96)
        z = noisy_probe(hz, hx, 544, seed+1000004, 96)
        total_x = tuple(u+v for u, v in zip(cx, x))
        total_z = tuple(u+v for u, v in zip(cz, z))
        key = ((bad,) + coarse[:7] + probe_key(total_x, total_z)
               + coarse[7:] + (number,))
        if winner is None or key < winner[0]:
            winner = key, entry, total_x, total_z

    key, entry, total_x, total_z = winner
    _, _, A, B, hx, hz = entry
    assert css_ok(hx, hz)
    assert lifted_k(G, A, B) == 192
    assert max(row.bit_count() for row in hx + hz) <= W_MAX
    assert 6 * sum(row.bit_count() for row in hx + hz) == 23616
    assert 0.325 <= 192/544 < 0.375
    for side, grid in (("A", A), ("B", B)):
        columns = component_columns(side, grid)
        assert columns == transpose(
            verified_component(side, grid), len(columns))
        assert all(len(cell) <= 3 and len(set(cell)) == len(cell)
                   for row in grid for cell in row)
    return {
        "idx": idx, "cells": {"A": A, "B": B}, "k": 192,
        "contract": (architecture, marker, batch),
        "relative_geometry": geometry, "screen": key,
        "noisy_syndrome_holdout": (total_x, total_z),
        "cnots": 23616,
    }


def CELL(rng, G, side, i, j, ctx):
    """Emit the selected full cell without late mutation or randomization."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Equal (n,k,d2s) determines circuit performance: parent Z17 slots with (578,204,5) scored 3.691-4.347, and slot 50 was strongly Z-limited despite its weight-seven Z-type witnesses.
# REFUTED: Weight-eight witnesses in both bases imply balanced memory reliability: current parent slot 20 had 100/8250 Z failures versus 100/50250 X failures, with only diffuse block enrichment.
# REFUTED: All five degree-three A partitions retain weight-eight protection on this Z13 bed: slots 12-19 had d2s=4, whereas slots 0-3 had d2s=8.
# REFUTED: Extra B doubletons and d2s=8 suffice for balanced memory: slot 1 failed 103/7500 X versus 101/33000 Z shots, with worse-basis s=4.851; this does not isolate term count from labels.
# REFUTED: Exact weight-eight component multiplicity screening suffices for balanced near-ceiling C13 memory: fitness was 5.0888, all five measured codes hit both failed-shot caps, and X failures reached 100-101 in 10750-16000 shots versus 100-101 in 25250-32500 Z shots.
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
