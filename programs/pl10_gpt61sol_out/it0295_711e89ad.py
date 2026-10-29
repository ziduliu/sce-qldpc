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
CONCEPT = ("For cyclic C19, order 19, shapes (2,4)x(3,6), equally half-rate degree-three components with balanced short-ambiguity screening should improve mean worse-basis circuit fitness above the parent's 7.0514.")
GROUP = ("Preserve the cyclic construction rule under L2, changing only its order parameter. The new size factor thirty requires q<20; C19 gives n=570 and ord_19(2)=18. Related cyclic half-rate-component beds reached about 7.2, but neither those readings nor the cyclotomic fact establishes a circuit advantage on this untried asymmetric bed.")
SHAPES = ("Target cell 0.200 with (2,4)x(3,6): n=570, kb=114 and guaranteed rate 6/30=0.200. A has 38 checks on 76 variables; B has 57 checks on 114 variables. Both components have row degree six and column degree three. Require actual k=114. Parent failure maps implicated multiple Acol/Bcol and Acol/Brow blocks, including Bcol2/Bcol3; the small failure samples do not identify a persistent local defect.")
FILLING = ("Twenty contracts cross two A support architectures, five full-augmentation-rank B architectures and two internal-difference overlap policies. Every component row has six terms and every column three, so both CSS check weights are nine and six-round extraction uses 24624 CNOTs rather than 25056. Fresh offset portfolios use only verified component lifts for connectivity, full rank, four/six cycles, exact kernels through six, sampled short witnesses and triple-boundary concentration. Sampled protection preference saturates at twelve; no screen is measured pL. An exact uncolored full-check overlap invariant is hash-partitioned by slot, so different emitted slots cannot be Tanner-isomorphic. This targets the parent's X-basis tail: slots4/5 had 12/13 failures and 28/48 LSD settlements per 200000 shots despite d2s=12, while slot2 observed no failures.")
FAMILY_NAME = "r020_C19_24x36_balanced9_invariant_partition_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 19

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("This cyclic family requires q=19")
    # Same cyclic rule; enumeration and operations really have order q.
    els = [(a,) for a in range(q)]

    def red(g):
        return (int(g[0]) % q,)

    def idx(g):
        return red(g)[0]

    def mul(g, h):
        return ((g[0] + h[0]) % q,)

    def inv(g):
        return ((-g[0]) % q,)

    return dict(elements=els, ident=(0,), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 4)               # (ra, ca)
B_SHAPE = (3, 6)               # (rb, cb)


def _retired_C29_fourier_ctx(rng, G, idx):
    """Historical C29 search; never called for the new shape or order."""
    from itertools import combinations

    # First-row counts; the second row completes each column to three.
    # The first two profiles have row totals 6/6. The third has 5/7,
    # replacing the parent's augmentation-rank-one, all-even row.
    # Every profile still contains exactly twelve terms.
    profiles = ((1, 1, 2, 2), (0, 1, 2, 3), (1, 1, 1, 2))
    pairs = ((0, 0), (0, 1), (1, 1), (0, 2))
    architecture = (idx // 5) % len(pairs)
    pa, pb = pairs[architecture]

    def support(p):
        first = profiles[p]
        return [list(first), [3 - value for value in first]]

    sa, sb = support(pa), support(pb)
    assert _grid_weights(sa, sb) <= min(10, W_MAX)
    assert _no_dead_lines(sa, sb)
    for counts in (sa, sb):
        assert sum(map(sum, counts)) == 12
        assert all(sum(column) == 3 for column in zip(*counts))
        augmentation = [
            sum((count & 1) << j for j, count in enumerate(row))
            for row in counts
        ]
        assert gf2_rank(augmentation) == 2
    q = G.q

    def normalized(cell, multiplier):
        if not cell:
            return ()
        values = [(multiplier * value) % q for value in cell]
        return min(tuple(sorted((value - origin) % q for value in values))
                   for origin in values)

    def geometry_key(grids):
        # Quotient independent cell origins, cyclic multipliers,
        # component row/column permutations and side exchange.
        # This is not a complete Tanner-isomorphism classifier.
        images = []
        for u in range(1, q):
            sides = []
            for counts, grid in zip((sa, sb), grids):
                columns = [
                    (counts[0][j], normalized(grid[0][j], u),
                     counts[1][j], normalized(grid[1][j], u))
                    for j in range(4)
                ]
                normal = tuple(sorted(columns))
                exchanged = tuple(sorted(
                    (c, d, a, b) for a, b, c, d in columns))
                sides.append(min(normal, exchanged))
            images.append(tuple(sorted(sides)))
        return min(images)

    # Five indexed cross-row briefs, including the fully separated case.
    # Internal differences never repeat within a component row.
    # Other architectures distinguish the joint normalized cell geometry.
    repetition_cards = ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2))
    # Four Sidon trinomial shapes, with the fifth card distinguishing
    # joint binomial/trinomial geometry rather than claiming a fifth
    # affine-equivalence class of single trinomials.
    markers = (3, 4, 5, 9, 3)
    geometry_rng = random.Random(291817 + 1559 * architecture)
    wanted = 5 * (idx // 20) + idx % 5
    book, seen_geometries = [], set()

    for attempt in range(10000):
        card = len(book) % 5
        grids = []
        legal = True
        for side, counts in enumerate((sa, sb)):
            sites = [(i, j) for i in range(2) for j in range(4)
                     if counts[i][j] == 2]
            values = geometry_rng.sample(range(1, (q + 1) // 2),
                                         len(sites))
            if architecture == 0:
                repetitions = repetition_cards[card][side]
                if repetitions:
                    values[2] = values[0]
                if repetitions == 2:
                    values[3] = values[1]
            directions = dict(zip(sites, values))
            grid = []
            for i in range(2):
                row = []
                row_differences = set()
                for j in range(4):
                    weight = counts[i][j]
                    if weight == 0:
                        cell = []
                    elif weight == 1:
                        cell = [0]
                    elif weight == 2:
                        cell = [0, directions[i, j]]
                    else:
                        step = geometry_rng.randrange(1, q)
                        marker = markers[(card + side) % len(markers)]
                        cell = [0, step, step * marker % q]
                    # Repeated ordered differences within one row force
                    # component four-cycles regardless of cell offsets.
                    for x in cell:
                        for y in cell:
                            if x == y:
                                continue
                            difference = (x - y) % q
                            if difference in row_differences:
                                legal = False
                            row_differences.add(difference)
                    row.append(cell)
                grid.append(row)
            grids.append(grid)
        if not legal:
            continue
        key = geometry_key(grids)
        if key in seen_geometries:
            continue
        seen_geometries.add(key)
        book.append(grids)
        if len(book) > wanted:
            break

    if len(book) <= wanted:
        raise ValueError("insufficient new relative-support architectures")
    geometry = book[wanted]

    def labels(s, side):
        grid = []
        for i in range(2):
            row = []
            for j in range(4):
                origin = rng.randrange(q)
                cell = [((origin + value) % q,)
                        for value in geometry[side][i][j]]
                assert len(cell) == s[i][j]
                row.append(cell)
            grid.append(row)
        return grid

    q = G.q
    mask = (1 << q) - 1
    component_mask = (1 << (4 * q)) - 1

    def components(hx, hz):
        # Extract from the verified construction, not a separate lift.
        # Fix B-column 0 in HX to obtain the left-lifted A component.
        ar = []
        for i in range(2):
            for r in range(q):
                row = hx[(i * 4) * q + r]
                compact = 0
                for j in range(4):
                    compact |= ((row >> ((j * 4) * q)) & mask) << (j * q)
                ar.append(compact)
        # Fix A-column 0 in HZ to obtain the right-lifted B component.
        br = [row & component_mask for row in hz[:2 * q]]
        return ar, br

    from math import cos, sin, pi, sqrt

    # These phases analyze the overlap graph of verified component rows;
    # they do not construct a separate left/right lift.
    phases = [
        [complex(cos(2 * pi * f * t / q),
                 sin(2 * pi * f * t / q)) for t in range(q)]
        for f in range(1, (q + 1) // 2)
    ]

    def expansion_radius(rows):
        """Nontrivial normalized overlap spectrum of a cyclic component.

        Columns have degree three. The two check-block degrees may
        differ for the 5/7 profile, so normalize each block separately.
        The excluded Perron eigenvalue is one. This is an expansion
        statistic, not a calibrated circuit-failure estimate.
        """
        weights = [
            [
                [(rows[i * q] & rows[j * q + t]).bit_count()
                 if i != j or t != 0 else 0
                 for t in range(q)]
                for j in range(2)
            ]
            for i in range(2)
        ]
        degrees = [sum(map(sum, block)) for block in weights]
        if not all(degrees):
            return 1.0
        scale = sqrt(degrees[0] * degrees[1])
        # At frequency zero, remove the Perron eigenvalue exactly.
        a0 = sum(weights[0][0]) / degrees[0]
        d0 = sum(weights[1][1]) / degrees[1]
        radius = abs(a0 + d0 - 1.0)
        for phase in phases:
            a = sum(w * z.real for w, z in zip(weights[0][0], phase))
            d = sum(w * z.real for w, z in zip(weights[1][1], phase))
            b = sum(w * z for w, z in zip(weights[0][1], phase))
            a /= degrees[0]
            d /= degrees[1]
            b /= scale
            middle = (a + d) / 2
            spread = sqrt(((a - d) / 2) ** 2 + abs(b) ** 2)
            radius = max(radius, abs(middle + spread),
                         abs(middle - spread))
        return round(radius, 9)

    def cycle_cost(rows):
        """Exact C4/C6, full rank, and verified-graph expansion."""
        m = len(rows)
        columns = [[] for _ in range(4 * q)]
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                columns[bit.bit_length() - 1].append(r)
                row ^= bit

        adj = [dict() for _ in range(m)]
        for ns in columns:
            for a, b in combinations(ns, 2):
                adj[a][b] = adj[a].get(b, 0) + 1
                adj[b][a] = adj[b].get(a, 0) + 1

        seen, todo = {0}, [0]
        while todo:
            a = todo.pop()
            for b in adj[a]:
                if b not in seen:
                    seen.add(b)
                    todo.append(b)
        if len(seen) != m:
            return None

        four = sum(v * (v - 1) // 2
                   for a in range(m) for b, v in adj[a].items() if a < b)
        six = 0
        for a in range(m):
            for b, ab in adj[a].items():
                if b <= a:
                    continue
                for c, bc in adj[b].items():
                    if c > b and c in adj[a]:
                        six += ab * bc * adj[a][c]
        # Remove triangles whose check-pair edges reuse a variable.
        for ns in columns:
            for a, b, c in combinations(ns, 3):
                six -= adj[a][b] + adj[a][c] + adj[b][c] - 2
        deficit = m - gf2_rank(rows)
        if deficit:
            return None
        return (four, expansion_radius(rows), six, deficit)

    pools = [[], []]
    for trial in range(192):
        a, b = labels(sa, 0), labels(sb, 1)
        hx, hz, _ = lifted_matrices(G, a, b)
        ar, br = components(hx, hz)
        for side, grid, rows in ((0, a, ar), (1, b, br)):
            cost = cycle_cost(rows)
            if cost is not None:
                pools[side].append((cost, trial, grid, rows))
                pools[side].sort(key=lambda item: (item[0], item[1]))
                del pools[side][6:]

    if not all(pools):
        raise ValueError("no connected component candidate")

    def kernel_upper_bound(rows):
        """Sample kernel words using randomized information sets.

        This is only an upper-bound heuristic, not a distance certificate
        and not an estimate of the scored circuit-level failure rate.
        """
        n = 4 * q
        best = n
        small = set()

        def record(word):
            nonlocal best
            if not word:
                return
            weight = word.bit_count()
            best = min(best, weight)
            if weight <= 12:
                small.add(word)

        # Deterministic minor witnesses supplement information-set draws.
        # Obtain polynomial coefficients from the VERIFIED column
        # syndromes, and verify every generator against the actual rows.
        columns = [0] * n
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                columns[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        coefficients = [
            [(columns[j * q] >> (i * q)) & mask for j in range(4)]
            for i in range(2)
        ]

        def convolution(x, y):
            result = 0
            while x:
                bit = x & -x
                shift = bit.bit_length() - 1
                result ^= ((y << shift) | (y >> (q - shift))) & mask
                x ^= bit
            return result

        generators = set()
        for a, b, c in combinations(range(4), 3):
            blocks = [0] * 4
            for destination, (u, v) in zip(
                    (a, b, c), ((b, c), (a, c), (a, b))):
                blocks[destination] = (
                    convolution(coefficients[0][u], coefficients[1][v])
                    ^ convolution(coefficients[0][v], coefficients[1][u]))
            for shift in range(q):
                word = 0
                for j, block in enumerate(blocks):
                    rotated = ((block << shift)
                               | (block >> (q - shift))) & mask
                    word |= rotated << (j * q)
                if word:
                    assert all((row & word).bit_count() % 2 == 0
                               for row in rows), "unverified minor witness"
                    generators.add(word)
        ordered = sorted(generators)
        for i, word in enumerate(ordered):
            record(word)
            for other in ordered[:i]:
                record(word ^ other)

        # Give every finalist the same permutation stream. Witness
        # minima remain upper bounds, not certified distances.
        probe_rng = random.Random(291991 + 1009 * idx)
        for _ in range(64):
            mat = list(rows)
            order = list(range(n))
            probe_rng.shuffle(order)
            pivots = []
            rank = 0
            for c in order:
                bit = 1 << c
                found = next((r for r in range(rank, len(mat))
                              if mat[r] & bit), None)
                if found is None:
                    continue
                mat[rank], mat[found] = mat[found], mat[rank]
                pivot_row = mat[rank]
                for r in range(len(mat)):
                    if r != rank and mat[r] & bit:
                        mat[r] ^= pivot_row
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
                weight = word.bit_count()
                best = min(best, weight)
                if weight <= 12:
                    small.add(word)

            for i, x in enumerate(basis):
                for j in range(i):
                    word = x ^ basis[j]
                    weight = word.bit_count()
                    best = min(best, weight)
                    if weight <= 12:
                        small.add(word)
            if best <= 6:
                break

        # Count translation orbits rather than rewarding accidental
        # resampling of the same word. This permutes witnesses only;
        # it does not construct or substitute a different lifted matrix.
        orbit_words = set()
        for word in small:
            blocks = [(word >> (j * q)) & mask for j in range(4)]
            representative = None
            for shift in range(q):
                image = 0
                for j, block in enumerate(blocks):
                    rotated = ((block << shift) |
                               (block >> (q - shift))) & mask
                    image |= rotated << (j * q)
                if representative is None or image < representative:
                    representative = image
            orbit_words.add(representative)
        loads = [0] * 4
        for word in orbit_words:
            for j in range(4):
                loads[j] += ((word >> (j * q)) & mask).bit_count()
        return best, len(orbit_words), max(loads)

    def small_kernel_word(rows):
        """Exact kernel exclusion through six, plus triple-boundary exposure.

        The boundary score is a trapping-set heuristic, not measured pL.
        Counting its load by component column tests concentration without
        claiming that one noisy failure-map block identifies a defect.
        """
        columns = [0] * (4 * q)
        for r, row in enumerate(rows):
            check_bit = 1 << r
            while row:
                bit = row & -row
                columns[bit.bit_length() - 1] |= check_bit
                row ^= bit
        seen = {0}
        for value in columns:
            if value in seen:
                return True, 10**9, 10**9
            seen.add(value)
        for i, x in enumerate(columns):
            for j in range(i):
                value = x ^ columns[j]
                if value in seen:
                    return True, 10**9, 10**9
                seen.add(value)

        exposure = 0
        loads = [0] * 4
        for i, x in enumerate(columns):
            for j in range(i):
                pair = x ^ columns[j]
                for h in range(j):
                    value = pair ^ columns[h]
                    if value in seen:
                        return True, 10**9, 10**9
                    seen.add(value)
                    weight = value.bit_count()
                    if weight <= 3:
                        penalty = (5 - weight) ** 2
                        exposure += penalty
                        loads[i // q] += penalty
                        loads[j // q] += penalty
                        loads[h // q] += penalty
        return False, exposure, max(loads)

    unit = [[[G.e]]]

    def projected(side, grid):
        # The first sector of this smaller VERIFIED identity-partner
        # construction is exactly the requested component.
        if side == 0:
            rows = lifted_matrices(G, grid, unit)[0]
        else:
            rows = lifted_matrices(G, unit, grid)[1]
        return [row & component_mask for row in rows]

    def has_eight_kernel(rows):
        """Exact weight-eight existence test after <=6 exclusion.

        Enumerate one representative of every four-support translation
        orbit. Equal canonical syndromes from different support orbits
        give equal-syndrome four-sets. With no smaller kernel these sets
        are disjoint, hence their difference has weight eight.
        C29 acts freely on four-sets and on their nonzero syndromes.
        """
        columns = [0] * (4 * q)
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                columns[bit.bit_length() - 1] |= 1 << r
                row ^= bit

        def translated_syndrome(value, shift):
            lo, hi = value & mask, value >> q
            lo = ((lo >> shift) | (lo << (q - shift))) & mask
            hi = ((hi >> shift) | (hi << (q - shift))) & mask
            return lo | (hi << q)

        # Check the orbit convention against every verified column.
        for j in range(4):
            seed = columns[j * q]
            for t in range(q):
                assert columns[j * q + t] == (
                    translated_syndrome(seed, (-t) % q))

        def canonical_syndrome(value):
            if not value:
                return 0
            # A minimal rotation has an occupied bit at lift zero in
            # its highest nonzero check block. Only those shifts matter.
            occupied = (value >> q) or (value & mask)
            answer = None
            while occupied:
                bit = occupied & -occupied
                shift = bit.bit_length() - 1
                image = translated_syndrome(value, shift)
                if answer is None or image < answer:
                    answer = image
                occupied ^= bit
            return answer

        seen = set()
        size = len(columns)
        for anchor in range(0, size, q):
            first_block = anchor // q
            for a in range(anchor + 1, size - 2):
                pair = columns[anchor] ^ columns[a]
                for b in range(a + 1, size - 1):
                    triple = pair ^ columns[b]
                    for c in range(b + 1, size):
                        vertices = (anchor, a, b, c)
                        canonical = True
                        # Choose the lexicographically least support
                        # with lift zero in its first occupied block.
                        for v in vertices[1:]:
                            if v // q != first_block:
                                break
                            shift = v % q
                            image = tuple(sorted(
                                (u // q) * q + (u % q - shift) % q
                                for u in vertices))
                            if image < vertices:
                                canonical = False
                                break
                        if not canonical:
                            continue
                        syndrome = triple ^ columns[c]
                        if not syndrome:
                            return True
                        key = canonical_syndrome(syndrome)
                        if key in seen:
                            return True
                        seen.add(key)
        return False

    finalists = []
    for side, pool in enumerate(pools):
        entries = []
        for cost, trial, grid, rows in pool:
            # Shift an entire cell: preserve its indexed internal geometry
            # while changing inter-cell voltages and component cycles.
            for step in range(64):
                proposal = [[list(cell) for cell in row] for row in grid]
                i, j = rng.randrange(2), rng.randrange(4)
                shift = rng.randrange(1, q)
                proposal[i][j] = [((g[0] + shift) % q,)
                                  for g in proposal[i][j]]
                proposed_rows = projected(side, proposal)
                proposed_cost = cycle_cost(proposed_rows)
                if proposed_cost is not None and proposed_cost <= cost:
                    grid, rows, cost = proposal, proposed_rows, proposed_cost
            bad, boundary, boundary_peak = small_kernel_word(rows)
            d, low_orbits, word_peak = kernel_upper_bound(rows)
            # The orbit-eight argument is used only after the exact
            # through-six screen passes. Failure is a selection penalty,
            # never represented as an exact weight-eight census.
            eight_bad = int(bad or has_eight_kernel(rows))
            entries.append((cost, d, int(bad), eight_bad,
                            low_orbits, word_peak,
                            boundary, boundary_peak, trial, grid))
        finalists.append(entries)

    # Match exact short protection and a capped sampled floor first.
    # Then protect expansion in the weaker component, rather than
    # reward undersampled word counts or an optimistic larger distance.
    # None of these statistics substitutes for ASC memory measurement.
    options = []
    for a in finalists[0]:
        for b in finalists[1]:
            ac, ad, abad, a8, alow, awpeak, abound, abpeak, at, ag = a
            bc, bd, bbad, b8, blow, bwpeak, bbound, bbpeak, bt, bg = b
            da, db = min(ad, 12), min(bd, 12)
            key = (abad + bbad, a8 + b8,
                   int(min(da, db) < 12), ac[0] + bc[0],
                   max(ac[1], bc[1]), ac[1] + bc[1],
                   max(abpeak, bbpeak), abound + bbound,
                   max(awpeak, bwpeak), max(alow, blow), alow + blow,
                   ac[2] + bc[2], abs(da - db),
                   ac[3] + bc[3], at, bt)
            options.append((key, ag, bg))
    options.sort(key=lambda item: item[0])

    for key, a, b in options:
        hx, hz, n = lifted_matrices(G, a, b)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k == 116 and css_ok(hx, hz):
            assert n == 580
            assert max(row.bit_count() for row in hx + hz) <= min(10, W_MAX)
            assert 6 * sum(row.bit_count() for row in hx + hz) == 25056
            # Confirm the final components are the same full-rank
            # projections used throughout the voltage search.
            ar, br = components(hx, hz)
            assert gf2_rank(ar) == gf2_rank(br) == 2 * q
            return {"idx": idx, "cells": {"A": a, "B": b}, "k": k,
                    "architecture": architecture,
                    "geometry": wanted, "screen": key,
                    "small_kernel_safe": key[0] == 0,
                    "through_eight_safe": key[1] == 0}
    raise ValueError("no full-rank filling passed verified rate-0.200 screening")


def NEW_CTX(rng, G, idx):
    """Pure C19 support contracts, component portfolios and invariant reservation."""
    from itertools import combinations, permutations
    from collections import Counter
    import hashlib
    import numpy as np

    q = G.q
    assert q == 19 and A_SHAPE == (2, 4) and B_SHAPE == (3, 6)
    assert 0 <= idx < N_SLOTS
    architecture = (idx // 2) % 10
    ap, bp = divmod(architecture, 5)
    batch, policy = idx // 20, idx % 2

    a_profiles = (
        ((1, 1, 2, 2), (2, 2, 1, 1)),
        ((0, 1, 2, 3), (3, 2, 1, 0)),
    )
    triangle = ((2, 1, 0), (0, 2, 1), (1, 0, 2))
    reciprocal = (
        ((2, 1, 0), (0, 1, 2)),
        ((2, 0, 1), (0, 2, 1)),
        ((1, 2, 0), (1, 0, 2)),
    )
    b_columns = (
        triangle + triangle,
        triangle + ((1, 1, 1),) * 3,
        reciprocal[0] + reciprocal[1] + reciprocal[2],
        triangle + reciprocal[0] + ((1, 1, 1),),
        ((3, 0, 0), (0, 3, 0), (0, 0, 3))
        + ((1, 1, 1),) * 3,
    )
    sa = [list(row) for row in a_profiles[ap]]
    sb = [[column[i] for column in b_columns[bp]] for i in range(3)]
    supports = (sa, sb)
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) == 9 <= W_MAX
    for counts in supports:
        assert all(sum(row) == 6 for row in counts)
        assert all(sum(column) == 3 for column in zip(*counts))
        assert all(0 <= t <= min(3, CELL_MAX)
                   for row in counts for t in row)
        augmentation = [
            sum((t & 1) << j for j, t in enumerate(row))
            for row in counts
        ]
        assert gf2_rank(augmentation) == len(counts)

    def normalized(cell, u):
        if not cell:
            return ()
        values = [u * a % q for a in cell]
        return min(tuple(sorted((a - origin) % q for a in values))
                   for origin in values)

    def geometry_key(grids):
        images = []
        for u in range(1, q):
            sides = []
            for grid in grids:
                nr, nc = len(grid), len(grid[0])
                relatives = [[normalized(cell, u) for cell in row]
                             for row in grid]
                sides.append(min(
                    tuple(sorted(tuple(relatives[i][j] for i in order)
                                 for j in range(nc)))
                    for order in permutations(range(nr))))
            # Shapes differ, so do not mistake exchanging the two
            # component geometries for a same-shaped labeling.
            images.append(tuple(sides))
        return min(images)

    # Each brief fixes relative supports before any offset optimization.
    # Policy zero separates all internal directions across component rows;
    # policy one permits exactly one cross-row direction repetition.
    grng = random.Random(190019 + 1637 * architecture)
    wanted = 2 * batch + policy
    book, seen = [], set()
    for attempt in range(60000):
        requested_overlap = len(book) % 2
        grids = []
        legal = True
        for counts in supports:
            grid, row_sets = [], []
            for row_counts in counts:
                row, used = [], set()
                for weight in row_counts:
                    relative = ([] if weight == 0 else
                                [0] + grng.sample(range(1, q), weight - 1))
                    for a, b in combinations(relative, 2):
                        difference = (a - b) % q
                        direction = min(difference, q - difference)
                        if direction in used:
                            legal = False
                        used.add(direction)
                    row.append(relative)
                grid.append(row)
                row_sets.append(used)
            union = set().union(*row_sets)
            overlap = sum(map(len, row_sets)) - len(union)
            if overlap != requested_overlap:
                legal = False
            grids.append(grid)
        if not legal:
            continue
        signature = geometry_key(grids)
        if signature in seen:
            continue
        seen.add(signature)
        book.append(grids)
        if len(book) > wanted:
            break
    if len(book) <= wanted:
        raise ValueError("C19 relative-support contract book exhausted")
    geometry = book[wanted]

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def draw(side):
        result = []
        for row in geometry[side]:
            out = []
            for relative in row:
                origin = rng.randrange(q)
                out.append([((origin + value) % q,) for value in relative])
            result.append(out)
        return result

    unit = [[[G.e]]]

    def component(side, grid):
        # Project only the scaffold's verified construction.
        if side == 0:
            rows = lifted_matrices(G, grid, unit)[0]
        else:
            rows = lifted_matrices(G, unit, grid)[1]
        mask = (1 << (len(grid[0]) * q)) - 1
        return [row & mask for row in rows]

    def columns(rows, width):
        result = [0] * width
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                result[bit.bit_length() - 1] |= tag
                row ^= bit
        return result

    def assess(side, grid):
        rows = component(side, grid)
        if gf2_rank(rows) != len(rows):
            return None
        width = len(grid[0]) * q
        cols = columns(rows, width)
        assert all(col.bit_count() == 3 for col in cols)
        adj = [dict() for _ in rows]
        neighborhoods = []
        for syndrome in cols:
            ns = []
            while syndrome:
                bit = syndrome & -syndrome
                ns.append(bit.bit_length() - 1)
                syndrome ^= bit
            neighborhoods.append(ns)
            for a, b in combinations(ns, 2):
                adj[a][b] = adj[a].get(b, 0) + 1
                adj[b][a] = adj[b].get(a, 0) + 1
        reached, todo = {0}, [0]
        while todo:
            a = todo.pop()
            for b in adj[a]:
                if b not in reached:
                    reached.add(b)
                    todo.append(b)
        if len(reached) != len(rows):
            return None
        four = sum(v * (v - 1) // 2
                   for a, ns in enumerate(adj)
                   for b, v in ns.items() if a < b)
        six = 0
        for a, ns in enumerate(adj):
            for b, ab in ns.items():
                if b <= a:
                    continue
                for c, bc in adj[b].items():
                    if c > b and c in ns:
                        six += ab * bc * ns[c]
        for ns in neighborhoods:
            for a, b, c in combinations(ns, 3):
                six -= adj[a][b] + adj[a][c] + adj[b][c] - 2
        return (four, six), rows, cols

    def short_and_boundary(cols):
        """Exact <=4 exclusion and six-word collision census.

        All columns have odd degree three, excluding odd kernel weights.
        After <=4 exclusion, each six-word has ten triple partitions.
        Boundary concentration is only a decoder-ambiguity surrogate.
        """
        if 0 in cols or len(set(cols)) != len(cols):
            return 1, 10**12, 10**12, 10**12
        pair_syndromes = set()
        for i, x in enumerate(cols):
            for y in cols[:i]:
                syndrome = x ^ y
                if not syndrome or syndrome in pair_syndromes:
                    return 1, 10**12, 10**12, 10**12
                pair_syndromes.add(syndrome)
        triples = {}
        collisions = boundary = 0
        loads = [0] * (len(cols) // q)
        for i, x in enumerate(cols):
            for j in range(i):
                pair = x ^ cols[j]
                for h in range(j):
                    syndrome = pair ^ cols[h]
                    previous = triples.get(syndrome, 0)
                    collisions += previous
                    triples[syndrome] = previous + 1
                    weight = syndrome.bit_count()
                    if weight <= 3:
                        penalty = (5 - weight) ** 2
                        boundary += penalty
                        loads[i // q] += penalty
                        loads[j // q] += penalty
                        loads[h // q] += penalty
        assert collisions % 10 == 0
        return int(collisions != 0), collisions // 10, boundary, max(loads)

    def witness_probe(rows, cols, side):
        """Verified kernel witnesses: upper bounds, not distance certificates."""
        probe = random.Random(190117 + 1009 * idx + side)
        order = list(range(len(cols)))
        minimum = len(cols) + 1
        small = set()

        def offer(word):
            nonlocal minimum
            if not word:
                return
            weight = word.bit_count()
            minimum = min(minimum, weight)
            if weight <= 10:
                small.add(word)

        # The two-row component has inexpensive deterministic minor
        # witnesses. Coefficients come from the verified column syndromes;
        # every resulting word is checked against the actual component.
        if side == 0:
            mask = (1 << q) - 1
            coefficients = [
                [(cols[j * q] >> (i * q)) & mask for j in range(4)]
                for i in range(2)
            ]

            def convolution(x, y):
                result = 0
                while x:
                    bit = x & -x
                    shift = bit.bit_length() - 1
                    result ^= ((y << shift) | (y >> (q - shift))) & mask
                    x ^= bit
                return result

            generators = set()
            for a, b, c in combinations(range(4), 3):
                blocks = [0] * 4
                for destination, (u, v) in zip(
                        (a, b, c), ((b, c), (a, c), (a, b))):
                    blocks[destination] = (
                        convolution(coefficients[0][u], coefficients[1][v])
                        ^ convolution(coefficients[0][v], coefficients[1][u]))
                for shift in range(q):
                    word = sum(
                        (((block << shift) | (block >> (q - shift))) & mask)
                        << (j * q) for j, block in enumerate(blocks))
                    if word:
                        assert all((row & word).bit_count() % 2 == 0
                                   for row in rows)
                        generators.add(word)
            ordered = sorted(generators)
            for i, word in enumerate(ordered):
                offer(word)
                for other in ordered[:i]:
                    offer(word ^ other)

        for trial in range(24):
            probe.shuffle(order)
            pivots, basis = {}, []
            for c in order:
                syndrome, word = cols[c], 1 << c
                while syndrome:
                    p = syndrome.bit_length() - 1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = syndrome, word
                        break
                    syndrome ^= old[0]
                    word ^= old[1]
                if not syndrome and word:
                    basis.append(word)
                    offer(word)
            for sweep in range(2):
                probe.shuffle(basis)
                for i, word in enumerate(basis):
                    weight = word.bit_count()
                    for other in basis[:i]:
                        proposal = word ^ other
                        offer(proposal)
                        pw = proposal.bit_count()
                        if proposal and pw < weight:
                            word, weight = proposal, pw
                    basis[i] = word
            if minimum <= 6:
                break

        # Count short translation orbits, not repeat discoveries.
        mask = (1 << q) - 1
        representatives = set()
        for word in small:
            blocks = [(word >> (j * q)) & mask
                      for j in range(len(cols) // q)]
            representatives.add(min(
                sum((((block << shift) | (block >> (q - shift))) & mask)
                    << (j * q) for j, block in enumerate(blocks))
                for shift in range(q)))
        loads = [
            sum(((word >> (j * q)) & mask).bit_count()
                for word in representatives)
            for j in range(len(cols) // q)]
        return minimum, len(representatives), max(loads, default=0)

    def portfolio(side):
        positions = [(i, j) for i, row in enumerate(supports[side])
                     for j, weight in enumerate(row) if weight]
        pool, seen_labels = [], set()
        for trial in range(192):
            grid = draw(side)
            result = assess(side, grid)
            if result is None:
                continue
            signature = tuple(tuple(sorted(g[0] for g in cell))
                              for row in grid for cell in row)
            if signature in seen_labels:
                continue
            seen_labels.add(signature)
            cost, rows, cols = result
            pool.append((cost, trial, grid, rows, cols))
            pool.sort(key=lambda entry: entry[:2])
            del pool[16:]

        finalists = []
        for cost, trial, grid, rows, cols in pool:
            for step in range(48):
                proposal = clone(grid)
                i, j = rng.choice(positions)
                shift = rng.randrange(1, q)
                proposal[i][j] = [((g[0] + shift) % q,)
                                  for g in proposal[i][j]]
                result = assess(side, proposal)
                if result is not None and result[0] <= cost:
                    cost, rows, cols = result
                    grid = proposal
            bad, six_words, boundary, peak = short_and_boundary(cols)
            distance, short_orbits, word_peak = witness_probe(rows, cols, side)
            finalists.append(dict(
                grid=grid, rows=rows, cost=cost, bad=bad,
                six_words=six_words, boundary=boundary, peak=peak,
                distance=distance, short=short_orbits, word_peak=word_peak))
        return finalists

    def invariant_bucket(hx, hz):
        """Exact uncolored check-overlap invariant of the full Tanner graph.

        A Tanner isomorphism permutes this overlap matrix and its square.
        Histograms therefore cannot change under qubit/check relabeling,
        including exchange or mixing of the check bases. Hash collisions
        only discard diversity; they cannot admit an isomorphic pair
        into two different slot buckets.
        """
        rows = hx + hz
        m = len(rows)
        overlap = np.zeros((m, m), dtype=np.int64)
        for i, row in enumerate(rows):
            overlap[i, i] = row.bit_count()
            for j in range(i):
                value = (row & rows[j]).bit_count()
                overlap[i, j] = overlap[j, i] = value
        square = overlap @ overlap

        def histogram(array):
            values, counts = np.unique(array, return_counts=True)
            return tuple((int(v), int(c)) for v, c in zip(values, counts))

        degrees = Counter(
            col.bit_count() for col in columns(rows, 570))
        identity = (
            tuple(sorted(degrees.items())),
            histogram(overlap),
            histogram(square),
            histogram(np.diag(square)),
        )
        digest = hashlib.sha256(repr(identity).encode("ascii")).digest()
        return int.from_bytes(digest, "big") % N_SLOTS

    # Bucket reservation is a hard distinctness criterion, not a fitness
    # surrogate. Rebuild fresh portfolios if no finalist belongs to this
    # slot's invariant partition. All state stays local to this pure call.
    for restart in range(5):
        aa, bb = portfolio(0), portfolio(1)
        options = []
        for a in aa:
            for b in bb:
                da, db = min(a["distance"], 12), min(b["distance"], 12)
                key = (
                    a["bad"] + b["bad"],
                    int(min(da, db) < 12),
                    max(a["six_words"], b["six_words"]),
                    a["six_words"] + b["six_words"],
                    a["cost"][0] + b["cost"][0],
                    max(a["peak"], b["peak"]),
                    a["boundary"] + b["boundary"],
                    max(a["word_peak"], b["word_peak"]),
                    a["short"] + b["short"],
                    a["cost"][1] + b["cost"][1],
                    abs(da - db),
                )
                options.append((key, a, b))
        options.sort(key=lambda entry: entry[0])

        for key, a, b in options:
            A, B = a["grid"], b["grid"]
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if n != 570 or k != 114:
                continue
            if invariant_bucket(hx, hz) != idx:
                continue
            assert css_ok(hx, hz)
            assert max(row.bit_count() for row in hx + hz) == 9
            assert 6 * sum(row.bit_count() for row in hx + hz) == 24624
            assert component(0, A) == a["rows"]
            assert component(1, B) == b["rows"]
            assert all(len(set(cell)) == len(cell)
                       for grid in (A, B) for row in grid for cell in row)
            return {
                "idx": idx, "cells": {"A": A, "B": B}, "k": k,
                "contract": (ap, bp, policy, batch),
                "geometry": geometry, "screen": key,
                "invariant_partition": (idx, N_SLOTS),
                "component_witness_bounds": (a["distance"], b["distance"]),
                "through_six_safe": key[0] == 0,
            }

    raise ValueError("no full-rank C19 finalist in this slot's invariant partition")


def CELL(rng, G, side, i, j, ctx):
    """Return the complete screened cell without drawing new labels."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: d2s>=8 guarantees basis-balanced memory performance; seed slot19 had X failures 100/78000 versus Z 29/200000, with scores 5.634 versus 6.509.
# REFUTED: Equal d2s and larger k guarantee no worse circuit score: current-parent slot16 had d2s=12,k=116,s=6.389, versus slot21 with d2s=12,k=112,s=6.931.
# REFUTED: Internal binomial-step orbit signatures alone ensure distinct submitted Tanner codes: the C29 parent retained only 25 distinct codes from 60 slots, including slot2 marked iso_dup_of_1.
# REFUTED: Weight-nine extraction and d2s=12 uniformly attain the C29 sample ceiling: parent slot17 had seven X failures/200000 and s=6.992, while slot36 had six Z failures/200000 and s=7.035.
# REFUTED: Adding two terms to each C29 component necessarily improves this low-load program: the measured degree-three/four child scored 7.0220 versus the weight-nine parent's 7.1850, with 29232 rather than 25056 extraction CNOTs.
# REFUTED: Full-rank Fourier-expansion and orbit-eight screening raises mean C29 circuit fitness above 7.1850: the measured program scored 7.0514; slots4/5 had 12/13 X failures and 28/48 X LSD settlements per 200000 shots despite d2s=12.
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
