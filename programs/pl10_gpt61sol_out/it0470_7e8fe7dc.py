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
CONCEPT = ("Cyclic C19, order 19, shapes (2,5)x(2,5): balanced degree-three/four components selected by independent noisy-syndrome BP holdouts should exceed the approximately6.2 cell-0.300 frontier at verified k=171; actual ASC circuit pL decides.")
GROUP = ("Jump from C43 to genuinely different C19. BEDS reports about5.7 on C19 (2,5)x(2,5) and about5.8 on the related C20 bed, neither isolating a group effect. Order19 avoids involutions and gives n=551. This tests a decoder-aware filling on an existing destination bed, not an asserted group advantage.")
SHAPES = ("Equal (2,5) components each have95 variables and38 checks. Size factor29 permits q<=20; order19 gives n=551, kb=171 and guaranteed rate9/29=0.310345. Complete verified ranks require k=171, retaining cell0.300 with zero k-kb. Parent slot42 had one X failure/200000,27 collective logical flips and mild enrichment at Acol0/Bcol2 and Acol1/Bcol2; the other measured siblings had no failures. This does not establish a persistent block or logical-qubit defect.")
FILLING = ("Canonical count contracts prescribe column totals3/4, row totals7/8 and15..16 terms per side, excluding multiple pure columns in one component row. Both CSS weight sums are<=12 and six-round CNOT counts23940..25536. Augmentation rank two and verified component ranks38 precede complete k=171 verification. Slots own different unordered count-contract pairs; overflow budgets change a prescribed relative-binomial ratio rather than submit saved orbit images. Fresh voltages are annealed using helper-derived actions. Exact component kernel<=4 rejection, exact weight-five/six counts, block-balanced triple boundaries and capped sampled kernel warnings shortlist alternatives. Complete verified CSS finalists undergo two-interval noisy-syndrome BP training followed by independent common-stream holdouts. Probes include degree-weighted exposure and random opposite-check suffix faults, omit ASC ordering and relay/LSD, and are not circuit pL. Target collective ambiguity and worse-basis failures without maximizing a distance upper bound.")
FAMILY_NAME = "r030_C19_25x25_balanced_counts_noisysyndrome_holdout_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 19

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("this cyclic family requires order 19")
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


def _design_supports(idx):
    """Twenty different support designs, not permutation-orbit copies."""
    profiles = (
        (2, 2, 2, 2, 2, 2),
        (1, 2, 2, 2, 2, 2),
        (1, 2, 3, 2, 2, 1),
        (1, 2, 3, 3, 2, 1),
        (1, 3, 2, 2, 1, 3),
        (3, 2, 2, 1, 2, 2),
        (3, 3, 1, 1, 1, 3),
        (3, 2, 2, 1, 3, 1),
        (3, 2, 1, 2, 2, 2),
        (3, 2, 2, 2, 2, 2),
        (3, 3, 1, 2, 1, 3),
        (3, 2, 2, 1, 2, 3),
    )
    pairs = (
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
        (0, 5), (0, 6), (0, 7), (0, 8), (0, 9),
        (0, 10), (0, 11), (1, 9), (2, 9), (3, 3),
        (3, 4), (3, 9), (4, 4), (4, 9), (9, 9),
    )
    a, b = pairs[idx % len(pairs)]
    return ([list(profiles[a][:3]), list(profiles[a][3:])],
            [list(profiles[b][:3]), list(profiles[b][3:])])


def _adjacent_coset_grid(rng, supports):
    """Every intra-cell ratio has order 6 or 7."""
    grid = []
    for row in supports:
        out = []
        for weight in row:
            t = rng.randrange(6)
            bs = [t] if weight == 1 else [t, (t + 1) % 6]
            while len(bs) < weight:
                bs.append((t + rng.randrange(2)) % 6)
            cell = []
            for b in bs:
                g = (rng.randrange(7), b)
                while g in cell:
                    g = (rng.randrange(7), b)
                cell.append(g)
            out.append(cell)
        grid.append(out)
    return grid


def _four_cycles(rows):
    total = 0
    for i, row in enumerate(rows):
        for other in rows[:i]:
            overlap = (row & other).bit_count()
            total += overlap * (overlap - 1) // 2
    return total


def _row_span(rows):
    pivots = {}
    for row in rows:
        while row:
            p = row.bit_length() - 1
            if p in pivots:
                row ^= pivots[p]
            else:
                pivots[p] = row
                break
    return pivots


def _in_row_span(v, pivots):
    while v:
        p = v.bit_length() - 1
        if p not in pivots:
            return False
        v ^= pivots[p]
    return True


def _has_short_logical(hx, hz, n):
    """Exact detection of non-stabilizer logicals of weight 1..4.

    Singleton/pair syndrome collisions cover all weights through four.
    Membership in the opposite check-row span distinguishes stabilizers.
    Both matrices come exclusively from lifted_matrices.
    """
    for checks, stabilizers in ((hx, hz), (hz, hx)):
        columns = [0] * n
        for r, row in enumerate(checks):
            tag = 1 << r
            while row:
                bit = row & -row
                columns[bit.bit_length() - 1] ^= tag
                row ^= bit
        span = _row_span(stabilizers)
        seen = {0: 0}
        bits = [1 << i for i in range(n)]
        for i, syndrome in enumerate(columns):
            v = bits[i]
            old = seen.get(syndrome)
            if old is None:
                seen[syndrome] = v
            elif not _in_row_span(v ^ old, span):
                return True
        for i in range(n):
            for j in range(i):
                syndrome = columns[i] ^ columns[j]
                v = bits[i] ^ bits[j]
                old = seen.get(syndrome)
                if old is None:
                    seen[syndrome] = v
                elif not _in_row_span(v ^ old, span):
                    return True
    return False


def _connected_checks(hx, hz, n):
    """Connectivity of the combined Tanner graph, using verified rows."""
    parent = list(range(n))

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for row in hx + hz:
        if not row:
            continue
        first = row & -row
        a = first.bit_length() - 1
        row ^= first
        while row:
            bit = row & -row
            b = bit.bit_length() - 1
            aa, bb = root(a), root(b)
            if aa != bb:
                parent[bb] = aa
            row ^= bit
    return len({root(i) for i in range(n)}) == 1


def _retired_c17_ctx(rng, G, idx):
    """Unused previous-shape search retained for provenance."""
    import math
    from itertools import combinations_with_replacement, permutations

    assert G.q == 17
    q = G.q
    width = 5 * q
    row_permutations = list(permutations(range(3)))

    # Each B variable keeps degree three or four despite zero cells.
    # A column uses at least two protograph check rows. Rewiring a
    # monomial into a doubleton changes parity without adding a gate
    # when compensated by a zero elsewhere in that column.
    column_types = [
        (a, b, c)
        for a in range(3) for b in range(3) for c in range(3)
        if a + b + c in (3, 4)
        and sum(t > 0 for t in (a, b, c)) >= 2
    ]
    patterns = set()
    for pattern in combinations_with_replacement(column_types, 5):
        totals = [sum(column[i] for column in pattern)
                  for i in range(3)]
        if min(totals) < 4 or max(totals) > 6 or sum(totals) > 18:
            continue
        if not any(0 in column for column in pattern):
            continue
        augmentation = [
            sum((column[i] & 1) << j
                for j, column in enumerate(pattern))
            for i in range(3)
        ]
        if gf2_rank(augmentation) != 3:
            continue
        canonical = min(
            tuple(sorted(tuple(column[p[i]] for i in range(3))
                         for column in pattern))
            for p in row_permutations
        )
        patterns.add(canonical)

    def card_priority(pattern):
        totals = [sum(column[i] for column in pattern)
                  for i in range(3)]
        return (sum(totals), max(totals) - min(totals),
                sum(0 in column for column in pattern), pattern)

    # These are different count architectures after explicit row/column
    # canonicalization, not saved-code automorphism images. Complete
    # Tanner equivalence remains the evaluator's decision.
    cards = sorted(patterns, key=card_priority)
    assert len(cards) >= max(20, N_SLOTS)
    design = idx
    mode = 0
    pattern = cards[design]
    sa = [[1] * 5 for _ in range(3)]
    sb = [[column[i] for column in pattern] for i in range(3)]
    assert _grid_weights(sa, sb) <= min(9, W_MAX)
    assert _no_dead_lines(sa, sb)
    assert any(count == 0 for row in sb for count in row)
    assert gf2_rank([sum((w & 1) << j for j, w in enumerate(row))
                     for row in sa]) == 1
    assert gf2_rank([sum((w & 1) << j for j, w in enumerate(row))
                     for row in sb]) == 3

    def grid(support):
        return [[rng.sample(G.elements, count) for count in row]
                for row in support]

    # Cache only actions returned by the construction helper. XOR and
    # block placement below use linearity, not independently coded
    # group left/right conventions.
    mask = (1 << q) - 1
    actions = {"A": {}, "B": {}}
    for g in G.elements:
        hx, _, _ = lifted_matrices(G, [[[g]]], [[[]]])
        _, hz, _ = lifted_matrices(G, [[[]]], [[[g]]])
        actions["A"][G.idx(g)] = tuple(row & mask for row in hx)
        actions["B"][G.idx(g)] = tuple(row & mask for row in hz)

    def component(grid_cells, side):
        rows = []
        for cell_row in grid_cells:
            out = [0] * q
            for j, cell in enumerate(cell_row):
                for g in cell:
                    block = actions[side][G.idx(g)]
                    for r, word in enumerate(block):
                        out[r] ^= word << (j * q)
            rows.extend(out)
        return rows

    def components(hx, hz):
        """Project A-left/B-right from the complete verified CSS lift."""
        mask = (1 << q) - 1
        ar = []
        for i in range(3):
            for r in range(q):
                row = hx[(i * 5) * q + r]
                packed = 0
                for j in range(5):
                    packed |= ((row >> (j * 5 * q)) & mask) << (j * q)
                ar.append(packed)
        bmask = (1 << width) - 1
        br = [hz[a * q + r] & bmask
              for a in range(3) for r in range(q)]
        return ar, br

    def columns(rows):
        cols = [0] * width
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                cols[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return cols

    def short_and_boundary(cols):
        """Exact component kernel<=6 exclusion and triple-boundary cost."""
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
        for i, x in enumerate(cols):
            for j in range(i):
                pair = x ^ cols[j]
                for h in range(j):
                    value = pair ^ cols[h]
                    if value in seen:
                        return None
                    seen.add(value)
                    weight = value.bit_count()
                    if weight < 6:
                        boundary += (6 - weight) ** 2
        return boundary

    def sampled_words(cols, side):
        """Verified kernel witnesses; minimum remains only an upper bound.

        Give finalists the same permutation stream within this slot.
        Count only distinct sampled short supports, not logical qubits.
        """
        srng = random.Random(81371 + 1009 * idx + int(side == "B"))
        order = list(range(width))
        best = width + 1
        short = set()
        for _ in range(32):
            srng.shuffle(order)
            pivots = {}
            basis = []
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
                if not syndrome:
                    candidates = [word]
                    candidates.extend(word ^ other for other in basis)
                    for v in candidates:
                        if not v:
                            continue
                        weight = v.bit_count()
                        best = min(best, weight)
                        if weight <= 12:
                            short.add(v)
                    basis.append(word)
        return best, len(short)

    def six_cycles(rows):
        """Exact Tanner six-cycles using three distinct variables."""
        nr = len(rows)
        overlap = [[0] * nr for _ in range(nr)]
        for i in range(nr):
            for j in range(i):
                overlap[i][j] = overlap[j][i] = (
                    rows[i] & rows[j]).bit_count()
        total = 0
        for i in range(nr):
            for j in range(i):
                x = overlap[i][j]
                if not x:
                    continue
                common = rows[i] & rows[j]
                for h in range(j):
                    y, z = overlap[j][h], overlap[i][h]
                    if y and z:
                        t = (common & rows[h]).bit_count()
                        total += x * y * z - t * (x + y + z) + 2 * t
        return total

    def cycle_cost(rows):
        """Exact C4 count and a cheap C6 proposal heuristic.

        The second statistic is exact when C4=0. Final refinement
        below always uses the exact six-cycle routine.
        """
        nr = len(rows)
        adjacency = [0] * nr
        four = 0
        for i, row in enumerate(rows):
            for j in range(i):
                common = (row & rows[j]).bit_count()
                four += common * (common - 1) // 2
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
        stars = sum(d * (d - 1) * (d - 2) // 6
                    for d in (c.bit_count() for c in columns(rows)))
        return four, max(0, triangles - stars)

    def clone(cells):
        return [[list(cell) for cell in row] for row in cells]

    def portfolio(side, support):
        """Anneal labels inside this slot's fixed support architecture."""
        positions = [(i, j) for i, row in enumerate(support)
                     for j, count in enumerate(row) if count]
        wanted_rank = 3 * q - 2 if side == "A" else 3 * q
        pool = []
        seen = set()
        serial = 0

        def energy(cost):
            return (64.0 * cost[0] + cost[1] / 6.0) / q

        def remember(cells, rows, cost):
            nonlocal serial
            if len(pool) >= 10 and cost > pool[-1][0]:
                return
            signature = tuple(
                tuple(sorted(G.idx(g) for g in cell))
                for row in cells for cell in row)
            if signature in seen:
                return
            seen.add(signature)
            # These component ranks are only a preliminary screen.
            # Complete verified CSS ranks still decide final k.
            if gf2_rank(rows) != wanted_rank:
                return
            serial += 1
            pool.append((cost, serial, clone(cells), list(rows)))
            pool.sort(key=lambda entry: entry[:2])
            del pool[10:]

        for restart in range(3):
            cells = grid(support)
            rows = component(cells, side)
            current = cycle_cost(rows)
            remember(cells, rows, current)
            for step in range(240):
                i, j = rng.choice(positions)
                old = cells[i][j]
                cells[i][j] = rng.sample(G.elements, support[i][j])
                proposed_rows = component(cells, side)
                proposed = cycle_cost(proposed_rows)
                delta = energy(proposed) - energy(current)
                temperature = 8.0 * (0.985 ** step) + 0.05
                accept = (delta <= 0 or rng.random() <
                          math.exp(-min(700.0, delta / temperature)))
                if not accept:
                    cells[i][j] = old
                    continue
                improved = proposed < current
                rows, current = proposed_rows, proposed
                if improved or step % 16 == 0:
                    remember(cells, rows, current)
            remember(cells, rows, current)
        return pool

    for batch in range(3):
        apool = portfolio("A", sa)
        bpool = portfolio("B", sb)
        pool = []
        trial = 0
        for acost, _, A, expected_a in apool:
            for bcost, _, B, expected_b in bpool:
                hx, hz, n = lifted_matrices(G, A, B)
                ar, br = components(hx, hz)
                # Confirm both cached action conventions against the
                # complete matrices that will actually be submitted.
                assert ar == expected_a and br == expected_b
                coarse = (max(acost[0], bcost[0]),
                          acost[0] + bcost[0],
                          max(acost[1], bcost[1]),
                          acost[1] + bcost[1])
                pool.append((coarse, trial, A, B, hx, hz, n, ar, br))
                trial += 1
        pool.sort(key=lambda item: (item[0], item[1]))

        finalists = []
        eligible = 0
        refined = {}

        def profile(rows, side):
            signature = side, tuple(rows)
            if signature not in refined:
                cols = columns(rows)
                boundary = short_and_boundary(cols)
                if boundary is None:
                    refined[signature] = None
                else:
                    distance, count = sampled_words(cols, side)
                    refined[signature] = (
                        boundary, distance, count, six_cycles(rows))
            return refined[signature]

        for item in pool:
            coarse, trial, A, B, hx, hz, n, ar, br = item
            # Augmentation and component ranks motivate 72 but never
            # replace ranks of the actual evaluator-convention matrices.
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (n == 578 and k == 72 and 8 * k < n):
                continue
            if not _connected_checks(hx, hz, n):
                continue
            eligible += 1
            ap = profile(ar, "A")
            bp = profile(br, "B") if ap is not None else None
            if ap is not None and bp is not None:
                ba, da, na, ca = ap
                bb, db, nb, cb = bp
                distance = min(da, db)
                boundary = max(ba, bb), ba + bb
                multiplicity = max(na, nb), na + nb
                # A sampled witness below twelve is a warning, not a
                # distance certificate. Above that floor minimize the
                # worse component's exposure, then combined exposure.
                key = (int(distance < 12), boundary, multiplicity,
                       coarse, max(ca, cb), ca + cb,
                       -min(distance, 14), abs(da - db), trial)
                finalists.append((key, item, k))
            if eligible >= 48:
                break

        finalists.sort(key=lambda entry: entry[0])
        for key, item, k in finalists:
            _, _, A, B, hx, hz, n, _, _ = item
            if _has_short_logical(hx, hz, n):
                continue
            assert css_ok(hx, hz)
            assert max(row.bit_count() for row in hx + hz) <= min(9, W_MAX)
            expected_cnots = 6 * q * 8 * (
                sum(map(sum, sa)) + sum(map(sum, sb)))
            assert 24480 <= expected_cnots <= 26928
            assert 6 * sum(row.bit_count() for row in hx + hz) == expected_cnots
            return {"cells": {"A": A, "B": B}, "design": design,
                    "mode": mode, "k": k, "screen": key,
                    "support_card": pattern,
                    "cnots": expected_cnots}

    raise ValueError("No gate-lean zero/doubleton k=72 filling passed screening")


def _retired_c43_ctx(rng, G, idx):
    """Localized spectral architectures and verified boundary screening.

    Field calculations propose labels only. All matrix actions, ranks,
    incidence screens and final acceptance use the verified helpers.
    No catalogue state persists between calls.
    """
    import math
    import hashlib
    from collections import Counter
    from itertools import combinations

    assert G.q == 43
    assert A_SHAPE == B_SHAPE == (2, 3)
    q = G.q
    width = 3 * q
    wanted = max(20, N_SLOTS)
    assert 0 <= idx < N_SLOTS

    def remainder(value, divisor):
        degree = divisor.bit_length() - 1
        while value and value.bit_length() - 1 >= degree:
            value ^= divisor << (value.bit_length() - 1 - degree)
        return value

    def polynomial_product(a, b):
        out = 0
        while b:
            if b & 1:
                out ^= a
            a <<= 1
            b >>= 1
        return out

    # ord_43(2)=14. Enumerate actual factors rather than trusting a
    # handwritten irreducible polynomial or a guessed spectral model.
    assert pow(2, 14, q) == 1
    assert all(pow(2, d, q) != 1 for d in (1, 2, 7))
    factors = [
        (1 << 14) | (middle << 1) | 1
        for middle in range(1 << 13)
        if remainder((1 << q) | 1,
                     (1 << 14) | (middle << 1) | 1) == 0
    ]
    assert len(factors) == 3
    product = 3
    for factor in factors:
        product = polynomial_product(product, factor)
    assert product == (1 << q) | 1
    modulus = min(factors)

    def multiply(a, b):
        out = 0
        while b:
            if b & 1:
                out ^= a
            b >>= 1
            a <<= 1
            if a & (1 << 14):
                a ^= modulus
        return out

    def power(a, exponent):
        out = 1
        while exponent:
            if exponent & 1:
                out = multiply(out, a)
            a = multiply(a, a)
            exponent >>= 1
        return out

    z = [power(2, a) for a in range(q)]
    assert len(set(z)) == q and power(2, q) == 1
    inverse = {
        d: power(1 ^ z[d], (1 << 14) - 2)
        for d in range(1, 22)
    }

    # Singleton action caches come exclusively from the scaffold.
    mask = (1 << q) - 1
    actions = {"A": {}, "B": {}}
    for g in G.elements:
        hx, _, _ = lifted_matrices(G, [[[g]]], [[[]]])
        _, hz, _ = lifted_matrices(G, [[[]]], [[[g]]])
        actions["A"][G.idx(g)] = tuple(row & mask for row in hx)
        actions["B"][G.idx(g)] = tuple(row & mask for row in hz)

    def component(cells, side):
        rows = []
        for cell_row in cells:
            out = [0] * q
            for j, cell in enumerate(cell_row):
                for g in cell:
                    block = actions[side][G.idx(g)]
                    for r, word in enumerate(block):
                        out[r] ^= word << (j * q)
            rows.extend(out)
        return rows

    def columns(rows):
        out = [0] * width
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return out

    def low4(cols):
        seen = {0}
        for value in cols:
            if value in seen:
                return True
            seen.add(value)
        for i, value in enumerate(cols):
            for j in range(i):
                syndrome = value ^ cols[j]
                if syndrome in seen:
                    return True
                seen.add(syndrome)
        return False

    def short_boundary(cols):
        """Exact kernel<=6 rejection and block-balanced triple exposure."""
        seen = {0}
        for value in cols:
            if value in seen:
                return None
            seen.add(value)
        for i, value in enumerate(cols):
            for j in range(i):
                syndrome = value ^ cols[j]
                if syndrome in seen:
                    return None
                seen.add(syndrome)
        total = 0
        loads = [0] * 3
        for i, value in enumerate(cols):
            for j in range(i):
                pair = value ^ cols[j]
                for h in range(j):
                    syndrome = pair ^ cols[h]
                    if syndrome in seen:
                        return None
                    seen.add(syndrome)
                    weight = syndrome.bit_count()
                    if weight <= 4:
                        penalty = (6 - weight) ** 2
                        total += penalty
                        for block in {i // q, j // q, h // q}:
                            loads[block] += penalty
        return max(loads), total

    def fingerprint(rows):
        """Rooted uncolored Tanner invariant, not full-code canonization.

        Lift translations are transitive within each component check
        orbit. Include both check-row orbits and discard equal profiles
        conservatively. Arbitrary full-code equivalence remains the
        evaluator's responsibility.
        """
        nr = len(rows)
        adjacency = [[] for _ in range(nr + width)]
        for c, syndrome in enumerate(columns(rows)):
            while syndrome:
                bit = syndrome & -syndrome
                r = bit.bit_length() - 1
                adjacency[r].append(nr + c)
                adjacency[nr + c].append(r)
                syndrome ^= bit
        results = []
        for root in (0, q):
            initial = [
                (int(v >= nr), len(adjacency[v]), int(v == root))
                for v in range(nr + width)
            ]
            palette = {key: t for t, key in enumerate(sorted(set(initial)))}
            colors = [palette[key] for key in initial]
            digest = hashlib.sha256()
            for _ in range(10):
                signatures = [
                    (colors[v],
                     tuple(sorted(colors[w] for w in adjacency[v])))
                    for v in range(nr + width)
                ]
                digest.update(repr(sorted(
                    Counter(signatures).items())).encode("ascii"))
                palette = {
                    key: t for t, key in enumerate(sorted(set(signatures)))
                }
                colors = [palette[key] for key in signatures]
            results.append(digest.hexdigest())
        return tuple(sorted(results))

    def clone(cells):
        return [[list(cell) for cell in row] for row in cells]

    # Normalize common row shifts in the field-ratio catalogue.
    # This removes obvious voltage gauges before graph screening.
    ratio_buckets = {}
    for d in range(1, 22):
        for e in range(1, 22):
            base = multiply(1 ^ z[e], inverse[d])
            images = [multiply(base, z[t]) for t in range(q)]
            ratio = min(images)
            if ratio == 1:
                # Globally proportional rows would lose all three
                # nontrivial sectors, not just the intended factor.
                continue
            t = images.index(ratio)
            entry = ((0, d), tuple(sorted((t, (t + e) % q))))
            ratio_buckets.setdefault(ratio, set()).add(entry)

    shift_cache = {}

    def shifted_min(value):
        if value not in shift_cache:
            shift_cache[value] = min(multiply(value, zz) for zz in z)
        return shift_cache[value]

    def ratio_orbit(value):
        # Frobenius automorphisms preserving the chosen spectral factor.
        images = []
        for _ in range(14):
            images.append(shifted_min(value))
            value = multiply(value, value)
        return min(images)

    ratios = [
        ratio for ratio, entries in sorted(ratio_buckets.items())
        if len(entries) >= 2 and ratio == ratio_orbit(ratio)
    ]
    assert ratios

    pair_values = {}
    all_pairs = []
    for a, b in combinations(range(q), 2):
        pair = (a, b)
        value = z[a] ^ z[b]
        pair_values.setdefault(value, []).append(pair)
        all_pairs.append((pair, value))
    quartet_cache = {}

    def quartets(target):
        """All distinct four-term realizations of this field value."""
        if target not in quartet_cache:
            found = set()
            for pair, value in all_pairs:
                for mate in pair_values.get(target ^ value, ()):
                    if pair >= mate or set(pair) & set(mate):
                        continue
                    found.add(tuple(sorted(pair + mate)))
            quartet_cache[target] = sorted(found)
        return quartet_cache[target]

    def encode(upper, lower):
        return [
            [[(a, 0) for a in cell] for cell in upper],
            [[(a, 0) for a in cell] for cell in lower],
        ]

    def architectures():
        # First offer fully doubleton architectures when three distinct
        # normalized columns share a field ratio.
        for ratio in ratios:
            entries = sorted(ratio_buckets[ratio])
            for chosen in combinations(entries, 3):
                yield encode([entry[0] for entry in chosen],
                             [entry[1] for entry in chosen])
        # A single quartet greatly expands the non-orbit catalogue
        # without exceeding either extraction row-weight budget.
        for ratio in ratios:
            entries = sorted(ratio_buckets[ratio])
            for first, second in combinations(entries, 2):
                used = {first[0][1], second[0][1]}
                for d in range(1, 22):
                    if d in used:
                        continue
                    target = multiply(ratio, 1 ^ z[d])
                    for quartet in quartets(target):
                        yield encode(
                            [first[0], second[0], (0, d)],
                            [first[1], second[1], quartet])

    # Build the same deterministic catalogue in every pure call.
    # Distinct fingerprints define deliberate structural cards.
    # Each slot owns three disjoint catalogue positions for retries.
    bank = []
    identities = set()
    for serial, A in enumerate(architectures()):
        ar = component(A, "A")
        if gf2_rank(ar) != 70:
            continue
        ac = columns(ar)
        if low4(ac) or not _connected_checks(ar, [], width):
            continue
        identity = fingerprint(ar)
        if identity in identities:
            continue
        identities.add(identity)
        bank.append((_four_cycles(ar), identity, serial, A, ar))
        if len(bank) >= 3 * wanted:
            break
    bank.sort(key=lambda item: item[:3])
    if len(bank) < 3 * wanted:
        raise ValueError("insufficient distinct localized spectral cards")

    chosen = None
    for position in (idx, idx + wanted, idx + 2 * wanted):
        a4, identity, serial, A, ar = bank[position]
        abound = short_boundary(columns(ar))
        if abound is not None:
            chosen = (a4, identity, serial, A, ar, abound)
            break
    if chosen is None:
        raise ValueError("owned spectral cards contain component words<=6")
    a4, identity, serial, A, ar, abound = chosen

    def cycle_cost(rows):
        """C4 and a C6 proposal heuristic; neither is a pL estimate."""
        nr = len(rows)
        adjacency = [0] * nr
        four = 0
        for i, row in enumerate(rows):
            for j in range(i):
                common = (row & rows[j]).bit_count()
                four += common * (common - 1) // 2
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
        stars = sum(d * (d - 1) * (d - 2) // 6
                    for d in (c.bit_count() for c in columns(rows)))
        return four, max(0, triangles - stars)

    def sampled_words(rows, side):
        """Actual kernel witnesses; sampled minimum is only an upper bound."""
        srng = random.Random(4301701 + 1009 * idx + int(side == "B"))
        cols = columns(rows)
        order = list(range(width))
        best = width + 1
        short = set()
        for _ in range(40):
            srng.shuffle(order)
            pivots = {}
            basis = []
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
                if not syndrome:
                    candidates = [word]
                    candidates.extend(word ^ other for other in basis)
                    for v in candidates:
                        if v:
                            weight = v.bit_count()
                            best = min(best, weight)
                            if weight <= 18:
                                short.add(v)
                    basis.append(word)
        return best, len(short)

    da, na = sampled_words(ar, "A")

    def draw_B():
        return [[rng.sample(G.elements, 2) for _ in range(3)]
                for _ in range(2)]

    def energy(cost):
        return (80.0 * cost[0] + cost[1] / 8.0) / q

    for batch in range(3):
        pool = []
        seen = set()
        sequence = 0

        def remember(B, br, cost):
            nonlocal sequence
            if len(pool) >= 12 and cost > pool[-1][0]:
                return
            signature = tuple(
                tuple(sorted(G.idx(g) for g in cell))
                for row in B for cell in row)
            if signature in seen:
                return
            seen.add(signature)
            if gf2_rank(br) != 84:
                return
            sequence += 1
            pool.append((cost, sequence, clone(B), list(br)))
            pool.sort(key=lambda item: item[:2])
            del pool[12:]

        for restart in range(3):
            B = draw_B()
            br = component(B, "B")
            current = cycle_cost(br)
            remember(B, br, current)
            for step in range(200):
                i, j = rng.randrange(2), rng.randrange(3)
                old = B[i][j]
                B[i][j] = rng.sample(G.elements, 2)
                proposed_rows = component(B, "B")
                proposed = cycle_cost(proposed_rows)
                delta = energy(proposed) - energy(current)
                temperature = 10.0 * (0.985 ** step) + 0.05
                if (delta <= 0 or rng.random() <
                        math.exp(-min(700.0, delta / temperature))):
                    br, current = proposed_rows, proposed
                    if step % 12 == 0 or delta < 0:
                        remember(B, br, current)
                else:
                    B[i][j] = old
            remember(B, br, current)

        finalists = []
        for cost, number, B, br in pool:
            if not _connected_checks(br, [], width):
                continue
            bbound = short_boundary(columns(br))
            if bbound is None:
                continue
            db, nb = sampled_words(br, "B")
            distance = min(da, db)
            # A sampled short witness is a warning, not a certificate.
            # Beyond the warning floor, minimize balanced exposure
            # rather than maximizing an optimistic distance estimate.
            key = (
                int(distance < 18),
                max(abound[0], bbound[0]), abound[0] + bbound[0],
                max(abound[1], bbound[1]), abound[1] + bbound[1],
                max(na, nb), na + nb,
                max(a4, cost[0]), a4 + cost[0], cost[1],
                -min(distance, 20), abs(da - db), number,
            )
            finalists.append((key, B, br))
        finalists.sort(key=lambda item: item[0])

        for key, B, expected_b in finalists:
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if n != 559 or k != 69 or 8 * k >= n:
                continue
            actual_a = [
                sum(((hx[(i * 3) * q + r] >> (j * 3 * q)) & mask)
                    << (j * q) for j in range(3))
                for i in range(2) for r in range(q)
            ]
            actual_b = [
                hz[a * q + r] & ((1 << width) - 1)
                for a in range(2) for r in range(q)
            ]
            assert actual_a == ar and actual_b == expected_b
            if not _connected_checks(hx, hz, n):
                continue
            if _has_short_logical(hx, hz, n):
                continue
            sa = [[len(cell) for cell in row] for row in A]
            sb = [[len(cell) for cell in row] for row in B]
            assert _no_dead_lines(sa, sb)
            assert _grid_weights(sa, sb) <= min(12, W_MAX)
            assert all(len(set(cell)) == len(cell)
                       for grid in (A, B) for row in grid for cell in row)
            assert css_ok(hx, hz)
            assert max(row.bit_count() for row in hx + hz) <= W_MAX
            expected_cnots = 6 * q * 5 * (
                sum(map(sum, sa)) + sum(map(sum, sb)))
            assert expected_cnots in (30960, 33540)
            assert 6 * sum(row.bit_count() for row in hx + hz) == expected_cnots
            return {
                "cells": {"A": A, "B": B},
                "design": (idx, serial, identity),
                "spectral_factor": modulus,
                "k": k, "screen": key, "cnots": expected_cnots,
            }

    raise ValueError("no connected k=69 spectral pair passed verified screens")


def NEW_CTX(rng, G, idx):
    """Balanced count contracts and full-code noisy-syndrome holdouts."""
    import math
    from itertools import combinations_with_replacement

    q = G.q
    assert q == 19 and A_SHAPE == B_SHAPE == (2, 5)
    width, mask = 5*q, (1 << q)-1

    # Canonicalize count cards under row exchange and column permutation.
    # Different cards are deliberate structures, not saved labeling images.
    types = [(a, d-a) for d in (3, 4) for a in range(d+1)]
    cards = set()
    for columns_card in combinations_with_replacement(types, 5):
        totals = [sum(c[i] for c in columns_card) for i in range(2)]
        if not all(7 <= t <= 8 for t in totals):
            continue
        if any(sum(c[i] > 0 and c[1-i] == 0
                   for c in columns_card) > 1 for i in range(2)):
            continue
        parity = [sum((c[i] & 1) << j
                      for j, c in enumerate(columns_card))
                  for i in range(2)]
        if gf2_rank(parity) != 2:
            continue
        direct = tuple(sorted(columns_card))
        swapped = tuple(sorted((b, a) for a, b in columns_card))
        cards.add(min(direct, swapped))
    book = sorted(cards, key=lambda card: (
        sum(map(sum, card)), sum(0 in c for c in card), card))
    contracts = list(combinations_with_replacement(range(len(book)), 2))
    contracts.sort(key=lambda pair: (
        sum(map(sum, book[pair[0]])) + sum(map(sum, book[pair[1]])),
        pair))
    assert len(contracts) >= 20
    phase, position = divmod(idx, len(contracts))
    pa, pb = contracts[position]
    sa = [[c[i] for c in book[pa]] for i in range(2)]
    sb = [[c[i] for c in book[pb]] for i in range(2)]
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert all(0 <= t <= CELL_MAX
               for support in (sa, sb) for row in support for t in row)
    # Relative ratios two and three are not common multiplier images
    # when the first two distinguished doubleton differences are fixed.
    ratio = 2 + phase
    if ratio >= q:
        raise ValueError("relative-ratio contract budget exhausted")

    actions = {"A": {}, "B": {}}
    for g in G.elements:
        hx, _, _ = lifted_matrices(G, [[[g]]], [[[]]])
        _, hz, _ = lifted_matrices(G, [[[]]], [[[g]]])
        actions["A"][G.idx(g)] = tuple(r & mask for r in hx)
        actions["B"][G.idx(g)] = tuple(r & mask for r in hz)

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def component(grid, side):
        rows = []
        for grid_row in grid:
            packed = [0]*q
            for j, cell in enumerate(grid_row):
                for g in cell:
                    for r, word in enumerate(actions[side][G.idx(g)]):
                        packed[r] ^= word << (j*q)
            rows.extend(packed)
        return rows

    def columns(rows, nc):
        cols = [0]*nc
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                cols[bit.bit_length()-1] |= 1 << r
                row ^= bit
        return cols

    def clean4(cols):
        seen = {0}
        for value in cols:
            if value in seen:
                return False
            seen.add(value)
        for i, value in enumerate(cols):
            for other in cols[:i]:
                syndrome = value ^ other
                if syndrome in seen:
                    return False
                seen.add(syndrome)
        return True

    def cycles(rows):
        adjacency = [0]*len(rows)
        four = 0
        for i, row in enumerate(rows):
            for j in range(i):
                common = (row & rows[j]).bit_count()
                four += common*(common-1)//2
                if common:
                    adjacency[i] |= 1 << j
                    adjacency[j] |= 1 << i
        triangles = 0
        for i in range(len(rows)):
            lower = adjacency[i] & ((1 << i)-1)
            while lower:
                bit = lower & -lower
                j = bit.bit_length()-1
                triangles += (adjacency[i] & adjacency[j]
                              & ((1 << j)-1)).bit_count()
                lower ^= bit
        stars = sum(d*(d-1)*(d-2)//6
                    for d in (c.bit_count()
                              for c in columns(rows, width)))
        # Exact C6 if C4=0; otherwise only a proposal statistic.
        return four, max(0, triangles-stars)

    def census56(cols):
        """Exact five/six counts after clean4; separate boundary surrogate."""
        pairs = {}
        for i, x in enumerate(cols):
            for y in cols[:i]:
                value = x ^ y
                pairs[value] = pairs.get(value, 0)+1
        triples = {}
        five = six = boundary = 0
        loads = [0]*5
        for i, x in enumerate(cols):
            for j in range(i):
                pair = x ^ cols[j]
                for h in range(j):
                    value = pair ^ cols[h]
                    five += pairs.get(value, 0)
                    six += triples.get(value, 0)
                    triples[value] = triples.get(value, 0)+1
                    weight = value.bit_count()
                    if weight <= 4:
                        penalty = (5-weight)**2
                        boundary += penalty
                        for block in {i//q, j//q, h//q}:
                            loads[block] += penalty
        # With kernels<=4 excluded, matching supports are disjoint.
        assert five % 10 == 0 and six % 10 == 0
        return five//10, six//10, max(loads), boundary

    def sampled_words(cols, side):
        """Actual kernel witnesses; sampled minima are upper bounds."""
        srng = random.Random(190551091 + 1009*idx + (side == "B"))
        order = list(range(width))
        minimum = width+1
        short = set()
        for _ in range(24):
            srng.shuffle(order)
            pivots, basis = {}, []
            for c in order:
                syndrome, word = cols[c], 1 << c
                while syndrome:
                    p = syndrome.bit_length()-1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = syndrome, word
                        break
                    syndrome ^= old[0]
                    word ^= old[1]
                if not syndrome:
                    basis.append(word)
            for i, word in enumerate(basis):
                candidates = [word] + [word ^ other for other in basis[:i]]
                for value in candidates:
                    if value:
                        weight = value.bit_count()
                        minimum = min(minimum, weight)
                        if weight <= 10:
                            short.add(value)
        return minimum, len(short)

    def portfolio(side, support):
        sites = [(i, j) for i, row in enumerate(support)
                 for j, count in enumerate(row) if count]
        doubletons = [site for site in sites
                      if support[site[0]][site[1]] == 2]
        assert side != "A" or len(doubletons) >= 2
        fixed = ({doubletons[0]: 1, doubletons[1]: ratio}
                 if side == "A" else {})

        def draw_cell(i, j):
            count = support[i][j]
            if (i, j) in fixed:
                origin = rng.randrange(q)
                return [(origin,), ((origin+fixed[i, j]) % q,)]
            return rng.sample(G.elements, count)

        def draw():
            return [[draw_cell(i, j) if count else []
                     for j, count in enumerate(row)]
                    for i, row in enumerate(support)]

        def assess(grid):
            rows = component(grid, side)
            cost = cycles(rows)
            deficit = 2*q-gf2_rank(rows)
            energy = (10000*deficit + 100*cost[0] + cost[1]/5)/q
            return rows, cost, deficit, energy

        pool, seen = [], set()
        serial = 0

        def offer(grid, state):
            nonlocal serial
            rows, cost, deficit, _ = state
            if deficit or (len(pool) >= 10 and cost > pool[-1][0]):
                return
            identity = tuple(tuple(sorted(G.idx(g) for g in cell))
                             for row in grid for cell in row)
            if identity in seen:
                return
            seen.add(identity)
            serial += 1
            pool.append((cost, serial, clone(grid), list(rows)))
            pool.sort(key=lambda item: item[:2])
            del pool[10:]

        for restart in range(4):
            grid = draw()
            state = assess(grid)
            offer(grid, state)
            for step in range(220):
                i, j = rng.choice(sites)
                old = grid[i][j]
                grid[i][j] = draw_cell(i, j)
                proposed = assess(grid)
                delta = proposed[-1]-state[-1]
                temperature = 10*(0.98**step)+0.08
                if delta <= 0 or rng.random() < math.exp(
                        -min(700, delta/temperature)):
                    state = proposed
                    if delta < 0 or step % 12 == 0:
                        offer(grid, state)
                else:
                    grid[i][j] = old
            offer(grid, state)

        finalists = []
        for cost, number, grid, rows in pool:
            cols = columns(rows, width)
            if not clean4(cols) or not _connected_checks(rows, [], width):
                continue
            five, six, peak, boundary = census56(cols)
            d, multiplicity = sampled_words(cols, side)
            # Above eight, do not reward a larger optimistic bound.
            key = (five, six, int(d < 8), peak, boundary,
                   multiplicity, cost[0], cost[1], -min(d, 10))
            finalists.append((key, number, grid, rows))
        finalists.sort(key=lambda item: item[:2])
        return finalists[:3]

    def noisy_probe(checks, stabilizers, n, seed, shots):
        """Two fault intervals with one noisy and one perfect syndrome.

        This is an uncalibrated BP surrogate, not ASC, relay-BP or LSD.
        Settled residuals are classified modulo the verified stabilizers.
        """
        import numpy as np

        def sites(row):
            out = []
            while row:
                bit = row & -row
                out.append(bit.bit_length()-1)
                row ^= bit
            return out

        m = len(checks)
        variables = 2*n+m
        augmented = [row | (1 << (2*n+r))
                     for r, row in enumerate(checks)]
        augmented += [row | (row << n) for row in checks]
        neighborhoods = [sites(row) for row in augmented]
        row_width = max(map(len, neighborhoods))
        incidence = np.full((2*m, row_width), variables, dtype=np.int32)
        for r, ns in enumerate(neighborhoods):
            incidence[r, :len(ns)] = ns
        flat = incidence.ravel()
        valid = flat != variables
        edges = len(flat)
        degree = np.bincount(flat[valid], minlength=variables)
        assert np.all(degree > 0)
        variable_edges = np.full(
            (variables+1, int(degree.max())), edges, dtype=np.int32)
        used = np.zeros(variables, dtype=np.int32)
        for edge in np.flatnonzero(valid):
            c = int(flat[edge])
            variable_edges[c, used[c]] = edge
            used[c] += 1

        sensed = np.array([c.bit_count() for c in columns(checks, n)])
        opposite = np.array([c.bit_count() for c in columns(stabilizers, n)])
        levels = np.array((0.65, 1.0, 1.35))[np.arange(shots) % 3]
        data_mu = 3*0.002*(2+sensed+opposite)*2/3
        measure_mu = np.array([
            3*0.002*(row.bit_count()+2)*2/3 for row in checks])
        data_p = 0.5*(1-np.exp(-2*levels[:, None]*data_mu[None, :]))
        measure_p = 0.5*(1-np.exp(-2*levels[:, None]*measure_mu[None, :]))
        channel = np.random.default_rng(seed)
        error = np.zeros((shots, variables+1), dtype=bool)
        error[:, :n] = channel.random((shots, n)) < data_p
        error[:, n:2*n] = channel.random((shots, n)) < data_p
        error[:, 2*n:variables] = channel.random((shots, m)) < measure_p

        # Random proper suffixes are possible spread patterns, not
        # hooks certified for the circuit's actual ASC gate ordering.
        for row in stabilizers:
            order = channel.permutation(sites(row))
            event_p = levels*3*0.002*len(order)*2/3
            for interval in range(2):
                for shot in np.flatnonzero(channel.random(shots) < event_p):
                    cut = int(channel.integers(1, len(order)))
                    error[shot, interval*n+order[cut:]] ^= True

        marginal_mu = data_mu + 3*0.002*opposite*0.35
        marginal_p = np.clip(0.5*(1-np.exp(
            -2*levels[:, None]*marginal_mu[None, :])), 0.002, 0.25)
        probabilities = np.concatenate(
            (marginal_p, marginal_p, np.clip(measure_p, 0.002, 0.25)),
            axis=1)
        prior = np.full((shots, variables+1), 30.0)
        prior[:, :variables] = np.log((1-probabilities)/probabilities)
        target = (error[:, flat].reshape(
            shots, 2*m, row_width).sum(axis=2) & 1).astype(bool)
        target_sign = np.where(target, -1.0, 1.0)
        messages = prior[:, flat].copy()
        messages[:, ~valid] = 30.0
        true_final = error[:, :n] ^ error[:, n:2*n]
        pivots = _row_span(stabilizers)
        wrong = late = work = 0

        for iteration in range(1, 33):
            count = len(prior)
            shaped = messages.reshape(count, 2*m, row_width)
            signs = np.where(shaped < 0, -1.0, 1.0)
            logs = np.log(np.clip(
                np.tanh(np.abs(shaped)/2), 1e-12, 1-1e-8))
            product = np.minimum(
                np.exp(logs.sum(axis=2, keepdims=True)-logs), 1-1e-8)
            outgoing = (
                (np.log1p(product)-np.log1p(-product))
                * signs * signs.prod(axis=2, keepdims=True)
                * target_sign[:, :, None]).reshape(count, edges)
            outgoing[:, ~valid] = 0.0
            padded = np.zeros((count, edges+1))
            padded[:, :edges] = outgoing
            posterior = prior + padded[:, variable_edges].sum(axis=2)
            posterior[:, variables] = 30.0
            hard = posterior < 0
            syndrome = (hard[:, flat].reshape(
                count, 2*m, row_width).sum(axis=2) & 1).astype(bool)
            settled = np.all(syndrome == target, axis=1)
            final = hard[:, :n] ^ hard[:, n:2*n]
            for local in np.flatnonzero(settled):
                word = 0
                for c in np.flatnonzero(final[local] ^ true_final[local]):
                    word |= 1 << int(c)
                wrong += int(not _in_row_span(word, pivots))
                late += int(iteration > 12)
                work += iteration
            keep = ~settled
            if not np.any(keep):
                return wrong, 0, late, work
            damping = 0.25 if iteration <= 12 else 0.50
            messages = np.clip(
                damping*messages
                + (1-damping)*(posterior[:, flat]-outgoing), -30, 30)[keep]
            messages[:, ~valid] = 30.0
            prior = prior[keep]
            target = target[keep]
            target_sign = target_sign[keep]
            true_final = true_final[keep]
        unresolved = len(prior)
        return wrong+unresolved, unresolved, late+unresolved, work+33*unresolved

    def quality(x, z):
        return tuple(v for a, b in zip(x, z)
                     for v in (max(a, b), a+b))

    seed = 190551701 + 1009*idx
    for batch in range(3):
        aa, bb = portfolio("A", sa), portfolio("B", sb)
        options = []
        for ak, at, A, ar in aa:
            for bk, bt, B, br in bb:
                geometry = tuple(v for a, b in zip(ak, bk)
                                 for v in (max(a, b), a+b))
                options.append((geometry, at, bt, A, B, ar, br))
        options.sort(key=lambda item: item[:3])
        training = []
        for number, option in enumerate(options[:6]):
            geometry, at, bt, A, B, ar, br = option
            hx, hz, n = lifted_matrices(G, A, B)
            k = n-gf2_rank(hx)-gf2_rank(hz)
            if n != 551 or k != 171:
                continue
            actual_a = [
                sum(((hx[(i*5)*q+r] >> (j*5*q)) & mask) << (j*q)
                    for j in range(5))
                for i in range(2) for r in range(q)]
            actual_b = [row & ((1 << width)-1) for row in hz[:2*q]]
            assert actual_a == ar and actual_b == br
            if not _connected_checks(hx, hz, n):
                continue
            if _has_short_logical(hx, hz, n):
                continue
            assert css_ok(hx, hz)
            x = noisy_probe(hz, hx, n, seed, 24)
            z = noisy_probe(hx, hz, n, seed+1, 24)
            training.append((quality(x, z)+geometry+(number,),
                             number, option, hx, hz, x, z))
            if len(training) >= 3:
                break
        training.sort(key=lambda item: item[:2])
        finalists = []
        for _, number, option, hx, hz, tx, tz in training[:2]:
            geometry, at, bt, A, B, ar, br = option
            x = noisy_probe(hz, hx, 551, seed+1000003, 48)
            z = noisy_probe(hx, hz, 551, seed+1000004, 48)
            key = quality(x, z)+quality(tx, tz)+geometry+(number,)
            finalists.append((key, A, B, hx, hz, x, z, tx, tz))
        if finalists:
            finalists.sort(key=lambda item: item[0])
            key, A, B, hx, hz, x, z, tx, tz = finalists[0]
            assert lifted_k(G, A, B) == 171
            assert 0.275 <= 171/551 < 0.325
            assert max(row.bit_count() for row in hx+hz) <= min(12, W_MAX)
            cnots = 6*sum(row.bit_count() for row in hx+hz)
            expected = 6*q*7*(sum(map(sum, sa))+sum(map(sum, sb)))
            assert cnots == expected and 23940 <= cnots <= 25536
            assert all(len(cell) == len(set(cell))
                       for grid in (A, B) for row in grid for cell in row)
            return {"cells": {"A": A, "B": B}, "k": 171,
                    "count_contract": (pa, pb), "ratio": ratio,
                    "training_surrogate": (tx, tz),
                    "holdout_surrogate": (x, z),
                    "screen": key, "cnots": cnots}
    raise ValueError("no full-rank C19 count-contract pair passed screening")


def CELL(rng, G, side, i, j, ctx):
    """Emit the complete holdout-selected cell without further draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: d2s=6 does not establish balanced circuit protection: parent slot 56 had 107/1250 failed Z shots versus 53/200000 failed X shots, with worse-basis s=3.973.
# REFUTED: Larger screened distance necessarily raises this bed's measured score: zero-failure slot 58 had d2s=22,k=45,s=7.165, below zero-failure slot 20 with d2s=20,k=58,s=7.276; the sample ceilings, not distance, limited both.
# REFUTED: k=66, check weight ten and d2s=15 certify zero-failure memory on this order-42 bed: slot25 had four Z failures, 47 logical flips and s=6.888, while slots2,5,13,28 with the same reported parameters had zero failures and s=7.332.
# REFUTED: This screened sparse-A-tripleton family improves the lighter order-17 bed: fitness instead fell from 7.2396 to 6.7924, and slot22 had 23 X failures/200000 with 623 logical flips despite d2s=14; supports and labels both changed, so this does not isolate a causal term-count effect.
# REFUTED: k=72, check weight nine and d2s=14 guarantee ceiling-level memory on C17 (3,5)x(3,5): current slot59 had two X failures, 39 collective logical flips and s=7.078, while slot44 with the same reported parameters saturated at s=7.370.
# REFUTED: k=69, check weight twelve and d2s=16 certify zero-failure C43 memory: slot42 had one X failure/200000 flipping27 logicals and s=7.172, while slots26,31,32,40 with those reported parameters saturated at s=7.351.
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
