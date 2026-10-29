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
CONCEPT = ("Cyclic C17, order 17, shapes (2,5)x(2,6): indexed sum-free gap contracts and balanced lower-load components should raise worse-basis circuit fitness beyond the cell-0.350 incumbent 5.1107. Reduced decoder burden and extraction exposure, not a larger distance upper bound, are the falsifiable objectives.")
GROUP = ("Keep the parent's cyclic construction rule and tuple encoding; retune only the order to 17 for the new size factor 34, whose largest legal lift gives n=578. C17 (3,5)x(3,5) has reached about 7.3, but C17 (2,5)x(2,6) only about 4.8: neither establishes an order-only advantage. Verified component and complete CSS ranks decide acceptance.")
SHAPES = ("Target 0.350 with (2,5)x(2,6): kb=204 and guaranteed rate 12/34=0.352941. A has 85 variables/34 checks and B has 102 variables/34 checks; require full component row ranks and actual k=204. Compared with the parent's 65/26 and 104/39 components, this lengthens A and reduces B's check count. Parent failures reached every logical qubit; diffuse Acol/Bcol and Acol/Brow enrichment, including Zchk:Acol3/Brow1, does not establish one persistent repair block.")
FILLING = ("Twenty briefs cross four count contracts with five canonical joint gap geometries; extra harness batches consume further geometries, not relabeled completed codes. A14/B18, A15/B18, A15/B17 and A16/B16 have maximum CSS weight twelve and six-round CNOT counts 24276,25092,24378,24480, versus the parent's measured 26832. Distinct sum-free row gaps suppress forced local cycles without requiring disjoint A/B palettes. One mixed-row degree-two B column is a forest; two must form one verified length-34 incidence cycle. Helper-built component screens retain several geometries, then complete CSS pairings face a fixed-bank two-interval benchmark with noisy intermediate checks, perfect final checks, degree-scaled faults, opposite-check suffix faults and stabilizer-aware BP residuals. This is not ASC scheduling, six-round history, relay-BP/LSD or measured pL. Target all five parental failure caps, collective flips and slot32's 103/8750 X failures, rather than infer reliability from d2s.")
FAMILY_NAME = "r035_C17_25x26_sumfree_contracts_temporal_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 17

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C17 requires order 17")
    els = [(a,) for a in range(q)]

    def red(g):
        return (int(g[0]) % q,)

    def mul(g, h):
        return ((g[0] + h[0]) % q,)

    def inv(g):
        return ((-g[0]) % q,)

    def idx(g):
        return red(g)[0]

    return dict(elements=els, ident=(0,), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 5)               # (ra, ca)
B_SHAPE = (2, 6)               # (rb, cb)


def _design_supports(idx):
    """Four extraction/protection contracts, each with five geometry cards."""
    contracts = (
        (((2, 2, 1, 1, 1), (1, 1, 2, 2, 2)),
         ((2, 2, 2, 1, 1, 1), (1, 1, 1, 2, 2, 2))),
        (((2, 2, 1, 1, 1), (1, 1, 2, 2, 2)),
         ((2, 2, 1, 1, 1, 1), (1, 1, 2, 2, 2, 1))),
        (((2, 2, 1, 1, 2), (1, 1, 2, 2, 2)),
         ((2, 2, 1, 1, 1, 1), (1, 1, 2, 2, 1, 1))),
        (((2, 2, 1, 1, 1), (1, 1, 2, 2, 1)),
         ((2, 2, 2, 1, 1, 1), (1, 1, 1, 2, 2, 2))),
    )
    a, b = contracts[(idx % 20) // 5]
    return [list(row) for row in a], [list(row) for row in b]


def _gap_contract(G, idx, sa, sb):
    """Prescribe distinct joint geometries before optimizing cell origins.

    Canonicalization removes common multipliers and component row/column
    copies of these relative-gap cards. It is not a certificate against
    arbitrary isomorphism of the complete quantum Tanner graph.
    """
    from itertools import combinations, permutations

    q = G.q
    assert q == 17
    model = (idx % 20) // 5
    wanted = 5 * (idx // 20) + idx % 5
    supports = {"A": sa, "B": sb}

    def direction(value):
        value %= q
        return min(value, q - value)

    def clean(gaps):
        signed = tuple(v for d in gaps for v in (d, -d))
        # Repeated choices are included: also exclude relations 2a=+/-b.
        # This excludes forced same-check-orbit triangles, not all C6.
        return all((a + b + c) % q
                   for a in signed for b in signed for c in signed)

    books = {
        number: [gaps for gaps in combinations(range(1, 9), number)
                 if clean(gaps)]
        for number in range(4)
    }
    assert all(books.values())

    def canonical(steps):
        images = []
        for multiplier in range(1, q):
            sides = []
            for side in ("A", "B"):
                grid = supports[side]
                row_images = []
                for order in permutations(range(len(grid))):
                    columns = [
                        tuple((grid[i][j],
                               direction(multiplier * steps[side][i][j]))
                              for i in order)
                        for j in range(len(grid[0]))
                    ]
                    row_images.append(tuple(sorted(columns)))
                sides.append(min(row_images))
            images.append(tuple(sides))
        return min(images)

    brng = random.Random(170526031 + 1009 * model)
    seen = set()
    number = 0
    for proposal in range(12000):
        steps = {}
        for side, grid in supports.items():
            out = []
            for row in grid:
                sites = [j for j, count in enumerate(row) if count == 2]
                gaps = list(brng.choice(books[len(sites)]))
                brng.shuffle(gaps)
                values = [0] * len(row)
                for j, gap in zip(sites, gaps):
                    values[j] = gap
                out.append(values)
            steps[side] = out

        # A degree-four column must not repeat its top/bottom direction.
        if any(sa[0][j] == sa[1][j] == 2
               and steps["A"][0][j] == steps["A"][1][j]
               for j in range(len(sa[0]))):
            continue
        key = canonical(steps)
        if key in seen:
            continue
        seen.add(key)
        if number == wanted:
            return steps, key
        number += 1
    raise ValueError("indexed sum-free joint gap catalogue exhausted")


def _memory_cell(rng, G, count, available):
    """A full zero, singleton or doubleton cell with a prescribed gap."""
    if count == 0:
        return []
    g = G.pick(rng)
    if count == 1:
        return [g]
    assert count == 2 and available
    d = rng.choice(available)
    return [g, G.mul(g, (d,))]


def _memory_gap(cell, q):
    if len(cell) != 2:
        return None
    d = (cell[1][0] - cell[0][0]) % q
    return min(d, q - d)


def _memory_grid(rng, G, supports, geometry):
    """Fresh origins realize the slot's prescribed full-cell gap geometry."""
    grid = []
    for i, counts in enumerate(supports):
        row = []
        for j, count in enumerate(counts):
            available = [geometry[i][j]] if count == 2 else []
            cell = _memory_cell(rng, G, count, available)
            assert count != 2 or _memory_gap(cell, G.q) == geometry[i][j]
            row.append(cell)
        grid.append(row)
    return grid


def _shape_component_rows(G, side, grid):
    """Use zero probes only to extract the scaffold's exact component."""
    if side == "A":
        rows = lifted_matrices(G, grid, [[[]]])[0]
    else:
        rows = lifted_matrices(G, [[[]]], grid)[1]
    mask = (1 << (len(grid[0]) * G.q)) - 1
    return [row & mask for row in rows]


def _shape_columns(rows, n):
    cols = [0] * n
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return cols


def _shape_short_counts(cols):
    """Kernel collision statistics through weight four.

    Counts may overcount partitions. A zero tuple certifies absence
    of component kernel words of weights one through four.
    """
    singles = {}
    one = two = three = four = 0
    for value in cols:
        one += int(value == 0)
        two += singles.get(value, 0)
        singles[value] = singles.get(value, 0) + 1
    pairs = {}
    for i, value in enumerate(cols):
        for other in cols[:i]:
            syndrome = value ^ other
            three += singles.get(syndrome, 0)
            four += pairs.get(syndrome, 0)
            pairs[syndrome] = pairs.get(syndrome, 0) + 1
    return one, two, three, four


def _shape_higher(cols):
    """Triple boundaries and weight-five/six collision exposure.

    These are circuit-performance surrogates, not measured pL.
    When shorter kernel words are absent, collisions correspond to
    partitions of weight-five or weight-six component words.
    """
    pairs = set()
    for i, value in enumerate(cols):
        for other in cols[:i]:
            pairs.add(value ^ other)
    triples = {}
    boundary = five = six = 0
    for i in range(len(cols)):
        for j in range(i):
            pair = cols[i] ^ cols[j]
            for h in range(j):
                syndrome = pair ^ cols[h]
                weight = syndrome.bit_count()
                if weight < 5:
                    boundary += (5 - weight) ** 2
                five += int(syndrome in pairs)
                six += triples.get(syndrome, 0)
                triples[syndrome] = triples.get(syndrome, 0) + 1
    return boundary, five, six


def _memory_witnesses(rng, cols, q, trials):
    """Sample actual component kernel words and reduce their multiplicity.

    These observations give upper bounds and sampled exposure, not
    distance certificates or circuit failure probabilities.
    """
    order = list(range(len(cols)))
    best = len(cols) + 1
    short = set()
    for _ in range(trials):
        rng.shuffle(order)
        pivots = {}
        basis = []
        for c in order:
            syndrome, word = cols[c], 1 << c
            while syndrome:
                p = syndrome.bit_length() - 1
                if p not in pivots:
                    pivots[p] = (syndrome, word)
                    break
                old_s, old_w = pivots[p]
                syndrome ^= old_s
                word ^= old_w
            if not syndrome and word:
                basis.append(word)
                weight = word.bit_count()
                best = min(best, weight)
                if weight <= 10:
                    short.add(word)

        # XORs of kernel words remain verified kernel witnesses.
        for sweep in range(2):
            rng.shuffle(basis)
            for i, word in enumerate(basis):
                weight = word.bit_count()
                for other in basis[:i]:
                    proposal = word ^ other
                    pw = proposal.bit_count()
                    if not proposal:
                        continue
                    best = min(best, pw)
                    if pw <= 10:
                        short.add(proposal)
                    if pw < weight:
                        word, weight = proposal, pw
                basis[i] = word

    eight = [word for word in short if word.bit_count() <= 8]
    mask = (1 << q) - 1
    exposure = [
        sum(bool((word >> (j * q)) & mask) for word in eight)
        for j in range(len(cols) // q)
    ]
    return best, len(eight), max(exposure, default=0), len(short)


def _shape_component(rng, G, side, supports, geometry, pressure):
    """Optimize origins, preserving the slot's indexed relative geometry."""
    import math

    n = len(supports[0]) * G.q
    augmentation = [
        sum((count & 1) << j for j, count in enumerate(row))
        for row in supports
    ]
    allowed_deficit = len(supports) - gf2_rank(augmentation)
    sites = [(i, j) for i, row in enumerate(supports)
             for j, count in enumerate(row) if count]
    pool = []
    seen = set()

    def clone(grid):
        return [[list(c) for c in row] for row in grid]

    def assess(grid):
        rows = _shape_component_rows(G, side, grid)
        deficit = len(rows) - gf2_rank(rows)
        # Screen the actual matrix, not just the character prediction.
        if deficit != allowed_deficit:
            return None
        if not _connected_checks(rows, [], n):
            return None
        cols = _shape_columns(rows, n)
        weak = [column for column in cols if column.bit_count() == 2]
        if side == "B" and len(weak) == 2 * G.q:
            # Two mixed-row matchings must form one length-34 incidence
            # cycle, not seventeen short cycles. This screens only that
            # weak subgraph; interactions with strong columns remain tested.
            if gf2_rank(weak) != len(weak) - 1:
                return None
        short = _shape_short_counts(cols)
        cycles = _four_cycles(rows)
        return (short, deficit, cycles), cols

    def offer(base, grid, cols):
        signature = tuple(tuple(sorted(G.idx(g) for g in c))
                          for row in grid for c in row)
        if signature in seen:
            return
        seen.add(signature)
        # Random tie keys retain several neutral voltage geometries;
        # they do not create slots by transforming an existing code.
        pool.append((base, rng.random(), clone(grid), list(cols)))
        pool.sort(key=lambda item: item[:2])
        del pool[8:]

    for trial in range(64):
        grid = _memory_grid(rng, G, supports, geometry)
        result = assess(grid)
        if result is not None:
            base, cols = result
            offer(base, grid, cols)

    if not pool:
        raise ValueError("No connected rank-legal gap-contract component")

    def energy(base):
        short, deficit, cycles = base
        ambiguity = sum(v * w for v, w in zip(
            short, (100000, 10000, 1000, 400)))
        return (ambiguity + 80 * cycles) / G.q

    starts = list(pool[:4])
    for restart, (base, _, start, cols) in enumerate(starts):
        grid = clone(start)
        current = base
        steps = 260 + 30 * pressure
        for step in range(steps):
            i, j = rng.choice(sites)
            old = grid[i][j]
            available = [geometry[i][j]] if supports[i][j] == 2 else []
            grid[i][j] = _memory_cell(
                rng, G, supports[i][j], available)
            result = assess(grid)
            if result is None:
                grid[i][j] = old
                continue
            proposed, proposed_cols = result
            delta = energy(proposed) - energy(current)
            temperature = 12.0 * (0.982 ** step) + 0.12
            if delta > 0 and rng.random() >= math.exp(
                    -min(700.0, delta / temperature)):
                grid[i][j] = old
                continue
            current, cols = proposed, proposed_cols
            offer(current, grid, cols)

    finalists = []
    for base, tie, grid, cols in pool:
        boundary, five, six = _shape_higher(cols)
        distance, eight, worst, ten = _memory_witnesses(
            rng, cols, G.q, 48 + 8 * pressure)
        key = (base[0], five, six, max(0, 8 - distance),
               eight, worst, boundary, base[2], ten,
               -min(distance, 10))
        finalists.append((key, grid))
    finalists.sort(key=lambda item: item[0])
    return finalists[:6]


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


def _temporal_pressure(check_rows, stabilizer_rows, n, idx, basis):
    """Two-interval CSS stress benchmark, NOT the measured ASC experiment.

    Equations are H*e1+m=s1 and H*(e1+e2)=s2, with a perfect final
    observation. Gate exposure is attenuated; suffix order is randomized,
    not ASC scheduled. BP is not relay-BP/LSD. Unresolved samples are
    counted as burden, not asserted to be measured logical failures.
    """
    import numpy as np

    m = len(check_rows)
    width = 2 * n + m
    rows = [
        row | (1 << (2 * n + r))
        for r, row in enumerate(check_rows)
    ] + [row | (row << n) for row in check_rows]
    columns = _shape_columns(rows, width)
    edge_variables = []
    checks = [[] for _ in rows]
    variables = [[] for _ in range(width)]
    for v, column in enumerate(columns):
        while column:
            bit = column & -column
            c = bit.bit_length() - 1
            edge = len(edge_variables)
            edge_variables.append(v)
            checks[c].append(edge)
            variables[v].append(edge)
            column ^= bit

    ne = len(edge_variables)
    ev = np.asarray(edge_variables, dtype=np.int64)
    ce = np.full((len(rows), max(map(len, checks))), ne, dtype=np.int64)
    ve = np.full((width, max(map(len, variables))), ne, dtype=np.int64)
    for c, edges in enumerate(checks):
        ce[c, :len(edges)] = edges
    for v, edges in enumerate(variables):
        ve[v, :len(edges)] = edges
    real = ce != ne
    check_variables = np.asarray(
        edge_variables + [width], dtype=np.int64)[ce]

    shots = 96
    intensity = np.repeat(np.asarray((0.30, 0.50)), shots // 2)
    srng = np.random.default_rng(170622731 + 1009 * idx + 7919 * basis)
    degree = np.asarray([
        column.bit_count()
        for column in _shape_columns(check_rows + stabilizer_rows, n)
    ], dtype=float)
    single_mu = 3 * 0.002 * (2.0 / 3.0) * degree
    data_p = 0.5 * (
        1.0 - np.exp(-2.0 * intensity[:, None] * single_mu))
    errors = np.zeros((shots, width + 1), dtype=bool)
    errors[:, :n] = srng.random((shots, n)) < data_p
    errors[:, n:2*n] = srng.random((shots, n)) < data_p

    hook_load = np.zeros(n, dtype=float)
    for row in stabilizer_rows:
        sites = []
        bits = row
        while bits:
            bit = bits & -bits
            sites.append(bit.bit_length() - 1)
            bits ^= bit
        if len(sites) < 2:
            continue
        hook_load[sites] += len(sites) + 1
        order = np.asarray(sites, dtype=np.int64)
        srng.shuffle(order)
        event_p = intensity * 3 * 0.002 * (len(sites) + 1) / 3.0
        for interval in range(2):
            events = np.flatnonzero(srng.random(shots) < event_p)
            for shot in events:
                cut = int(srng.integers(1, len(sites)))
                errors[shot, interval * n + order[cut:]] ^= True

    measurement_mu = 3 * 0.002 * (2.0 / 3.0) * np.asarray([
        row.bit_count() + 2 for row in check_rows
    ], dtype=float)
    measurement_p = 0.5 * (
        1.0 - np.exp(-2.0 * intensity[:, None] * measurement_mu))
    errors[:, 2*n:width] = srng.random((shots, m)) < measurement_p

    marginal_mu = single_mu + 3 * 0.002 * hook_load / 6.0
    marginal_p = 0.5 * (
        1.0 - np.exp(-2.0 * intensity[:, None] * marginal_mu))
    probabilities = np.clip(np.concatenate(
        (marginal_p, marginal_p, measurement_p), axis=1), 0.001, 0.18)
    prior = np.log((1.0 - probabilities) / probabilities)
    syndrome = np.logical_xor.reduce(errors[:, check_variables], axis=2)
    signs = 1.0 - 2.0 * syndrome.astype(float)
    v2c = np.empty((shots, ne + 1), dtype=float)
    v2c[:, :ne] = prior[:, ev]
    v2c[:, ne] = 30.0
    c2v = np.zeros_like(v2c)
    ones = np.ones((shots, len(rows), 1), dtype=float)
    active = np.ones(shots, dtype=bool)
    span = _row_span(stabilizer_rows)
    wrong = work = 0
    early = None

    for iteration in range(40):
        work += int(active.sum())
        t = np.tanh(np.clip(v2c[:, ce], -18.0, 18.0) / 2.0)
        t[:, ~real] = 1.0
        left = np.concatenate(
            (ones, np.cumprod(t, axis=2)[:, :, :-1]), axis=2)
        suffix = np.cumprod(t[:, :, ::-1], axis=2)[:, :, ::-1]
        right = np.concatenate((suffix[:, :, 1:], ones), axis=2)
        outgoing = 2.0 * np.arctanh(np.clip(
            signs[:, :, None] * left * right, -0.999999, 0.999999))
        c2v[:, ce[real]] = outgoing[:, real]
        posterior = prior + c2v[:, ve].sum(axis=2)
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
            wrong += int(not _in_row_span(word, span))
        active[settled] = False
        if iteration == 11:
            early = int(active.sum())
        if not active.any():
            if early is None:
                early = 0
            break
        damping = 0.25 if iteration < 12 else 0.50
        proposal = posterior[:, ev] - c2v[:, :ne]
        v2c[:, :ne] = (
            damping * v2c[:, :ne] + (1.0 - damping) * proposal)

    remaining = int(active.sum())
    if early is None:
        early = remaining
    return ((wrong + remaining) / shots, wrong / shots,
            early / shots, work / (40.0 * shots))


def NEW_CTX(rng, G, idx):
    """Build one indexed 2x5/2x6 contract and compare complete CSS pairings."""
    sa, sb = _design_supports(idx)
    q = G.q
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    assert q == 17
    assert (ra, ca, rb, cb) == (2, 5, 2, 6)
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert all(1 <= t <= 2 for grid in (sa, sb) for row in grid for t in row)
    for grid in (sa, sb):
        augmentation = [
            sum((t & 1) << j for j, t in enumerate(row))
            for row in grid
        ]
        assert gf2_rank(augmentation) == len(grid)

    geometry, contract = _gap_contract(G, idx, sa, sb)
    design = idx % 20
    pressure = (idx // 20) % 3
    expected_cnots = 6 * q * (
        (cb + rb) * sum(map(sum, sa))
        + (ca + ra) * sum(map(sum, sb)))
    assert expected_cnots in (24276, 25092, 24378, 24480)

    for attempt in range(3):
        apool = _shape_component(
            rng, G, "A", sa, geometry["A"], pressure)
        bpool = _shape_component(
            rng, G, "B", sb, geometry["B"], pressure)
        options = []
        for ia, (ka, A) in enumerate(apool):
            for ib, (kb, B) in enumerate(bpool):
                hx, hz, n = lifted_matrices(G, A, B)
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if n != 578 or k != 204:
                    continue
                if not _connected_checks(hx, hz, n):
                    continue
                cx, cz = _four_cycles(hx), _four_cycles(hz)
                # Exact short-component evidence leads the shortlist.
                # Sampled witnesses and cycle burden remain surrogates.
                guard = (
                    max(ka[0], kb[0]),
                    int(ka[1] + kb[1] > 0),
                    int(ka[2] + kb[2] > 0),
                )
                key = guard + (
                    ka[3] + kb[3],
                    max(ka[4], kb[4]), ka[4] + kb[4],
                    max(ka[5], kb[5]),
                    max(ka[6], kb[6]), ka[6] + kb[6],
                    max(cx / len(hx), cz / len(hz)), cx + cz,
                    ka[8] + kb[8], ia, ib,
                )
                options.append((key, A, B, hx, hz, n, k))
        options.sort(key=lambda item: item[0])

        winner = None
        screened = 0
        for entry in options[:12]:
            coarse, A, B, hx, hz, n, k = entry
            if _has_short_logical(hx, hz, n):
                continue
            sx = _temporal_pressure(hx, hz, n, idx, 0)
            sz = _temporal_pressure(hz, hx, n, idx, 1)
            # Neither basis gets an assumed causal preference. Protect
            # exact short-word guards, then reduce worse-basis burden.
            quality = coarse[:3] + (
                max(sx[0], sz[0]), sx[0] + sz[0],
                max(sx[1], sz[1]), max(sx[2], sz[2]),
                sx[3] + sz[3],
            ) + coarse[3:]
            if winner is None or quality < winner[0]:
                winner = (quality, entry, sx, sz)
            screened += 1
            if screened == 3:
                break

        if winner is None:
            continue
        quality, entry, sx, sz = winner
        _, A, B, hx, hz, n, k = entry
        assert n == 578 and k == 204
        assert 13 * n <= 40 * k < 15 * n
        assert max(row.bit_count() for row in hx) <= 11
        assert max(row.bit_count() for row in hz) <= 12
        assert max(row.bit_count() for row in hx + hz) <= W_MAX
        assert 6 * sum(row.bit_count() for row in hx + hz) == expected_cnots
        assert css_ok(hx, hz)

        # Final helper projections verify rank and every prescribed gap.
        for side, grid, counts in (("A", A, sa), ("B", B, sb)):
            rows = _shape_component_rows(G, side, grid)
            assert gf2_rank(rows) == len(counts) * q
            assert all(len(cell) == counts[i][j]
                       and len({G._key(g) for g in cell}) == len(cell)
                       and (len(cell) != 2 or
                            _memory_gap(cell, q) == geometry[side][i][j])
                       for i, row in enumerate(grid)
                       for j, cell in enumerate(row))
        bcols = _shape_columns(_shape_component_rows(G, "B", B), cb * q)
        weak = [column for column in bcols if column.bit_count() == 2]
        if weak:
            assert gf2_rank(weak) == len(weak) - int(len(weak) == 2 * q)
        return {
            "cells": {"A": A, "B": B}, "design": design,
            "pressure": pressure, "geometry": geometry,
            "contract": contract, "k": k, "screen": quality,
            "temporal_pressure": (sx, sz), "cnots": expected_cnots,
        }

    raise ValueError("No fresh indexed C17 shape contract passed CSS screening")


def CELL(rng, G, side, i, j, ctx):
    """Emit the selected complete cell without late mutation or redraws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: d2s=6 does not establish balanced circuit protection: parent slot 56 had 107/1250 failed Z shots versus 53/200000 failed X shots, with worse-basis s=3.973.
# REFUTED: Larger screened distance necessarily raises this bed's measured score: zero-failure slot 58 had d2s=22,k=45,s=7.165, below zero-failure slot 20 with d2s=20,k=58,s=7.276; the sample ceilings, not distance, limited both.
# REFUTED: Even-quotient doubletons at weight ten uniformly retain zero-failure performance: current slot 25 had four failed Z shots out of 200000, flipping 47 logical qubits, despite d2s=15 and k=66.
# REFUTED: Larger d2s guarantees better circuit reliability on the C7:C4 2x4 bed: slot17 had d2s=14 and 29/200000 failed X shots, s=6.509, versus slot28 with d2s=13 and seven failures in each basis, s=6.969; both had k=112.
# REFUTED: The C7:C4 1x3/2x5 degree-three trinomial-order construction beats the 0.350 incumbent through reduced extraction load: measured fitness was 4.1794, all five measured codes reached the X failure cap in 1750-2500 shots, and all retained weight-six witnesses despite only 20664 CNOTs.
# REFUTED: Twenty count cards with complementary three-gap palettes reliably supply twenty inequivalent designs: the current C13 parent retained only eleven distinct codes across sixty slots, with extensive Tanner-isomorphic duplication.
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
