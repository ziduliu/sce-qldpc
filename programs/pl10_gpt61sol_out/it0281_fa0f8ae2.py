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
CONCEPT = ("For cyclic C28, order 28, shapes (2,4)x(2,4), full-rank degree-three components selected by worse-basis logical BP failures should improve measured circuit fitness above 7.2133 while preserving check weight nine. This is a falsifiable decoder-ambiguity hypothesis, not distance as fitness.")
GROUP = ("Jump from C27 to genuinely different cyclic C28, an unlisted bed in BEDS. Related C29 reached 7.1850 and the C27 parent reached 7.2133 on these shapes; neither establishes a C28 advantage. C28=C7xC4 has a characteristic-two nilpotent C4 factor absent from the odd-order parent, motivating verified rank and kernel screening. Order 28 gives n=560.")
SHAPES = ("Keep equal (2,4) components, each with 112 variables and 56 checks. At q=28, n=560 and kb=112 guarantee rate 0.200; require exact verified k=112. Parent maps implicated multiple Acol/Bcol and Acol/Brow blocks, including Bcol2, but only zero to five failures per measured basis support those maps. There is insufficient evidence for a unique hotspot repair.")
FILLING = ("Twenty briefs combine three full-augmentation-rank balanced support pairs, a separate cross-row-overlap architecture and five order-seven/order-28 doubleton mixtures. Extra harness batches extend the book and use order-fourteen charged differences. Rows have six terms and columns three, so both CSS check weights remain nine and extraction uses 24192 CNOTs. Canonical relative books exclude obvious orbit copies, not arbitrary Tanner isomorphisms. Verified components screen cycles, short kernels, boundaries and checked cyclic minor witnesses through weight twelve. Several exact-k complete codes receive common-stream, degree-weighted BP probes in both bases; settled residuals are classified modulo actual stabilizers. These probes omit ASC ordering, measurement histories and relay/LSD and are not measured pL. Parent k=110 finalists had four/five Z failures versus one in regular slot3, while failed shots flipped many logicals collectively; avoid automatic rank-deficit promotion and test decoder-aware selection without claiming causation.")
FAMILY_NAME = "r020_C28_24x24_fullrank9_minor_jointbp_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 28

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("This cyclic family requires q=28")
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

A_SHAPE = (2, 4)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb)


def _design_supports(idx):
    """Three support pairs plus a distinct cross-row-overlap contract."""
    profiles = ((1, 1, 2, 2), (0, 1, 2, 3))
    pairs = ((0, 0), (0, 1), (1, 1), (0, 0))
    a, b = pairs[(idx // 5) % len(pairs)]

    def support(p):
        first = profiles[p]
        grid = [list(first), [3 - w for w in first]]
        augmentation = [
            sum((w & 1) << j for j, w in enumerate(row))
            for row in grid
        ]
        assert gf2_rank(augmentation) == 2
        return grid

    return support(a), support(b)


def _lowload_geometry(G, idx, sa, sb):
    """Order and overlap briefs, canonicalized against obvious orbit copies."""
    from math import gcd

    q = G.q
    assert q == 28
    architecture = (idx // 5) % 4
    wanted = 5 * (idx // 20) + idx % 5
    cards = ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2))
    units = [d for d in range(1, q // 2) if gcd(d, q) == 1]
    seven = [d for d in range(1, q // 2) if q // gcd(d, q) == 7]
    fourteen = [d for d in range(1, q // 2)
                if q // gcd(d, q) == 14]
    multipliers = [u for u in range(1, q) if gcd(u, q) == 1]
    grng = random.Random(280079 + 1031 * architecture)

    def normalized(cell, u):
        if not cell:
            return ()
        values = [u * a % q for a in cell]
        return min(tuple(sorted((a - origin) % q for a in values))
                   for origin in values)

    def signature(grids):
        images = []
        for u in multipliers:
            sides = []
            for grid in grids:
                columns = [(normalized(grid[0][j], u),
                            normalized(grid[1][j], u)) for j in range(4)]
                normal = tuple(sorted(columns))
                swapped = tuple(sorted((b, a) for a, b in columns))
                sides.append(min(normal, swapped))
            images.append(tuple(sorted(sides)))
        return min(images)

    book, seen = [], set()
    for attempt in range(100000):
        card = len(book) % 5
        batch = len(book) // 5
        charged_palette = seven if batch % 2 == 0 else fourteen
        grids = []
        legal = True
        for side, counts in enumerate((sa, sb)):
            sites = [(i, j) for i in range(2) for j in range(4)
                     if counts[i][j] == 2]
            charged = set(grng.sample(sites, cards[card][side]))
            grid, row_directions = [], []
            for i, row_counts in enumerate(counts):
                row = []
                used_differences = set()
                directions = set()
                for j, weight in enumerate(row_counts):
                    if weight == 0:
                        cell = []
                    elif weight == 1:
                        cell = [0]
                    elif weight == 2:
                        palette = charged_palette if (i, j) in charged else units
                        cell = [0, grng.choice(palette)]
                    else:
                        cell = [0] + grng.sample(range(1, q), weight - 1)

                    assert len(set(cell)) == weight
                    for a in cell:
                        for b in cell:
                            if a == b:
                                continue
                            difference = (a - b) % q
                            # Ratios of order two/four introduce very short
                            # local cycles; reserve the longer directions.
                            if q // gcd(difference, q) < 7:
                                legal = False
                            if difference in used_differences:
                                legal = False
                            used_differences.add(difference)
                            directions.add(min(difference, q - difference))
                    row.append(cell)
                grid.append(row)
                row_directions.append(directions)

            overlap = len(row_directions[0] & row_directions[1])
            if architecture == 0 and overlap != 0:
                legal = False
            if architecture == 3 and overlap != 1:
                legal = False
            grids.append(grid)

        if not legal:
            continue
        key = signature(grids)
        if key in seen:
            continue
        seen.add(key)
        book.append(grids)
        if len(book) > wanted:
            return book[wanted]
    raise ValueError("C28 order/overlap book exhausted before indexed design")


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


def _shape_kernel_estimate(rng, cols, trials, q):
    """Sample actual kernel witnesses and their block concentration.

    The minimum is an upper bound. Witness counts are sampled exposure
    statistics, not an exact spectrum or a circuit-failure estimate.
    """
    order = list(range(len(cols)))
    best = len(cols) + 1
    short = set()

    def offer(word):
        nonlocal best
        if not word:
            return
        weight = word.bit_count()
        best = min(best, weight)
        if weight <= 12:
            short.add(word)

    # Construct cyclic minor witnesses from the VERIFIED column syndromes.
    # The group ring need not be semisimple: the determinant identity
    # still holds over this commutative characteristic-two ring.
    from itertools import combinations
    mask = (1 << q) - 1
    blocks = len(cols) // q
    nr_blocks = max(c.bit_length() for c in cols) // q
    assert nr_blocks == 2
    coefficients = [
        [(cols[j * q] >> (i * q)) & mask for j in range(blocks)]
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
    for a, b, c in combinations(range(blocks), 3):
        parts = [0] * blocks
        for destination, (u, v) in zip(
                (a, b, c), ((b, c), (a, c), (a, b))):
            parts[destination] = (
                convolution(coefficients[0][u], coefficients[1][v])
                ^ convolution(coefficients[0][v], coefficients[1][u]))
        for shift in range(q):
            word = 0
            for j, part in enumerate(parts):
                rotated = ((part << shift) | (part >> (q - shift))) & mask
                word |= rotated << (j * q)
            if not word:
                continue
            syndrome = 0
            occupied = word
            while occupied:
                bit = occupied & -occupied
                syndrome ^= cols[bit.bit_length() - 1]
                occupied ^= bit
            assert syndrome == 0, "cyclic minor failed verified kernel check"
            generators.add(word)

    ordered_generators = sorted(generators)
    for i, word in enumerate(ordered_generators):
        offer(word)
        for other in ordered_generators[:i]:
            offer(word ^ other)

    for _ in range(trials):
        rng.shuffle(order)
        pivots, basis = {}, []
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
                offer(word)

        # Elementary basis operations preserve verified kernel membership.
        for sweep in range(2):
            rng.shuffle(basis)
            for i, word in enumerate(basis):
                weight = word.bit_count()
                for j in range(i):
                    proposal = word ^ basis[j]
                    offer(proposal)
                    pw = proposal.bit_count()
                    if proposal and pw < weight:
                        word, weight = proposal, pw
                basis[i] = word
        if best <= 6:
            break

    # Count translation orbits, not resampled copies of the same witness.
    # These permutations classify witnesses only; submitted cells are
    # never replaced by translated copies of another code.
    vulnerable = set()
    for word in short:
        parts = [(word >> (j * q)) & mask for j in range(blocks)]
        representative = None
        for shift in range(q):
            image = 0
            for j, part in enumerate(parts):
                rotated = ((part << shift) | (part >> (q - shift))) & mask
                image |= rotated << (j * q)
            if representative is None or image < representative:
                representative = image
        vulnerable.add(representative)
    loads = [
        sum(((word >> (j * q)) & mask).bit_count()
            for word in vulnerable)
        for j in range(blocks)
    ]
    return best, len(vulnerable), max(loads, default=0)


def _lowload_cycles(rows, cols):
    """Exact Tanner four/six-cycle counts from verified component rows."""
    from itertools import combinations

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

    four = sum(v * (v - 1) // 2
               for a, neighbors in enumerate(adj)
               for b, v in neighbors.items() if a < b)
    six = 0
    for a, neighbors in enumerate(adj):
        for b, ab in neighbors.items():
            if b <= a:
                continue
            for c, bc in adj[b].items():
                if c > b and c in neighbors:
                    six += ab * bc * neighbors[c]
    # Correct triangles that reuse a variable for two or three edges.
    for ns in neighborhoods:
        for a, b, c in combinations(ns, 3):
            six -= adj[a][b] + adj[a][c] + adj[b][c] - 2
    return four, six


def _shape_component(rng, G, side, supports, geometry, mode):
    """Optimize offsets without changing the slot's relative-support brief."""
    n = len(supports[0]) * G.q

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def draw():
        result = []
        for row in geometry:
            out = []
            for relative in row:
                origin = rng.randrange(G.q)
                out.append([((origin + a) % G.q,) for a in relative])
            result.append(out)
        return result

    def assess(grid):
        rows = _shape_component_rows(G, side, grid)
        deficit = len(rows) - gf2_rank(rows)
        if deficit > 2 or not _connected_checks(rows, [], n):
            return None
        cols = _shape_columns(rows, n)
        assert all(col.bit_count() == 3 for col in cols)
        short = _shape_short_counts(cols)
        cycles = _lowload_cycles(rows, cols)
        return (short, deficit, cycles), cols

    pool = []
    for trial in range(128):
        grid = draw()
        result = assess(grid)
        if result is None:
            continue
        base, cols = result
        pool.append((base, trial, grid, cols))
        pool.sort(key=lambda item: (item[0], item[1]))
        del pool[6:]

    sites = [(i, j) for i in range(2) for j in range(4)
             if supports[i][j]]
    finalists = []
    for base, trial, grid, cols in pool:
        # Cell translations alter inter-cell holonomy while preserving
        # every indexed internal difference and every term count.
        for step in range(72):
            proposal = clone(grid)
            i, j = rng.choice(sites)
            shift = rng.randrange(1, G.q)
            proposal[i][j] = [((g[0] + shift) % G.q,)
                              for g in proposal[i][j]]
            result = assess(proposal)
            if result is not None and result[0] <= base:
                base, cols = result
                grid = proposal

        boundary, five, six = _shape_higher(cols)
        estimate, short_load, peak = _shape_kernel_estimate(
            rng, cols, 48, G.q)
        # Above a coarse sampled protection floor, prioritize collective
        # ambiguity and boundaries; do not reward larger upper bounds.
        exposure = (peak, short_load) if mode % 2 else (short_load, peak)
        key = (base[0], int(estimate < 12), five, six,
               exposure, boundary, base[2], base[1], trial)
        finalists.append((key, grid))

    if not finalists:
        raise ValueError("No connected degree-three component passed screening")
    finalists.sort(key=lambda item: item[0])
    # Preserve alternatives for joint, actual-code decoder screening.
    # Independent component minima need not give the best memory circuit.
    return finalists[:4]


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


def _c28_bp_probe(checks, stabilizers, n, seed):
    """Complete-code BP ambiguity surrogate, NOT an ASC memory experiment.

    Common streams compare finalists within one slot. Data rates depend
    on verified incidence degrees. Possible opposite-check pairs are
    not certified ASC hooks. Measurement histories and relay/LSD are
    omitted; settled residuals are classified modulo actual stabilizers.
    """
    import numpy as np

    def sites(row):
        result = []
        while row:
            bit = row & -row
            result.append(bit.bit_length() - 1)
            row ^= bit
        return result

    neighborhoods = [sites(row) for row in checks]
    opposite = [sites(row) for row in stabilizers]
    nr, width = len(neighborhoods), max(map(len, neighborhoods))
    incidence = np.full((nr, width), n, dtype=np.int32)
    valid = np.zeros((nr, width), dtype=bool)
    for r, row in enumerate(neighborhoods):
        incidence[r, :len(row)] = row
        valid[r, :len(row)] = True
    flat = incidence.ravel()
    edges = len(flat)
    degrees = np.bincount(flat[flat < n], minlength=n)
    other_degrees = np.asarray(
        [col.bit_count() for col in _shape_columns(stabilizers, n)],
        dtype=np.int32)
    assert np.all(degrees > 0) and np.all(other_degrees > 0)

    variable_edges = np.full(
        (n, int(degrees.max())), edges, dtype=np.int32)
    used = np.zeros(n, dtype=np.int32)
    for edge, c in enumerate(flat):
        if c < n:
            variable_edges[c, used[c]] = edge
            used[c] += 1

    shots = 96
    channel = np.random.default_rng(seed)
    levels = np.asarray((0.012, 0.020, 0.030))
    rates = np.minimum(
        0.24, levels[np.arange(shots) % 3, None]
        * (degrees + other_degrees)[None, :] / 6.0)
    errors = np.zeros((shots, n + 1), dtype=bool)
    errors[:, :n] = channel.random((shots, n)) < rates
    for shot in range(shots):
        if channel.random() < 0.35:
            row = opposite[int(channel.integers(len(opposite)))]
            endpoints = channel.choice(len(row), 2, replace=False)
            for endpoint in endpoints:
                errors[shot, row[int(endpoint)]] ^= True

    prior_rates = np.minimum(
        0.28, rates + 0.7 * other_degrees[None, :]
        / int(other_degrees.sum()))
    prior = np.zeros((shots, n + 1), dtype=np.float64)
    prior[:, :n] = np.log((1.0 - prior_rates) / prior_rates)
    target = (errors[:, incidence].sum(axis=2) & 1).astype(bool)
    target_sign = np.where(target, -1.0, 1.0)
    messages = prior[:, flat].reshape(shots, nr, width).copy()
    messages = np.where(valid[None, :, :], messages, 0.0)

    done = np.zeros(shots, dtype=bool)
    decoded = np.zeros((shots, n), dtype=bool)
    completion = np.full(shots, 41, dtype=np.int32)
    for iteration in range(1, 41):
        signs = np.where(messages < 0, -1.0, 1.0)
        logarithms = np.log(np.clip(
            np.tanh(np.abs(messages) / 2.0), 1e-12, 1.0 - 1e-7))
        logarithms = np.where(valid[None, :, :], logarithms, 0.0)
        products = np.minimum(np.exp(
            logarithms.sum(axis=2, keepdims=True) - logarithms),
            1.0 - 1e-7)
        magnitude = np.log1p(products) - np.log1p(-products)
        outgoing = (
            magnitude * signs * signs.prod(axis=2, keepdims=True)
            * target_sign[:, :, None])
        outgoing = np.where(valid[None, :, :], outgoing, 0.0)
        outgoing_flat = outgoing.reshape(shots, edges)
        padded = np.zeros((shots, edges + 1), dtype=np.float64)
        padded[:, :edges] = outgoing_flat

        posterior = np.zeros((shots, n + 1), dtype=np.float64)
        posterior[:, :n] = (
            prior[:, :n] + padded[:, variable_edges].sum(axis=2))
        hard = posterior < 0
        syndrome = (hard[:, incidence].sum(axis=2) & 1).astype(bool)
        settled = np.all(syndrome == target, axis=1)
        newly = settled & ~done
        decoded[newly] = hard[newly, :n]
        completion[newly] = iteration
        done |= settled
        if np.all(done):
            break
        extrinsic = (
            posterior[:, flat] - outgoing_flat).reshape(shots, nr, width)
        messages = np.clip(
            0.35 * messages + 0.65 * extrinsic, -24.0, 24.0)
        messages = np.where(valid[None, :, :], messages, 0.0)

    span = _row_span(stabilizers)
    failed = ~done
    for shot in np.flatnonzero(done):
        word = 0
        for c in np.flatnonzero(errors[shot, :n] ^ decoded[shot]):
            word |= 1 << int(c)
        if not _in_row_span(word, span):
            failed[shot] = True
    return (int(np.count_nonzero(failed)),
            int(np.count_nonzero(~done)), int(completion.sum()))


def NEW_CTX(rng, G, idx):
    """Build an indexed C28 brief, then select among exact-rate CSS finalists."""
    assert G.q == 28 and A_SHAPE == B_SHAPE == (2, 4)
    sa, sb = _design_supports(idx)
    assert _no_dead_lines(sa, sb)
    # Both budgets independently equal 6+3=9.
    assert _grid_weights(sa, sb) == 9
    assert 9 <= W_MAX
    for supports in (sa, sb):
        assert all(sum(row) == 6 for row in supports)
        assert all(sum(column) == 3 for column in zip(*supports))
        assert all(0 <= count <= CELL_MAX
                   for row in supports for count in row)

    geometry = _lowload_geometry(G, idx, sa, sb)
    design, mode = idx % 20, idx // 20
    for attempt in range(4):
        apool = _shape_component(rng, G, "A", sa, geometry[0], mode)
        bpool = _shape_component(rng, G, "B", sb, geometry[1], mode)
        options = []
        for screen_a, A in apool:
            for screen_b, B in bpool:
                prefix = (
                    max(screen_a[0], screen_b[0]),
                    int(bool(screen_a[1] or screen_b[1])),
                    screen_a[2] + screen_b[2],
                    screen_a[3] + screen_b[3])
                key = prefix + (
                    max(screen_a[4], screen_b[4]),
                    screen_a[5] + screen_b[5],
                    screen_a[6][0] + screen_b[6][0],
                    screen_a[6][1] + screen_b[6][1],
                    screen_a[7] + screen_b[7],
                    screen_a[8], screen_b[8])
                options.append((key, A, B, screen_a, screen_b))
        options.sort(key=lambda item: item[0])

        verified = []
        seen_fillings = set()
        for key, A, B, screen_a, screen_b in options:
            if verified and key[:4] != verified[0][0][:4]:
                break
            signature = tuple(
                tuple(sorted(G.idx(g) for g in cell))
                for grid in (A, B) for row in grid for cell in row)
            if signature in seen_fillings:
                continue
            seen_fillings.add(signature)
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if n != 560 or k != 112:
                continue
            if not _connected_checks(hx, hz, n):
                continue
            if _has_short_logical(hx, hz, n):
                continue
            assert max(row.bit_count() for row in hx + hz) == 9
            assert 6 * sum(row.bit_count() for row in hx + hz) == 24192
            assert css_ok(hx, hz)
            verified.append(
                (key, A, B, hx, hz, screen_a, screen_b))
            if len(verified) == 6:
                break
        if not verified:
            continue

        winner = None
        probe_seed = 28017003 + 1009 * idx
        for key, A, B, hx, hz, screen_a, screen_b in verified:
            x = _c28_bp_probe(hz, hx, 560, probe_seed)
            z = _c28_bp_probe(hx, hz, 560, probe_seed + 1)
            decoder_key = (
                max(x[0], z[0]), x[0] + z[0],
                max(x[1], z[1]), x[1] + z[1],
                x[2] + z[2])
            quality = key[:4] + decoder_key + key[4:]
            if winner is None or quality < winner[0]:
                winner = (
                    quality, A, B, screen_a, screen_b, x, z)

        quality, A, B, screen_a, screen_b, x, z = winner
        return {
            "cells": {"A": A, "B": B}, "design": design,
            "mode": mode, "k": 112, "relative_geometry": geometry,
            "screen": quality, "component_screen": (screen_a, screen_b),
            "bp_surrogate": (x, z),
        }

    raise ValueError("No connected C28 filling passed exact-k=112 CSS screening")


def CELL(rng, G, side, i, j, ctx):
    """Emit the complete screened 2x4 cell without late randomization."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: d2s=6 does not establish balanced circuit protection: parent slot 56 had 107/1250 failed Z shots versus 53/200000 failed X shots, with worse-basis s=3.973.
# REFUTED: Larger screened distance necessarily raises this bed's measured score: zero-failure slot 58 had d2s=22,k=45,s=7.165, below zero-failure slot 20 with d2s=20,k=58,s=7.276; the sample ceilings, not distance, limited both.
# REFUTED: Even-quotient doubletons at weight ten uniformly retain zero-failure performance: current slot 25 had four failed Z shots out of 200000, flipping 47 logical qubits, despite d2s=15 and k=66.
# REFUTED: Raising the screened distance within the degree-three/four parent necessarily improves circuit reliability: slot17 had d2s=14,s=6.509 and 29 X failures, while slot28 had d2s=13,s=6.969 and seven failures in each basis, all at 200000 shots.
# REFUTED: Distinct canonical internal relative-support keys suffice for Tanner-distinct submissions: the measured C27 program retained only 24 distinct codes from 60 slots, including slot2 marked iso_dup_of_1.
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
