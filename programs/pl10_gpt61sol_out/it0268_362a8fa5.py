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
CONCEPT = ("Cyclic C17, order 17, shapes (2,5)x(2,6): four-cycle-compatible degree-four A reinforcement, financed by a smaller B term budget and selected by joint decoder burden, should exceed the parent's worse-basis circuit fitness 5.0830.")
GROUP = ("Keep the additive cyclic construction rule; change only its order from 13 to 17 to accompany the new shape. Size factor 34 permits q=17 and n=578. The existing C17 (2,5)x(2,6) bed scored about 4.8, so this is not a group-only advantage claim. Eight undirected differences and seventeen cross-row voltages permit denser A packing than C13; verified matrix ranks remain authoritative.")
SHAPES = ("Target cell 0.350 with (2,5)x(2,6): n=578, kb=204 and guaranteed rate 12/34=0.352941. A has 85 variables/34 checks and B has 102 variables/34 checks; require complete verified k=204. Dense 18-term A has sixteen cross-row term pairs, below seventeen available voltages, whereas comparable reinforcement exceeded thirteen. Parent failures reached every logical qubit with mostly diffuse 1.1-1.3x Acol/Bcol and Acol/Brow enrichment; slot32 X failed 103/8750 versus Z 100/28250, motivating global rather than localized redesign.")
FILLING = ("Twenty contracts cross four A budgets with five B architectures. A has 15-18 terms; the 15-term contract uses 18-term B partners, while reinforced A uses 16-term B partners with row totals eight. Explicit HX/HZ budgets are <=12 and six-round CNOT counts are 25092,24480,25296,26112. Indexed canonical internal-ratio cards prescribe disjoint side palettes and row-internal Sidon differences; extra harness batches receive further cards, not transformed completed codes. Verified singleton actions accelerate component screening for rank, short kernels, cycles and boundaries. Complete finalists then receive common-random-number BP probes with residuals classified modulo actual stabilizers and sampled opposite-check pairs. The probe omits ASC ordering, measurement histories, relay decoding and LSD; measured circuit failure, not this surrogate or d2s, decides success.")
FAMILY_NAME = "r035_C17_25x26_gateexchange_jointbp_v2"

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
    """Four gate allocations, each with five different B architectures."""
    architecture, b_model = divmod(idx % 20, 5)
    a_profiles = (
        ((2, 2, 1, 1, 1), (1, 1, 2, 2, 2)),
        ((2, 2, 1, 1, 2), (1, 1, 2, 2, 2)),
        ((2, 2, 2, 1, 1), (2, 2, 1, 2, 2)),
        ((2, 2, 2, 2, 1), (2, 2, 2, 1, 2)),
    )
    if architecture == 0:
        # Eighteen terms, row totals nine. The last two cards trade
        # degree-three columns for different degree-two/four profiles.
        columns = (
            ((2,1), (2,1), (2,1), (1,2), (1,2), (1,2)),
            ((3,0), (2,1), (1,2), (1,2), (1,2), (1,2)),
            ((3,0), (0,3), (2,1), (2,1), (1,2), (1,2)),
            ((2,2), (1,2), (2,1), (1,2), (2,1), (1,1)),
            ((2,2), (2,2), (2,1), (1,2), (1,1), (1,1)),
        )[b_model]
    else:
        # Sixteen terms, row totals eight. Pure binomials/trinomials
        # occur at most once per row; their interactions are screened.
        columns = (
            ((2,1), (2,1), (1,2), (1,2), (1,1), (1,1)),
            ((2,1), (1,2), (1,2), (1,2), (2,0), (1,1)),
            ((2,1), (2,1), (1,2), (1,2), (2,0), (0,2)),
            ((3,0), (0,3), (2,1), (1,2), (1,1), (1,1)),
            ((3,0), (1,2), (1,2), (1,2), (1,1), (1,1)),
        )[b_model]
    sa = [list(row) for row in a_profiles[architecture]]
    sb = [[column[i] for column in columns] for i in range(2)]
    assert len(sa) == len(sb) == 2
    assert len(sa[0]) == 5 and len(sb[0]) == 6
    assert tuple(map(sum, sb)) == ((9, 9) if architecture == 0
                                  else (8, 8))
    assert _no_dead_lines(sa, sb)
    wx = max(map(sum, sa)) + max(map(sum, columns))
    wz = max(map(sum, sb)) + max(
        sum(row[j] for row in sa) for j in range(5))
    assert max(wx, wz) == _grid_weights(sa, sb) <= min(12, W_MAX)
    return sa, sb


def _memory_ratio_card(G, sa, sb, idx):
    """Prescribe fresh internal geometry, not gauges of a completed code.

    Canonicalization removes cyclic automorphisms, independent cell
    translations, and component row/column permutations from the
    ratio-card bank. It is not a complete Tanner-isomorphism test.
    """
    from itertools import combinations, permutations

    q = G.q
    supports = {"A": sa, "B": sb}
    book_rng = random.Random(170025026 + 1009 * (idx % 20))
    batch = idx // 20
    a_size = max(sum(t * (t - 1) // 2 for t in row) for row in sa)

    def directions(values):
        return {min((a - b) % q, (b - a) % q)
                for a, b in combinations(values, 2)}

    def normalized(values, multiplier):
        values = [multiplier * v % q for v in values]
        return min(tuple(sorted((v - anchor) % q for v in values))
                   for anchor in values) if values else ()

    def canonical(templates):
        images = []
        for multiplier in range(1, q):
            sides = []
            for side in ("A", "B"):
                grid = templates[side]
                versions = []
                for order in permutations(range(2)):
                    versions.append(tuple(sorted(
                        tuple(normalized(grid[i][j], multiplier)
                              for i in order)
                        for j in range(len(grid[0])))))
                sides.append(min(versions))
            images.append(tuple(sides))
        return min(images)

    triples = [(0, a, b)
               for a, b in combinations(range(1, q), 2)
               if len(directions((0, a, b))) == 3]
    seen = set()
    for proposal in range(6000):
        pa = set(book_rng.sample(range(1, (q + 1) // 2), a_size))
        pb = set(range(1, (q + 1) // 2)) - pa
        templates = {}
        possible = True
        for side, palette in (("A", pa), ("B", pb)):
            grid = []
            for counts in supports[side]:
                row = [None] * len(counts)
                used = set()
                for j in sorted(range(len(counts)),
                                key=lambda j: -counts[j]):
                    t = counts[j]
                    if t == 3:
                        choices = [v for v in triples
                                   if directions(v) <= palette
                                   and not (directions(v) & used)]
                    elif t == 2:
                        choices = [(0, d) for d in sorted(palette - used)]
                    else:
                        choices = [(0,)] if t else [()]
                    if not choices:
                        possible = False
                        break
                    row[j] = book_rng.choice(choices)
                    used.update(directions(row[j]))
                if not possible:
                    break
                grid.append(row)
            if not possible:
                break
            # Two doubletons in one column must have different gaps;
            # otherwise their four cross-row differences repeat.
            if any(len(grid[0][j]) == len(grid[1][j]) == 2
                   and directions(grid[0][j]) == directions(grid[1][j])
                   for j in range(len(grid[0]))):
                possible = False
                break
            templates[side] = grid
        if not possible:
            continue
        signature = canonical(templates)
        if signature in seen:
            continue
        seen.add(signature)
        if len(seen) == batch + 1:
            return templates, signature
    raise ValueError("no unused order-17 internal-ratio card")


def _memory_cell(rng, G, count, template):
    """Translate one prescribed full cell without changing its ratios."""
    assert len(template) == count
    g = G.pick(rng)
    cell = [G.mul(g, (value,)) for value in template]
    assert len({G._key(value) for value in cell}) == count
    return cell


def _memory_grid(rng, G, supports, templates):
    return [[_memory_cell(rng, G, count, templates[i][j])
             for j, count in enumerate(row)]
            for i, row in enumerate(supports)]


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


def _shape_component(rng, G, side, supports, palette, pressure):
    """Optimize voltages inside a prescribed internal-ratio architecture."""
    import math

    n = len(supports[0]) * G.q
    # Cache only blocks returned by the verified helper. Linear
    # composition below is not a separately implemented lift.
    actions = {
        G.idx(g): tuple(_shape_component_rows(G, side, [[[g]]]))
        for g in G.elements
    }

    def component_rows(grid):
        rows = []
        for grid_row in grid:
            out = [0] * G.q
            for j, cell in enumerate(grid_row):
                for g in cell:
                    for r, value in enumerate(actions[G.idx(g)]):
                        out[r] ^= value << (j * G.q)
            rows.extend(out)
        return rows
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
        rows = component_rows(grid)
        deficit = len(rows) - gf2_rank(rows)
        # Screen the actual matrix, not just the character prediction.
        if deficit != allowed_deficit:
            return None
        if not _connected_checks(rows, [], n):
            return None
        cols = _shape_columns(rows, n)
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
        grid = _memory_grid(rng, G, supports, palette)
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
            grid[i][j] = _memory_cell(
                rng, G, supports[i][j], palette[i][j])
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
    for _, grid in finalists[:6]:
        assert component_rows(grid) == _shape_component_rows(G, side, grid)
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


def _memory_joint_probe(checks, stabilizers, n, seed, shots=96):
    """Stabilizer-aware BP surrogate on the complete verified code.

    Data noise and sampled shared-opposite-check pairs are included.
    ASC order, noisy measurement histories, relay decoding and LSD
    are omitted: these results are not circuit failure measurements.
    """
    import numpy as np

    columns = _shape_columns(checks, n)
    nr = len(checks)
    edge_variables = []
    check_edges = [[] for _ in range(nr)]
    variable_edges = [[] for _ in range(n)]
    for v, column in enumerate(columns):
        while column:
            bit = column & -column
            c = bit.bit_length() - 1
            edge = len(edge_variables)
            edge_variables.append(v)
            check_edges[c].append(edge)
            variable_edges[v].append(edge)
            column ^= bit
    ne = len(edge_variables)
    ev = np.asarray(edge_variables, dtype=np.int64)
    ce = np.full((nr, max(map(len, check_edges))),
                 ne, dtype=np.int64)
    ve = np.full((n, max(map(len, variable_edges))),
                 ne, dtype=np.int64)
    for c, edges in enumerate(check_edges):
        ce[c, :len(edges)] = edges
    for v, edges in enumerate(variable_edges):
        ve[v, :len(edges)] = edges
    real = ce != ne
    cv = np.asarray(edge_variables + [n], dtype=np.int64)[ce]
    degrees = np.asarray([c.bit_count() for c in columns], dtype=float)
    assert np.all(degrees > 0)

    channel = np.random.default_rng(seed)
    levels = np.asarray((0.012, 0.020, 0.030))
    rates = levels[np.arange(shots) % 3, None]
    rates = np.minimum(rates * degrees[None, :] / degrees.mean(), 0.20)
    error = np.zeros((shots, n + 1), dtype=bool)
    error[:, :n] = channel.random((shots, n)) < rates

    # Opposite checks supply possible correlated pairs, not an assertion
    # that their two positions are adjacent in the ASC schedule.
    hook_rows = []
    for row in stabilizers:
        sites = []
        while row:
            bit = row & -row
            sites.append(bit.bit_length() - 1)
            row ^= bit
        if len(sites) >= 2:
            hook_rows.append(sites)
    for shot in np.flatnonzero(channel.random(shots) < 0.25):
        sites = hook_rows[int(channel.integers(len(hook_rows)))]
        pair = channel.choice(len(sites), 2, replace=False)
        error[shot, sites[int(pair[0])]] ^= True
        error[shot, sites[int(pair[1])]] ^= True

    target = np.logical_xor.reduce(error[:, cv], axis=2)
    signs = 1.0 - 2.0 * target.astype(float)
    prior_rates = np.clip(
        rates + 0.5 * degrees[None, :] / ne, 0.002, 0.25)
    prior = np.log((1.0 - prior_rates) / prior_rates)
    v2c = np.empty((shots, ne + 1), dtype=float)
    v2c[:, :ne] = prior[:, ev]
    v2c[:, ne] = 30.0
    c2v = np.zeros_like(v2c)
    span = _row_span(stabilizers)
    active = np.ones(shots, dtype=bool)
    wrong = iteration_sum = 0
    early = None
    ones = np.ones((shots, nr, 1), dtype=float)

    for iteration in range(36):
        t = np.tanh(np.clip(v2c[:, ce], -18.0, 18.0) / 2.0)
        t[:, ~real] = 1.0
        left = np.concatenate(
            (ones, np.cumprod(t, axis=2)[:, :, :-1]), axis=2)
        suffix = np.cumprod(t[:, :, ::-1], axis=2)[:, :, ::-1]
        right = np.concatenate((suffix[:, :, 1:], ones), axis=2)
        product = signs[:, :, None] * left * right
        outgoing = 2.0 * np.arctanh(
            np.clip(product, -0.999999, 0.999999))
        c2v[:, ce[real]] = outgoing[:, real]
        posterior = prior + c2v[:, ve].sum(axis=2)
        estimate = np.zeros((shots, n + 1), dtype=bool)
        estimate[:, :n] = posterior < 0
        matched = np.logical_xor.reduce(estimate[:, cv], axis=2)
        settled = active & np.all(matched == target, axis=1)
        for shot in np.flatnonzero(settled):
            residual = 0
            for v in np.flatnonzero(error[shot, :n] ^ estimate[shot, :n]):
                residual |= 1 << int(v)
            wrong += int(not _in_row_span(residual, span))
            iteration_sum += iteration + 1
        active[settled] = False
        if iteration == 11:
            early = int(active.sum())
        if not active.any():
            if early is None:
                early = 0
            break
        damping = 0.25 if iteration < 12 else 0.50
        proposal = posterior[:, ev] - c2v[:, :ne]
        v2c[:, :ne] = np.clip(
            damping * v2c[:, :ne] + (1.0 - damping) * proposal,
            -24.0, 24.0)

    unresolved = int(active.sum())
    if early is None:
        early = unresolved
    return wrong + unresolved, unresolved, early, (
        iteration_sum + 37 * unresolved)


def NEW_CTX(rng, G, idx):
    """Optimize a gate-allocation card, then compare complete-code BP burden."""
    sa, sb = _design_supports(idx)
    assert G.q == 17
    assert A_SHAPE == (2, 5) and B_SHAPE == (2, 6)
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    for supports in (sa, sb):
        augmentation = [
            sum((t & 1) << j for j, t in enumerate(row))
            for row in supports]
        assert gf2_rank(augmentation) == 2

    design = idx % 20
    pressure = (idx // 20) % 3
    templates, ratio_signature = _memory_ratio_card(G, sa, sb, idx)
    expected_cnots = 6 * G.q * (
        8 * sum(map(sum, sa)) + 7 * sum(map(sum, sb)))
    assert expected_cnots in (25092, 24480, 25296, 26112)

    for attempt in range(3):
        apool = _shape_component(
            rng, G, "A", sa, templates["A"], pressure)
        bpool = _shape_component(
            rng, G, "B", sb, templates["B"], pressure)
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
                # Exact short ambiguities lead; longer sampled witnesses
                # and graph exposure only choose the joint-probe shortlist.
                geometry = (
                    max(ka[0], kb[0]), ka[0], kb[0],
                    max(ka[1], kb[1]), ka[1] + kb[1],
                    max(ka[2], kb[2]), ka[2] + kb[2],
                    ka[3] + kb[3],
                    max(cx / len(hx), cz / len(hz)), cx + cz,
                    max(ka[4], kb[4]), ka[4] + kb[4],
                    max(ka[6], kb[6]), ka[6] + kb[6],
                    ia, ib,
                )
                options.append((geometry, A, B, hx, hz, n, k))
        options.sort(key=lambda item: item[0])

        finalists = []
        for geometry, A, B, hx, hz, n, k in options[:8]:
            if _has_short_logical(hx, hz, n):
                continue
            assert 13 * n <= 40 * k < 15 * n
            assert max(row.bit_count() for row in hx) <= min(12, W_MAX)
            assert max(row.bit_count() for row in hz) <= min(12, W_MAX)
            assert 6 * sum(row.bit_count() for row in hx + hz) == expected_cnots
            assert css_ok(hx, hz)

            # Each basis uses the same random stream across this slot's
            # alternative pairings. Residuals are stabilizer-classified.
            seed = 170026035 + 1009 * idx
            x = _memory_joint_probe(hz, hx, n, seed)
            z = _memory_joint_probe(hx, hz, n, seed + 1)
            quality = (
                max(x[0], z[0]), x[0] + z[0],
                max(x[1], z[1]), x[1] + z[1],
                max(x[2], z[2]), x[2] + z[2],
                x[3] + z[3],
            ) + geometry
            finalists.append((quality, A, B, x, z))
            if len(finalists) >= 3:
                break

        if finalists:
            finalists.sort(key=lambda item: item[0])
            quality, A, B, x, z = finalists[0]
            return {
                "cells": {"A": A, "B": B}, "design": design,
                "pressure": pressure, "ratio_card": ratio_signature,
                "k": 204, "screen": quality,
                "bp_surrogate": (x, z), "cnots": expected_cnots,
            }

    raise ValueError("No order-17 gate-allocation code passed verified screening")


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
# REFUTED: Side-disjoint gaps plus component short-kernel screening suffice for low, balanced C13 memory: the parent scored 5.0830, all five measured codes reached failure caps, and slot32 had X 103/8750 versus Z 100/28250 failures with collective flips across all 195 logical qubits.
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
