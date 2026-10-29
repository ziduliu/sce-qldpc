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
CONCEPT = ("For the square-affine matrix subgroup of GL(2,11), order 55, shapes (1,3)x(1,3), balanced (2,3,3) components selected by exact weight-seven spectra and noisy-syndrome holdout probes should improve cell-0.400 ASC fitness above the current 4.8678; measured circuit pL decides.")
GROUP = ("Frozen L1 matrix group, order and encoding. Keep exactly the square-affine subgroup over F11, isomorphic to C11:C5. This bed measured 4.8678, while the shown lower-load program reached 5.0213. Those readings motivate a controlled filling experiment, not a group advantage. The fixed order gives n=550.")
SHAPES = ("Frozen equal (1,3) sides give 55-check, 165-variable components, n=550, kb=220 and guaranteed rate 0.400; the size factor ten permits q<=59. Require complete helper-built k=220, retaining zero k-kb. Parent weight-eight witnesses span sector-one Acol/Bcol blocks and failures affect all 220 logicals; Acol/Brow check and data enrichment is mostly 1.0-1.1x, establishing no persistent isolated repair target.")
FILLING = ("Both components use (2,3,3): an order-eleven binomial and two noncommuting trinomials, giving maximum CSS weight eleven and 21120 six-round CNOTs instead of twelve and 23760. Canonical unordered quotient contracts include positive and negative repeated-slope orientations. Fresh helper-derived searches require connectivity, rank 55, no C4 and no component kernel through six; an exact anchored weight-seven census and sampled weight-eight exposure shortlist alternatives. Rooted fingerprints collapse proposals and exclude known displayed measured components, but the evaluator remains authoritative for the complete run archive. Complete verified CSS finalists require k=220 and no logical through four. Two-interval noisy-syndrome BP probes include gate-degree exposure and opposite-check suffix faults; independent holdout results lead selection. They omit actual ASC ordering, the six-round history and relay/LSD, and are not circuit fitness. Target collective flips, basis imbalance and parent slot1's 102/4500 X failures versus 101/11750 Z despite d2s=8.")
FAMILY_NAME = "r040_squareaffine_F11_q55_13x13_exactseven_noisysyndrome_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 55

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Square-affine matrices over F11 require q=55")
    # Row-major matrices [[b,a],[0,1]], with b a nonzero square.
    # This is a subgroup of GL(2,11), acting by t -> b*t+a.
    squares = sorted({t*t % 11 for t in range(1, 11)})
    els = [(b, a, 0, 1) for b in squares for a in range(11)]
    positions = {g: i for i, g in enumerate(els)}

    def red(g):
        return tuple(int(t) % 11 for t in g)

    def mul(g, h):
        a, b, c, d = red(g)
        e, f, k, l = red(h)
        return ((a*e+b*k) % 11, (a*f+b*l) % 11,
                (c*e+d*k) % 11, (c*f+d*l) % 11)

    def inv(g):
        a, b, c, d = red(g)
        determinant = (a*d-b*c) % 11
        reciprocal = pow(determinant, 9, 11)
        return ((reciprocal*d) % 11, (-reciprocal*b) % 11,
                (-reciprocal*c) % 11, (reciprocal*a) % 11)

    def idx(g):
        return positions[red(g)]

    return dict(elements=els, ident=(1, 0, 0, 1),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


def _f57_columns(rows, n):
    """Column syndromes of a verified lifted matrix."""
    columns = [0] * n
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return columns


def _f57_briefs():
    """Twenty quotient architectures, not twenty unconstrained draws."""
    from itertools import combinations_with_replacement

    # Types 0/1: quotient multiplicities (0,0,1)/(0,0,2).
    # Types 2/3: quotient triangles (0,1,2)/(0,1,3).
    # The latter have cyclic gap patterns (1,1,3)/(1,2,2).
    # Multisets remove obvious component-column permutation copies.
    book = list(combinations_with_replacement(range(4), 3))
    assert len(book) == 20
    return book


def _f57_draw(rng, G, brief):
    """Fresh noncommuting triangles in one prescribed quotient architecture."""
    def affine(translation, exponent):
        return (pow(3, exponent % 5, 11),
                translation % 11, 0, 1)

    first_rotation = next(
        (j for j, kind in enumerate(brief) if kind < 2), None)
    cells = []
    for j, kind in enumerate(brief):
        if kind < 2:
            # Normalize common scaling and complement-translation gauges.
            displacement = (1 if j == first_rotation
                            else rng.randrange(1, 11))
            phase = 0 if j == first_rotation else rng.randrange(11)
            cell = [G.e, affine(displacement, 0),
                    affine(phase, kind + 1)]
        else:
            exponent = 2 if kind == 2 else 3
            if first_rotation is None and j == 0:
                a, b = 0, 1
            else:
                a, b = rng.randrange(11), rng.randrange(11)
            u, v = affine(a, 1), affine(b, exponent)
            # Exclude triangles confined to one order-five complement.
            while G._key(G.mul(u, v)) == G._key(G.mul(v, u)):
                b = rng.randrange(11)
                v = affine(b, exponent)
            cell = [G.e, u, v]
        assert len({G._key(g) for g in cell}) == 3
        cells.append(cell)
    return [cells]


def _f57_distinct_differences(G, grid):
    """Cheap ordered-difference prefilter; verified matrices decide legality."""
    seen = set()
    for cell in grid[0]:
        for a in cell:
            for b in cell:
                if G._key(a) == G._key(b):
                    continue
                difference = G._key(G.mul(a, G.inv(b)))
                if difference in seen:
                    return False
                seen.add(difference)
    return True


def _f57_component(G, grid, side):
    """Extract the actual A-left or B-right component using a unit partner."""
    unit = [[[G.e]]]
    if side == "A":
        rows = lifted_matrices(G, grid, unit)[0]
    else:
        rows = lifted_matrices(G, unit, grid)[1]
    width = 3 * G.q
    mask = (1 << width) - 1
    return [row & mask for row in rows]


def _f57_clean4(columns):
    """Exact exclusion of component kernels through weight four."""
    singles = set(columns)
    if 0 in singles or len(singles) != len(columns):
        return False
    seen = set()
    for i, x in enumerate(columns):
        for j in range(i):
            syndrome = x ^ columns[j]
            if syndrome == 0 or syndrome in singles or syndrome in seen:
                return False
            seen.add(syndrome)
    return True


def _f57_cycles(rows, columns):
    """Verified four-cycles and exact six-cycles when four-cycles are absent."""
    m = len(rows)
    adjacency = [0] * m
    four = 0
    for i, row in enumerate(rows):
        for j in range(i):
            common = (row & rows[j]).bit_count()
            four += common * (common - 1) // 2
            if common:
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i
    if four:
        return four, None

    triangles = 0
    for i in range(m):
        lower = adjacency[i] & ((1 << i) - 1)
        while lower:
            bit = lower & -lower
            j = bit.bit_length() - 1
            triangles += (adjacency[i] & adjacency[j]
                          & ((1 << j) - 1)).bit_count()
            lower ^= bit
    stars = sum(d * (d - 1) * (d - 2) // 6
                for d in (c.bit_count() for c in columns))
    six = triangles - stars
    assert six >= 0
    return 0, six


def _f57_six_boundary(columns, limit=None):
    """Exact weight-six kernel count and small-boundary triple exposure.

    Every column has odd weight three, excluding odd kernel weights.
    After the <=4 exclusion, distinct matching triples are disjoint.
    Every six-set has ten unordered triple partitions. No group-orbit
    shortcut is used: order-three stabilizers cannot corrupt this count.
    """
    buckets = {}
    collisions = 0
    exposure = 0
    boundary_one = 0
    size = len(columns)
    for a in range(size - 2):
        x = columns[a]
        for b in range(a + 1, size - 1):
            pair = x ^ columns[b]
            for c in range(b + 1, size):
                syndrome = pair ^ columns[c]
                previous = buckets.get(syndrome, 0)
                collisions += previous
                if limit is not None and collisions > 10 * limit:
                    return None
                buckets[syndrome] = previous + 1
                weight = syndrome.bit_count()
                if weight <= 5:
                    exposure += (7 - weight) ** 2
                    boundary_one += int(weight == 1)
    assert collisions % 10 == 0
    return collisions // 10, boundary_one, exposure


def _f55_sampled_protection(columns, seed):
    """Sample valid kernel words; estimates are upper bounds, not certificates."""
    srng = random.Random(seed)
    width = len(columns)
    permutation = list(range(width))
    minimum = width + 1
    short = set()
    for trial in range(32):
        srng.shuffle(permutation)
        pivots, basis = {}, []
        for c in permutation:
            syndrome, word = columns[c], 1 << c
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
        for i, word in enumerate(basis):
            weight = word.bit_count()
            minimum = min(minimum, weight)
            if weight <= 8:
                short.add(word)
            for other in basis[:i]:
                candidate = word ^ other
                weight = candidate.bit_count()
                minimum = min(minimum, weight)
                if weight <= 8:
                    short.add(candidate)
    return min(minimum, 10), len(short)


def _f57_select(rng, G, side, brief, batch, actions):
    """Keep independently searched alternatives for complete-code selection."""
    q = G.q
    pool, seen = [], set()
    accepted = 0

    def component(grid):
        # Exact linear composition of helper-derived singleton blocks.
        rows = [0] * q
        for j, cell in enumerate(grid[0]):
            for g in cell:
                block = actions[G.idx(g)]
                for r, word in enumerate(block):
                    rows[r] ^= word << (j*q)
        return rows

    for trial in range(2400):
        grid = _f57_draw(rng, G, brief)
        if not _f57_distinct_differences(G, grid):
            continue
        signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                          for cell in grid[0])
        if signature in seen:
            continue
        seen.add(signature)
        rows = component(grid)
        if gf2_rank(rows) != q:
            continue
        columns = _f57_columns(rows, 3*q)
        assert all(c.bit_count() == 3 for c in columns)
        four, cycles = _f57_cycles(rows, columns)
        if four or not _f57_clean4(columns):
            continue
        accepted += 1
        pool.append((cycles, signature, grid, rows, columns))
        pool.sort(key=lambda item: item[:2])
        del pool[8:]
        if accepted >= 64:
            break
    if not pool:
        raise ValueError("no eligible square-affine component for contract")

    entries = []
    probe_seed = 550030 + 29*batch + sum(
        (j+1)*kind for j, kind in enumerate(brief))
    for cycles, signature, grid, rows, columns in pool:
        limit = entries[-1][0][0] if len(entries) >= 3 else None
        census = _f57_six_boundary(columns, limit)
        if census is None:
            continue
        words6, one, exposure = census
        distance, short8 = _f55_sampled_protection(columns, probe_seed)
        tail = ((cycles, exposure) if batch % 2
                else (exposure, cycles))
        key = (words6, 10-distance, short8, one) + tail + (signature,)
        entries.append((key, grid, rows))
        entries.sort(key=lambda item: item[0])
        del entries[3:]
    if not entries:
        raise ValueError("square-affine component refinement failed")
    return entries


def _f55_probe(checks, stabilizers, n, seed, shots=96):
    """Common-random-number BP surrogate, not an ASC memory experiment.

    Includes degree-weighted data noise and sampled correlated pairs.
    Settled residuals are classified modulo the actual stabilizers.
    Measurement histories, relay decoding and ASC ordering are omitted.
    """
    import numpy as np

    neighbors = []
    for row in checks:
        current = []
        while row:
            bit = row & -row
            current.append(bit.bit_length()-1)
            row ^= bit
        neighbors.append(current)
    weight = len(neighbors[0])
    assert all(len(row) == weight for row in neighbors)
    incidence = np.asarray(neighbors, dtype=np.int32)
    flat = incidence.ravel()
    nr, edges = len(checks), len(flat)
    degrees = np.bincount(flat, minlength=n)
    assert np.all(degrees > 0)
    variable_edges = np.full(
        (n, int(degrees.max())), edges, dtype=np.int32)
    used = np.zeros(n, dtype=np.int32)
    for edge, variable in enumerate(flat):
        variable_edges[variable, used[variable]] = edge
        used[variable] += 1

    channel = np.random.default_rng(seed)
    levels = np.asarray((0.012, 0.022, 0.035))
    rates = levels[np.arange(shots) % 3, None] * degrees[None, :] / 3
    error = channel.random((shots, n)) < rates
    hook_rows = channel.integers(nr, size=shots)
    first = channel.integers(weight, size=shots)
    second = channel.integers(weight-1, size=shots)
    second += second >= first
    hooks = channel.random(shots) < 0.25
    for shot in np.flatnonzero(hooks):
        r = hook_rows[shot]
        error[shot, incidence[r, first[shot]]] ^= True
        error[shot, incidence[r, second[shot]]] ^= True

    prior_rates = np.minimum(
        rates + 0.5*degrees[None, :]/edges, 0.30)
    prior = np.log((1-prior_rates)/prior_rates)
    target = (
        error[:, flat].reshape(shots, nr, weight).sum(axis=2) & 1
    ).astype(bool)
    target_sign = np.where(target, -1.0, 1.0)
    messages = prior[:, flat].copy()

    pivots = {}
    for row in stabilizers:
        while row:
            p = row.bit_length()-1
            if p in pivots:
                row ^= pivots[p]
            else:
                pivots[p] = row
                break

    def nontrivial(word):
        while word:
            p = word.bit_length()-1
            if p not in pivots:
                return True
            word ^= pivots[p]
        return False

    failed = iteration_sum = 0
    for iteration in range(1, 33):
        count = len(error)
        shaped = messages.reshape(count, nr, weight)
        signs = np.where(shaped < 0, -1.0, 1.0)
        logarithms = np.log(np.clip(
            np.tanh(np.abs(shaped)/2), 1e-12, 1-1e-7))
        products = np.minimum(np.exp(
            logarithms.sum(axis=2, keepdims=True)-logarithms), 1-1e-7)
        magnitude = np.log1p(products)-np.log1p(-products)
        outgoing = (
            magnitude * signs * signs.prod(axis=2, keepdims=True)
            * target_sign[:, :, None]
        ).reshape(count, edges)

        padded = np.zeros((count, edges+1), dtype=np.float64)
        padded[:, :edges] = outgoing
        posterior = prior + padded[:, variable_edges].sum(axis=2)
        hard = posterior < 0
        syndrome = (
            hard[:, flat].reshape(count, nr, weight).sum(axis=2) & 1
        ).astype(bool)
        settled = np.all(syndrome == target, axis=1)
        for local in np.flatnonzero(settled):
            word = 0
            for c in np.flatnonzero(error[local] ^ hard[local]):
                word |= 1 << int(c)
            failed += int(nontrivial(word))
            iteration_sum += iteration

        keep = ~settled
        if not np.any(keep):
            return failed, 0, iteration_sum
        messages = np.clip(
            0.35*messages + 0.65*(posterior[:, flat]-outgoing),
            -24, 24)[keep]
        prior = prior[keep]
        target = target[keep]
        target_sign = target_sign[keep]
        error = error[keep]

    unresolved = len(error)
    return failed+unresolved, unresolved, iteration_sum+33*unresolved


def _f57_small_logical(checks, stabilizers, n):
    """Exact existence test for a quantum logical of weight <=4."""
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

    columns = _f57_columns(checks, n)
    seen = {0: 0}
    for i, syndrome in enumerate(columns):
        word = 1 << i
        previous = seen.get(syndrome)
        if previous is not None:
            if nontrivial(word ^ previous):
                return True
        else:
            seen[syndrome] = word
    for i, x in enumerate(columns):
        for j in range(i):
            syndrome = x ^ columns[j]
            word = (1 << i) | (1 << j)
            previous = seen.get(syndrome)
            if previous is not None:
                if nontrivial(word ^ previous):
                    return True
            else:
                seen[syndrome] = word
    return False


def NEW_CTX(rng, G, idx):
    """Exact-seven light components and noisy-syndrome holdout selection."""
    from itertools import combinations_with_replacement
    from collections import Counter
    import hashlib

    q = G.q
    assert q == 55 and A_SHAPE == B_SHAPE == (1, 3)
    counts = [[2, 3, 3]]
    assert _no_dead_lines(counts, counts)
    assert _grid_weights(counts, counts) == 11
    assert 8 + 3 <= min(12, W_MAX)
    mask, width = (1 << q)-1, 3*q

    # R1..R4 use quotient multiplicities (0,0,s), s=1..4.
    # T1/T2 use (0,1,2)/(0,1,3). Component columns and A/B
    # exchange are quotiented in the contract book.
    light = list(combinations_with_replacement(range(6), 2))
    reflection = (3, 2, 1, 0, 4, 5)

    def canonical_contract(a, b):
        direct = tuple(sorted((tuple(a), tuple(b))))
        reflected = tuple(sorted((
            tuple(sorted(reflection[t] for t in a)),
            tuple(sorted(reflection[t] for t in b)),
        )))
        # Conservatively omit a common reflected contract. This is
        # not asserted to be a group automorphism or full canonization.
        return min(direct, reflected)

    contracts = sorted({
        canonical_contract(a, b)
        for a, b in combinations_with_replacement(light, 2)
    })
    budget = max(20, N_SLOTS)
    if budget > len(contracts) or not 0 <= idx < budget:
        raise ValueError("exact-seven quotient contract book exhausted")
    ba, bb = contracts[idx*len(contracts)//budget]

    # Every block convention is extracted from lifted_matrices.
    # Wide zero-partner probes are never submitted codes.
    probe = [[[g] for g in G.elements]]
    zero = [[[]]]
    ax, _, _ = lifted_matrices(G, probe, zero)
    _, bz, _ = lifted_matrices(G, zero, probe)
    actions = {"A": {}, "B": {}}
    for t, g in enumerate(G.elements):
        actions["A"][G.idx(g)] = tuple(
            (row >> (t*q)) & mask for row in ax)
        actions["B"][G.idx(g)] = tuple(
            (row >> (t*q)) & mask for row in bz)

    def affine(slope, phase):
        return (slope % 11, phase % 11, 0, 1)

    def component(grid, side):
        rows = [0]*q
        for j, cell in enumerate(grid[0]):
            for g in cell:
                block = actions[side][G.idx(g)]
                for r, word in enumerate(block):
                    rows[r] ^= word << (j*q)
        return rows

    def fingerprint(rows):
        """Rooted uncolored component invariant, not full-code identity."""
        adjacency = [[] for _ in range(q+width)]
        columns = _f57_columns(rows, width)
        for c, value in enumerate(columns):
            while value:
                bit = value & -value
                r = bit.bit_length()-1
                adjacency[r].append(q+c)
                adjacency[q+c].append(r)
                value ^= bit
        colors = [0]*q + [1+c.bit_count() for c in columns]
        colors[0] = 9
        digest = hashlib.sha256()
        for _ in range(12):
            signatures = [
                (colors[v], tuple(sorted(colors[w]
                                         for w in adjacency[v])))
                for v in range(q+width)
            ]
            digest.update(repr(sorted(
                Counter(signatures).items())).encode("ascii"))
            palette = {s: i for i, s in enumerate(sorted(set(signatures)))}
            colors = [palette[s] for s in signatures]
        return digest.hexdigest()

    # Exclude component classes explicitly visible in the shown measured
    # long-binomial slots. This does NOT replace the complete run archive.
    displayed = (
        ((1,8), (3,0), (3,2), (9,1)),
        ((1,8), (3,0), (3,2), (9,2)),
        ((1,3), (3,0), (3,9), (9,1)),
        ((3,0), (9,7), (3,5), (5,2)),
        ((1,9), (3,0), (3,4), (5,2)),
        ((1,8), (3,0), (3,5), (5,9)),
        ((1,9), (3,0), (1,3), (5,6)),
        ((1,4), (4,0), (3,9), (5,0)),
        ((1,3), (3,0), (3,4), (9,9)),
        ((1,3), (5,0), (3,1), (5,5)),
    )
    taboo = set()
    for u, v, x, y in displayed:
        grid = [[[G.e, affine(1,1)],
                 [G.e, affine(*u), affine(*v)],
                 [G.e, affine(*x), affine(*y)]]]
        taboo.add(fingerprint(component(grid, "A")))

    def draw(brief):
        cells = [[G.e, affine(1,1)]]
        for position, kind in enumerate(brief):
            if kind < 4:
                u = affine(1, rng.randrange(1,11))
                v = affine(pow(3, kind+1, 11),
                           0 if position == 0 else rng.randrange(11))
            else:
                exponent = 2 if kind == 4 else 3
                u = affine(3, 0 if position == 0 else rng.randrange(11))
                v = affine(pow(3, exponent, 11), rng.randrange(11))
            if G.mul(u, v) == G.mul(v, u):
                return None
            cell = [G.e, u, v]
            assert len({G.idx(g) for g in cell}) == 3
            cells.append(cell)
        return [cells]

    def connected(rows):
        adjacency = [0]*q
        for i, row in enumerate(rows):
            for j in range(i):
                if row & rows[j]:
                    adjacency[i] |= 1 << j
                    adjacency[j] |= 1 << i
        reached = frontier = 1
        while frontier:
            following = 0
            while frontier:
                bit = frontier & -frontier
                following |= adjacency[bit.bit_length()-1]
                frontier ^= bit
            frontier = following & ~reached
            reached |= frontier
        return reached == (1 << q)-1

    def census56(columns):
        """Exact mixed-degree five/six counts after clean4."""
        pairs = {}
        for i, x in enumerate(columns):
            for y in columns[:i]:
                value = x ^ y
                pairs[value] = pairs.get(value, 0)+1
        triples = {}
        five = six = boundary = 0
        loads = [0]*3
        size = len(columns)
        for a in range(size-2):
            for b in range(a+1, size-1):
                pair = columns[a] ^ columns[b]
                for c in range(b+1, size):
                    value = pair ^ columns[c]
                    five += pairs.get(value, 0)
                    six += triples.get(value, 0)
                    triples[value] = triples.get(value, 0)+1
                    weight = value.bit_count()
                    if weight <= 4:
                        cost = (5-weight)**2
                        boundary += cost
                        loads[a//q] += cost
                        loads[b//q] += cost
                        loads[c//q] += cost
        # clean4 forces the matching supports to be disjoint.
        assert five % 10 == 0 and six % 10 == 0
        return five//10, six//10, max(loads), boundary

    def census7(columns):
        """Exact seven-word count and block exposure after clean6.

        For anchor a, two unanchored triples have syndrome XOR c_a.
        Exclusion through six makes their supports disjoint and makes
        triple syndromes unique. Each seven-word containing a contributes
        twenty ordered triple partitions of its other six variables.

        Regular lift translations are transitive within each column
        block. Summing anchored counts and multiplying by q counts every
        seven-word seven times; no free-orbit assumption is made.
        """
        anchored = []
        for anchor in (0, q, 2*q):
            other = columns[:anchor] + columns[anchor+1:]
            syndromes = set()
            size = len(other)
            for a in range(size-2):
                for b in range(a+1, size-1):
                    pair = other[a] ^ other[b]
                    for c in range(b+1, size):
                        syndromes.add(pair ^ other[c])
            assert len(syndromes) == size*(size-1)*(size-2)//6
            value = columns[anchor]
            hits = sum((s ^ value) in syndromes for s in syndromes)
            assert hits % 20 == 0
            anchored.append(hits//20)
        incidences = q*sum(anchored)
        assert incidences % 7 == 0
        return (incidences//7, max(anchored),
                sum(v*v for v in anchored), tuple(anchored))

    def select(side, brief, restart):
        pool, seen = [], set()
        accepted = 0
        for trial in range(3200):
            grid = draw(brief)
            if grid is None:
                continue
            identity = tuple(sorted(
                tuple(sorted(G.idx(g) for g in cell))
                for cell in grid[0]))
            if identity in seen:
                continue
            seen.add(identity)
            rows = component(grid, side)
            if gf2_rank(rows) != q:
                continue
            columns = _f57_columns(rows, width)
            assert all(c.bit_count() == (2 if j < q else 3)
                       for j, c in enumerate(columns))
            four, cycles = _f57_cycles(rows, columns)
            if four or not connected(rows) or not _f57_clean4(columns):
                continue
            accepted += 1
            pool.append((cycles, trial, grid, rows, columns))
            pool.sort(key=lambda item: item[:2])
            del pool[8:]
            if accepted >= 80 + 32*restart:
                break

        clean = []
        classes = set()
        for cycles, trial, grid, rows, columns in pool:
            fp = fingerprint(rows)
            if fp in taboo or fp in classes:
                continue
            classes.add(fp)
            five, six, peak, boundary = census56(columns)
            if five or six:
                continue
            clean.append(((peak, boundary, cycles), trial,
                          grid, rows, columns, fp))
        clean.sort(key=lambda item: item[:2])

        refined = []
        for coarse, trial, grid, rows, columns, fp in clean[:4]:
            words7, peak7, concentration7, exposure7 = census7(columns)
            distance, short8 = _f55_sampled_protection(
                columns, 550700791 + 1009*idx + restart)
            key = (words7, peak7, concentration7, short8,
                   max(0, 8-distance)) + coarse
            refined.append((key, fp, grid, rows, exposure7))
        refined.sort(key=lambda item: item[:2])
        return refined[:2]

    def noisy_probe(checks, stabilizers, n, seed, shots):
        """Two integrated fault intervals, NOT ASC circuit measurement.

        Equations: H*e1+m=s1 and H*(e1+e2)=s2.
        The intermediate syndrome is noisy and the final one perfect.
        Opposite-check proper suffixes approximate ancilla spread with
        unknown random gate orders. BP settlement is classified modulo
        the verified stabilizer span; unresolved shots count as failures.
        """
        import numpy as np

        def sites(row):
            result = []
            while row:
                bit = row & -row
                result.append(bit.bit_length()-1)
                row ^= bit
            return result

        m = len(checks)
        variables = 2*n+m
        augmented = [
            row | (1 << (2*n+r)) for r, row in enumerate(checks)
        ] + [row | (row << n) for row in checks]
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

        sensed = np.asarray([
            c.bit_count() for c in _f57_columns(checks, n)])
        opposite = np.asarray([
            c.bit_count() for c in _f57_columns(stabilizers, n)])
        levels = np.asarray((0.35, 0.60, 0.85))[np.arange(shots) % 3]
        data_mu = 3*0.002*(2+sensed+opposite)*2/3
        measurement_mu = np.asarray([
            3*0.002*(row.bit_count()+2)*2/3 for row in checks])
        data_p = 0.5*(1-np.exp(-2*levels[:, None]*data_mu[None, :]))
        measure_p = 0.5*(1-np.exp(
            -2*levels[:, None]*measurement_mu[None, :]))
        channel = np.random.default_rng(seed)
        error = np.zeros((shots, variables+1), dtype=bool)
        error[:, :n] = channel.random((shots, n)) < data_p
        error[:, n:2*n] = channel.random((shots, n)) < data_p
        error[:, 2*n:variables] = channel.random((shots, m)) < measure_p

        hooks = [sites(row) for row in stabilizers]
        for ns in hooks:
            order = channel.permutation(ns)
            event_p = levels*3*0.002*len(ns)*2/3
            for interval in range(2):
                for shot in np.flatnonzero(channel.random(shots) < event_p):
                    cut = int(channel.integers(1, len(order)))
                    error[shot, interval*n+order[cut:]] ^= True

        # Marginal hook allowance is heuristic, not a calibrated channel.
        marginal_mu = data_mu + 3*0.002*opposite*0.35
        marginal_p = np.clip(0.5*(1-np.exp(
            -2*levels[:, None]*marginal_mu[None, :])), 0.002, 0.25)
        probabilities = np.concatenate(
            (marginal_p, marginal_p, np.clip(measure_p, 0.002, 0.25)),
            axis=1)
        prior = np.full((shots, variables+1), 30.0)
        prior[:, :variables] = np.log((1-probabilities)/probabilities)
        target = (
            error[:, flat].reshape(shots, 2*m, row_width).sum(axis=2) & 1
        ).astype(bool)
        target_sign = np.where(target, -1.0, 1.0)
        messages = prior[:, flat].copy()
        messages[:, ~valid] = 30.0
        true_final = error[:, :n] ^ error[:, n:2*n]

        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length()-1
                if p in pivots:
                    row ^= pivots[p]
                else:
                    pivots[p] = row
                    break

        def nontrivial(word):
            while word:
                p = word.bit_length()-1
                if p not in pivots:
                    return True
                word ^= pivots[p]
            return False

        wrong = late = work = 0
        for iteration in range(1, 37):
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
                * target_sign[:, :, None]
            ).reshape(count, edges)
            outgoing[:, ~valid] = 0.0
            padded = np.zeros((count, edges+1))
            padded[:, :edges] = outgoing
            posterior = prior + padded[:, variable_edges].sum(axis=2)
            posterior[:, variables] = 30.0
            hard = posterior < 0
            syndrome = (
                hard[:, flat].reshape(count, 2*m, row_width).sum(axis=2) & 1
            ).astype(bool)
            settled = np.all(syndrome == target, axis=1)
            final = hard[:, :n] ^ hard[:, n:2*n]
            for local in np.flatnonzero(settled):
                word = 0
                for c in np.flatnonzero(final[local] ^ true_final[local]):
                    word |= 1 << int(c)
                wrong += int(nontrivial(word))
                late += int(iteration > 12)
                work += iteration
            keep = ~settled
            if not np.any(keep):
                return wrong, 0, late, work
            damping = 0.25 if iteration <= 12 else 0.50
            messages = np.clip(
                damping*messages +
                (1-damping)*(posterior[:, flat]-outgoing), -30, 30)[keep]
            messages[:, ~valid] = 30.0
            prior = prior[keep]
            target = target[keep]
            target_sign = target_sign[keep]
            true_final = true_final[keep]
        unresolved = len(prior)
        return (wrong+unresolved, unresolved,
                late+unresolved, work+37*unresolved)

    def quality(x, z):
        return (max(x[0], z[0]), x[0]+z[0],
                max(x[1], z[1]), x[1]+z[1],
                max(x[2], z[2]), x[2]+z[2], x[3]+z[3])

    seed = 550700791 + 1009*idx
    for restart in range(3):
        aa = select("A", ba, restart)
        be = select("B", bb, restart)
        options, identities = [], set()
        for ka, fa, A, ar, ea in aa:
            for kb, fb, B, br, eb in be:
                identity = tuple(sorted((fa, fb)))
                if identity in identities:
                    continue
                identities.add(identity)
                geometry = tuple(v for a, b in zip(ka, kb)
                                 for v in (max(a,b), a+b))
                options.append((geometry, identity, A, B, ar, br, ea, eb))
        options.sort(key=lambda item: item[:2])

        training = []
        for number, entry in enumerate(options[:4]):
            geometry, identity, A, B, ar, br, ea, eb = entry
            hx, hz, n = lifted_matrices(G, A, B)
            k = n-gf2_rank(hx)-gf2_rank(hz)
            if n != 550 or k != 220:
                continue
            actual_a = [
                sum(((row >> (3*j*q)) & mask) << (j*q)
                    for j in range(3)) for row in hx[:q]
            ]
            actual_b = [row & ((1 << width)-1) for row in hz[:q]]
            assert actual_a == ar and actual_b == br
            if _f57_small_logical(hx, hz, n):
                continue
            if _f57_small_logical(hz, hx, n):
                continue
            assert css_ok(hx, hz)
            assert max(row.bit_count() for row in hx+hz) == 11
            assert 6*sum(row.bit_count() for row in hx+hz) == 21120
            x = noisy_probe(hz, hx, n, seed, 48)
            z = noisy_probe(hx, hz, n, seed+1, 48)
            training.append((quality(x,z)+geometry+(number,),
                             number, entry, hx, hz, x, z))
        training.sort(key=lambda item: item[:2])

        finalists = []
        for _, number, entry, hx, hz, tx, tz in training[:2]:
            geometry, identity, A, B, ar, br, ea, eb = entry
            # Independent holdout results lead; training is only a tie-break.
            x = noisy_probe(hz, hx, 550, seed+1000003, 96)
            z = noisy_probe(hx, hz, 550, seed+1000004, 96)
            key = quality(x,z)+quality(tx,tz)+geometry+(number,)
            finalists.append((key, A, B, identity, ea, eb, x, z, tx, tz))
        if finalists:
            finalists.sort(key=lambda item: item[0])
            key, A, B, identity, ea, eb, x, z, tx, tz = finalists[0]
            return {"idx": idx, "briefs": (ba, bb), "k": 220,
                    "component_identity": identity,
                    "anchored_seven_exposure": (ea, eb),
                    "training_surrogate": (tx, tz),
                    "holdout_surrogate": (x, z), "screen": key,
                    "cells": {"A": A, "B": B}}
    raise ValueError("no exact-seven light code passed complete screening")


def CELL(rng, G, side, i, j, ctx):
    """Emit the complete holdout-selected cell without resampling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal rolled supports and uniform labels alone give reliable r040f memory: parent fitness 3.2187, all five measured codes hit a failed-shot cap, with weight-4 sector-1 witnesses and broadly distributed logical flips.
# REFUTED: Balanced short-collision optimization on D58 (1,3)x(1,3) ensures rare circuit failures: fitness was 4.3650, every measured basis hit the failed-shot cap, and slot 0 had 105/3500 X failures despite d2s=8.
# REFUTED: Removing involutions and independently optimizing four-cycle-free components through weight-six multiplicity ensures rare memory failures: F57 fitness was 4.8170, all ten measured bases hit failed-shot caps, and slot14 had 102/5000 X failures with flips across all 228 logical qubits.
# REFUTED: Full-rank affine trinomials, component weight-six screening and perfect-syndrome complete-code BP selection suffice for rare basis-balanced ASC failures: current fitness was 4.8678, all ten measured bases hit failed-shot caps, and slot1 failed 102/4500 X versus 101/11750 Z shots with flips across all 220 logicals.
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
