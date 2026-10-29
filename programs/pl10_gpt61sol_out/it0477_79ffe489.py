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
CONCEPT = ("For cyclic C19, order 19, shapes (1,4)x(2,7), two distinct binomials and controlled overlap between Sidon trinomials, paired with triangle-free graphic B components, should improve worse-basis ASC circuit fitness above the parent's 3.2824 in cell 0.500.")
GROUP = ("Jump from nonabelian M16 to C19, a genuinely different group and the largest legal lift for size factor thirty. The C19 bed has measured 3.3458, whereas the parent measured 3.2824. This motivates a fresh filling experiment, not a claim that cyclicity or order alone causes reliability.")
SHAPES = ("Keep (1,4)x(2,7): q=19 gives n=570, kb=285 and guaranteed rate 0.500. Unequal components have 76 variables/19 checks and 133 variables/38 checks. Require complete verified k=288, rate 0.505263 and k-kb=3. Parent failures affected all 243 logicals with mostly 1.0-1.2x Acol/Bcol and Acol/Brow enrichment; no persistent isolated repair block was established.")
FILLING = ("Twenty contracts cross five Sidon-overlap prescriptions with four B incidence architectures; further batches consume further invariant-distinct A classes. A changes from [1,3,3,3] to [2,2,3,3], eliminating the unit-pivot degree-three data blocks. Both trinomials have three distinct internal directions, but share one or two directions; prescribed binomial intersection counts distinguish five contracts. This differs from the displayed disjoint-Sidon and AP models. Canonical cyclic gauges and rooted component invariants conservatively discard copies. Verified components exclude words through three, B must be connected and triangle-free, and full-CSS singleton/pair classification rejects weight-three logical ambiguity. Independent two-interval noisy-syndrome selection samples compare finalists. They omit actual ASC ordering and relay/LSD and are uncalibrated pL surrogates. HX weighs twelve, HZ at most eleven, and extraction uses 18240 six-round CNOTs. Target the parent's 110-134 failed Z shots/250, 103-124 failed X shots/250 and collective flips; the complete evaluator archive remains authoritative.")
FAMILY_NAME = "r050_C19_14x27_overlapSidon_trianglefree_twotime_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 19

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C19 requires order 19")
    els = [(a,) for a in range(q)]

    def red(g):
        return (int(g[0]) % q,)

    def mul(g, h):
        a, b = red(g)[0], red(h)[0]
        return ((a+b) % q,)

    def inv(g):
        return ((-red(g)[0]) % q,)

    def idx(g):
        return red(g)[0]

    return dict(elements=els, ident=(0,), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 4)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb)


def _m16_columns(rows, width):
    """Transpose already verified component or CSS rows."""
    out = [0] * width
    for r, row in enumerate(rows):
        tag = 1 << r
        while row:
            bit = row & -row
            out[bit.bit_length()-1] |= tag
            row ^= bit
    return out


def _m16_identity(rows, width, q):
    """Rooted Tanner invariant, not complete graph canonization.

    Regular translations cover each check-row orbit. Minimize over
    one root per orbit; unequal invariants certify unequal components.
    Equal invariants are conservatively discarded.
    """
    import hashlib
    from collections import Counter

    nr = len(rows)
    adjacency = [[] for _ in range(nr + width)]
    for c, value in enumerate(_m16_columns(rows, width)):
        while value:
            bit = value & -value
            r = bit.bit_length()-1
            adjacency[r].append(nr+c)
            adjacency[nr+c].append(r)
            value ^= bit

    images = []
    for root in range(0, nr, q):
        colors = [0] * nr + [1] * width
        colors[root] = 2
        digest = hashlib.sha256()
        for iteration in range(11):
            signatures = [
                (colors[v], tuple(sorted(colors[w]
                                         for w in adjacency[v])))
                for v in range(nr + width)
            ]
            digest.update(repr(sorted(
                Counter(signatures).items())).encode("ascii"))
            palette = {s: t for t, s in enumerate(
                sorted(set(signatures)))}
            colors = [palette[s] for s in signatures]
        images.append(digest.hexdigest())
    return min(images)


def _component_rows(hx, hz, q):
    """Extract components from the verified lift, never rebuild a lift."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    ar = []
    for i in range(ra):
        for t in range(q):
            row = hx[(i * cb) * q + t]
            ar.append(sum(((row >> (j * cb * q)) & mask) << (j * q)
                          for j in range(ca)))
    bmask = (1 << (cb * q)) - 1
    br = [row & bmask for row in hz[:rb * q]]
    return ar, br


def _four_cycles(rows):
    return sum((v * (v - 1)) // 2
               for i, row in enumerate(rows)
               for other in rows[:i]
               for v in [(row & other).bit_count()])


def _six_cycles(rows):
    """Count cycles using three checks and three distinct data columns."""
    m = len(rows)
    overlap = [[0] * m for _ in range(m)]
    for i in range(m):
        for j in range(i):
            overlap[i][j] = overlap[j][i] = (
                rows[i] & rows[j]).bit_count()
    total = 0
    for i in range(m):
        for j in range(i):
            x = overlap[i][j]
            if not x:
                continue
            common = rows[i] & rows[j]
            for h in range(j):
                y, z = overlap[j][h], overlap[i][h]
                if y and z:
                    triple = (common & rows[h]).bit_count()
                    total += x * y * z - triple * (x + y + z) + 2 * triple
    return total


def _kernel_statistics(rows, ncols):
    """Short component words and small boundaries, from verified rows.

    When column syndromes are distinct and nonzero, the normalized
    collision counts exactly count weight-three and weight-four words.
    Otherwise the duplicate statistic takes priority in the search.
    These statistics are not asserted to predict circuit pL exactly.
    """
    columns = [0] * ncols
    for r, row in enumerate(rows):
        bit = 1 << r
        while row:
            low = row & -row
            columns[low.bit_length() - 1] |= bit
            row ^= low

    singles = set(columns)
    duplicates = ncols - len(singles) + int(0 in singles)
    pairs = {}
    triples = collisions = boundary = 0
    for i, x in enumerate(columns):
        for y in columns[:i]:
            syndrome = x ^ y
            triples += int(syndrome in singles)
            old = pairs.get(syndrome, 0)
            collisions += old
            pairs[syndrome] = old + 1
            w = syndrome.bit_count()
            if w < 3:
                boundary += (3 - w) ** 2
    return duplicates, triples // 3, collisions // 3, boundary


def NEW_CTX(rng, G, idx):
    """Overlapping-Sidon contracts and verified triangle-free B selection."""
    import math
    from itertools import combinations

    q = G.q
    assert q == 19
    assert A_SHAPE == (1, 4) and B_SHAPE == (2, 7)
    architecture, marker = divmod(idx % 20, 5)
    wanted = 5*(idx // 20) + marker
    overlap_card = ((1,0), (1,1), (1,2), (2,0), (2,1))[marker]
    mask = (1 << q)-1

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    # Cache the scaffold's actual left/right singleton actions.
    # These are not independently implemented regular representations.
    actions = {"A": {}, "B": {}}
    for g in G.elements:
        hx1, _, _ = lifted_matrices(G, [[[g]]], [[[]]])
        _, hz1, _ = lifted_matrices(G, [[[]]], [[[g]]])
        actions["A"][G.idx(g)] = tuple(row & mask for row in hx1)
        actions["B"][G.idx(g)] = tuple(row & mask for row in hz1)

    def component(grid, side):
        rows = [0] * (len(grid)*q)
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                for g in cell:
                    block = actions[side][G.idx(g)]
                    for t, value in enumerate(block):
                        rows[i*q+t] ^= value << (j*q)
        return rows, len(grid[0])*q

    # Every nonidentity element of C19 has order nineteen. An internal
    # graph edge therefore has two distinct endpoints, without the
    # parallel-edge problem caused by involutory ratios.
    representatives = [
        g for g in G.elements
        if g != G.e and G.idx(g) < G.idx(G.inv(g))]
    assert len(representatives) == 9

    def direction(value):
        value %= q
        return min(value, q-value)

    def directions(cell):
        return {direction(a-b) for a, b in combinations(cell, 2)}

    def normal(cell, multiplier=1):
        values = [(multiplier*a) % q for a in cell]
        return min(tuple(sorted((a-origin) % q for a in values))
                   for origin in values)

    def geometry_key(cells):
        # A has one check-row orbit. Independent column translations,
        # a common cyclic automorphism and column permutations are gauges.
        return min(tuple(sorted(normal(cell, u) for cell in cells))
                   for u in range(1, q))

    triangles = sorted({
        normal((0, a, b))
        for a, b in combinations(range(1, q), 2)
        if len(directions((0, a, b))) == 3
    })
    assert triangles and all(len(directions(t)) == 3 for t in triangles)

    def component_pair_loss(rows, width):
        """Degree-weighted classical ambiguity, not quantum circuit pL."""
        cols = _m16_columns(rows, width)
        exposure = [3 + value.bit_count() for value in cols]
        buckets = {}
        for i, value in enumerate(cols):
            for j in range(i):
                syndrome = value ^ cols[j]
                mass = exposure[i]*exposure[j]
                bucket = buckets.setdefault(syndrome, [0, 0])
                bucket[0] += mass
                bucket[1] = max(bucket[1], mass)
        losses = [total-largest for total, largest in buckets.values()]
        return sum(losses), max(losses, default=0)

    def component_connected(rows):
        reached = 1
        while True:
            variables = 0
            for r, row in enumerate(rows):
                if reached & (1 << r):
                    variables |= row
            following = reached | sum(
                1 << r for r, row in enumerate(rows) if row & variables)
            if following == reached:
                return reached == (1 << len(rows))-1
            reached = following

    # Five invariant cards prescribe:
    # (shared trinomial directions, binomial directions in their union).
    # All trinomials are Sidon, but none of these cards permits the
    # displayed difference-disjoint Sidon model.
    catalogue_rng = random.Random(190519731 + 7919*marker)
    bank, seen_raw, seen_classes = [], set(), set()
    capacity = max(32, wanted+8)
    for trial in range(8192):
        first, second = catalogue_rng.sample(triangles, 2)
        df, ds = directions(first), directions(second)
        if len(df & ds) != overlap_card[0]:
            continue
        union = df | ds
        choices = [
            b for b in range(2, (q+1)//2)
            if len({1, b} & union) == overlap_card[1]]
        if not choices:
            continue
        b = catalogue_rng.choice(choices)
        raw = geometry_key(((0,1), (0,b), first, second))
        if raw in seen_raw:
            continue
        seen_raw.add(raw)
        grid = [[[(t,) for t in cell] for cell in raw]]
        rows, width = component(grid, "A")
        stats = _kernel_statistics(rows, width)
        if any(stats[:2]) or gf2_rank(rows) != q:
            continue
        if not component_connected(rows):
            continue
        loss, peak = component_pair_loss(rows, width)
        key = (loss, peak, stats[2], _four_cycles(rows),
               _six_cycles(rows), stats[3])
        if len(bank) == capacity and key > bank[-1][0][:-1]:
            continue
        identity = _m16_identity(rows, width, q)
        if identity in seen_classes:
            continue
        seen_classes.add(identity)
        bank.append((key+(identity,), identity, grid, rows))
        bank.sort(key=lambda item: item[0])
        del bank[capacity:]

    if wanted >= len(bank):
        raise ValueError("invariant-distinct overlapping-Sidon A book exhausted")
    akey, aid, A, ar = bank[wanted]
    actual_tri = [tuple(g[0] for g in cell)
                  for cell in A[0] if len(cell) == 3]
    actual_bin = {
        next(iter(directions(tuple(g[0] for g in cell))))
        for cell in A[0] if len(cell) == 2}
    da, db = map(directions, actual_tri)
    assert all(len(d) == 3 for d in (da, db))
    assert (len(da & db), len(actual_bin & (da | db))) == overlap_card

    profiles = (
        ((1,1),) * 7,
        ((2,0),) + ((1,1),) * 6,
        ((2,0),(0,2)) + ((1,1),) * 5,
        ((2,0),(2,0),(0,2),(0,2),(1,1),(1,1),(1,1)),
    )
    profile = profiles[architecture]
    sa = [[len(cell) for cell in A[0]]]
    sb = [[p[i] for p in profile] for i in range(2)]
    assert sorted(sa[0]) == [2,2,3,3]
    assert sum(sa[0]) == 10 and sum(map(sum, sb)) == 14
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) == 12 <= W_MAX
    assert max(map(sum, sb)) + max(sa[0]) <= 11

    internal = [(i, j) for j, p in enumerate(profile)
                for i in range(2) if p[i] == 2]
    cross = [j for j, p in enumerate(profile) if p == (1,1)]
    mutable = internal + [(1, j) for j in cross]

    def internal_choices(grid, i, j):
        occupied = {G.idx(grid[i][c][1])
                    for r, c in internal if r == i and c != j
                    and len(grid[i][c]) == 2}
        return [g for g in representatives if G.idx(g) not in occupied]

    def draw_B():
        grid = [[[] for _ in range(7)] for _ in range(2)]
        for i, j in internal:
            grid[i][j] = [G.e, rng.choice(internal_choices(grid, i, j))]
        for j, voltage in zip(cross, rng.sample(G.elements, len(cross))):
            grid[0][j], grid[1][j] = [G.e], [voltage]
        return grid

    def graph_statistics(grid):
        """Exact incidence-graph statistics from helper-derived actions."""
        rows, width = component(grid, "B")
        columns = _m16_columns(rows, width)
        assert all(value.bit_count() == 2 for value in columns)
        repeated = width-len(set(columns))
        adjacency = [0] * len(rows)
        edges = set()
        for value in columns:
            bit = value & -value
            a = bit.bit_length()-1
            b = (value ^ bit).bit_length()-1
            edges.add((a, b))
            adjacency[a] |= 1 << b
            adjacency[b] |= 1 << a

        triangle_sum = sum((adjacency[a] & adjacency[b]).bit_count()
                           for a, b in edges)
        assert triangle_sum % 3 == 0
        triangles_count = triangle_sum // 3
        partitions = sum(
            v*(v-1)//2
            for a, neighbors in enumerate(adjacency)
            for other in adjacency[:a]
            for v in [(neighbors & other).bit_count()])
        assert partitions % 2 == 0
        quartets = partitions // 2

        unseen = (1 << len(rows))-1
        parts = 0
        while unseen:
            frontier = unseen & -unseen
            reached = frontier
            while frontier:
                following = 0
                while frontier:
                    bit = frontier & -frontier
                    following |= adjacency[bit.bit_length()-1]
                    frontier ^= bit
                frontier = following & ~reached
                reached |= frontier
            unseen &= ~reached
            parts += 1
        stats = (repeated, triangles_count, quartets, parts)
        energy = (800*repeated + 80*triangles_count + quartets
                  + 200*(parts-1)) / q
        return energy, stats, rows

    pool, seen_B = [], set()
    serial = 0

    def remember(grid, stats, rows):
        nonlocal serial
        if stats[0] or stats[1] or stats[3] != 1:
            return
        key = stats[1:3]
        if len(pool) == 12 and key > pool[-1][0]:
            return
        signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                          for row in grid for cell in row)
        if signature in seen_B:
            return
        seen_B.add(signature)
        assert gf2_rank(rows) == 2*q-1
        serial += 1
        pool.append((key, serial, clone(grid), list(rows)))
        pool.sort(key=lambda item: item[:2])
        del pool[12:]

    for restart in range(4):
        grid = draw_B()
        energy, stats, rows = graph_statistics(grid)
        remember(grid, stats, rows)
        for step in range(320):
            proposal = clone(grid)
            i, j = rng.choice(mutable)
            if (i, j) in internal:
                proposal[i][j] = [
                    G.e, rng.choice(internal_choices(proposal, i, j))]
            else:
                occupied = {proposal[1][c][0]
                            for c in cross if c != j}
                choices = [g for g in G.elements if g not in occupied]
                proposal[i][j] = [rng.choice(choices)]
            proposed, ps, pr = graph_statistics(proposal)
            delta = proposed-energy
            temperature = 8.0*(0.985**step) + 0.04
            if delta <= 0 or rng.random() < math.exp(
                    -min(700.0, delta/temperature)):
                grid, energy, stats, rows = proposal, proposed, ps, pr
                remember(grid, stats, rows)

    if not pool:
        raise ValueError("no connected simple B incidence graph")

    shortlist, bclasses = [], set()
    for coarse, serial, B, br in pool:
        bid = _m16_identity(br, 7*q, q)
        if bid in bclasses:
            continue
        bclasses.add(bid)
        shortlist.append((coarse, serial, bid, B, br))
        if len(shortlist) == 4:
            break

    def quotient_columns(stabilizers, width):
        """Linear canonical remainder modulo verified stabilizer rows."""
        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = row
                    break
                row ^= old
        out = [0] * width
        for c in range(width):
            row = pivots.get(c)
            if row is None:
                out[c] = 1 << c
                continue
            tail, value = row ^ (1 << c), 0
            while tail:
                bit = tail & -tail
                value ^= out[bit.bit_length()-1]
                tail ^= bit
            out[c] = value
        return out

    def fault_mass(checks, stabilizers, width):
        """Stabilizer-aware fault ambiguity, NOT an ASC pL estimate.

        Group singleton/pair faults by actual syndrome and quotient.
        Degree weights approximate gate exposure. Shared checks add
        only a diluted possible-spread term, not certified ASC hooks.
        """
        syndrome = _m16_columns(checks, width)
        spread = _m16_columns(stabilizers, width)
        residue = quotient_columns(stabilizers, width)
        exposure = [1+a.bit_count()+b.bit_count()
                    for a, b in zip(syndrome, spread)]
        buckets = {0: {0: 1000000000}}
        singles = {}
        for s, label in zip(syndrome, residue):
            singles.setdefault(s, set()).add(label)

        def add(s, label, mass):
            classes = buckets.setdefault(s, {})
            classes[label] = classes.get(label, 0)+mass

        for i, s in enumerate(syndrome):
            add(s, residue[i], 4096*exposure[i])
        triple_partitions = 0
        for i, si in enumerate(syndrome):
            for j in range(i):
                s = si ^ syndrome[j]
                label = residue[i] ^ residue[j]
                triple_partitions += int(any(
                    other != label for other in singles.get(s, ())))
                mass = 64*exposure[i]*exposure[j]
                mass += 48*(spread[i] & spread[j]).bit_count()
                add(s, label, mass)

        loss = peak = ambiguous = 0
        for classes in buckets.values():
            if len(classes) > 1:
                error = sum(classes.values())-max(classes.values())
                loss += error
                peak = max(peak, error)
                ambiguous += 1
        # With distinct nonzero syndrome columns, triple_partitions/3
        # is the exact nontrivial weight-three count. Otherwise retain
        # the raw ambiguity penalty without claiming exact multiplicity.
        return triple_partitions, loss, peak, ambiguous

    def two_time_probe(checks, stabilizers, width, seed, shots):
        """Noisy-syndrome BP stress test, NOT the ASC memory experiment.

        Decode H*e1+m=s1 and H*(e1+e2)=s2. The second syndrome is
        perfect. Proper opposite-check suffixes are possible spread
        faults, not certified hooks of the evaluator's ASC ordering.
        Settled final residuals are classified modulo actual stabilizers.
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
        variables = 2*width + m
        augmented = [
            row | (1 << (2*width+r))
            for r, row in enumerate(checks)
        ] + [row | (row << width) for row in checks]
        neighbors = [sites(row) for row in augmented]
        opposite = [sites(row) for row in stabilizers]
        edge_vars = []
        incident = [[] for _ in range(variables)]
        for row in neighbors:
            for c in row:
                incident[c].append(len(edge_vars))
                edge_vars.append(c)
        ne = len(edge_vars)
        assert all(incident)
        max_check = max(map(len, neighbors))
        ce = np.full((2*m, max_check), ne, dtype=np.int32)
        cv = np.full((2*m, max_check), variables, dtype=np.int32)
        real = np.zeros((2*m, max_check), dtype=bool)
        offset = 0
        for r, row in enumerate(neighbors):
            size = len(row)
            ce[r, :size] = np.arange(offset, offset+size)
            cv[r, :size] = row
            real[r, :size] = True
            offset += size
        ve = np.full(
            (variables, max(map(len, incident))), ne, dtype=np.int32)
        for c, edges in enumerate(incident):
            ve[c, :len(edges)] = edges
        ev = np.asarray(edge_vars, dtype=np.int32)

        channel = np.random.default_rng(seed)
        intensity = np.asarray((0.35, 0.65))[np.arange(shots) % 2]
        sensed = _m16_columns(checks, width)
        spread = _m16_columns(stabilizers, width)
        degree = np.asarray([
            a.bit_count()+b.bit_count()
            for a, b in zip(sensed, spread)], dtype=float)
        single_mu = 0.004*degree
        data_p = 0.5*(1-np.exp(
            -2*intensity[:, None]*single_mu[None, :]))
        measurement_mu = 0.004*np.asarray([
            row.bit_count()+2 for row in checks], dtype=float)
        measurement_p = 0.5*(1-np.exp(
            -2*intensity[:, None]*measurement_mu[None, :]))

        errors = np.zeros((shots, variables+1), dtype=bool)
        errors[:, :width] = channel.random((shots, width)) < data_p
        errors[:, width:2*width] = (
            channel.random((shots, width)) < data_p)
        errors[:, 2*width:variables] = (
            channel.random((shots, m)) < measurement_p)

        hook_load = np.zeros(width, dtype=float)
        for row in opposite:
            if len(row) < 2:
                continue
            hook_load[row] += len(row)+1
            ordering = np.asarray(row, dtype=np.int32)
            channel.shuffle(ordering)
            event_p = 0.002*(len(row)+1)*intensity
            for interval in range(2):
                for shot in np.flatnonzero(
                        channel.random(shots) < event_p):
                    cut = int(channel.integers(1, len(row)))
                    errors[shot, interval*width+ordering[cut:]] ^= True

        # Marginal priors acknowledge spread exposure, but do not
        # represent the true correlated channel or relay/LSD decoding.
        marginal_mu = single_mu + 0.001*hook_load
        marginal_p = np.clip(0.5*(1-np.exp(
            -2*intensity[:, None]*marginal_mu[None, :])), 0.003, 0.22)
        probabilities = np.concatenate((
            marginal_p, marginal_p,
            np.clip(measurement_p, 0.003, 0.22)), axis=1)
        prior = np.log((1-probabilities)/probabilities)
        target = np.logical_xor.reduce(errors[:, cv], axis=2)
        signs = 1-2*target.astype(float)
        true_final = errors[:, :width] ^ errors[:, width:2*width]
        residue = quotient_columns(stabilizers, width)
        messages = prior[:, ev].copy()
        wrong = work = 0

        for iteration in range(36):
            count = len(prior)
            work += count
            padded = np.concatenate((
                messages, np.full((count, 1), 30.0)), axis=1)
            t = np.tanh(np.clip(padded[:, ce], -24, 24)/2)
            t[:, ~real] = 1.0
            ones = np.ones((count, 2*m, 1))
            left = np.concatenate((
                ones, np.cumprod(t, axis=2)[:, :, :-1]), axis=2)
            suffix = np.cumprod(t[:, :, ::-1], axis=2)[:, :, ::-1]
            right = np.concatenate((suffix[:, :, 1:], ones), axis=2)
            update = 2*np.arctanh(np.clip(
                signs[:, :, None]*left*right, -0.999999, 0.999999))
            outgoing = np.zeros((count, ne+1))
            outgoing[:, :ne] = update[:, real]
            posterior = prior + outgoing[:, ve].sum(axis=2)
            hard = posterior < 0
            hard_pad = np.concatenate((
                hard, np.zeros((count, 1), dtype=bool)), axis=1)
            matched = np.logical_xor.reduce(hard_pad[:, cv], axis=2)
            settled = np.all(matched == target, axis=1)
            final = hard[:, :width] ^ hard[:, width:2*width]
            for shot in np.flatnonzero(settled):
                label = 0
                for c in np.flatnonzero(final[shot] ^ true_final[shot]):
                    label ^= residue[int(c)]
                wrong += int(bool(label))
            keep = ~settled
            if not np.any(keep):
                return wrong, 0, work
            proposal = posterior[:, ev]-outgoing[:, :ne]
            messages = np.clip(
                0.35*messages+0.65*proposal, -24, 24)[keep]
            prior = prior[keep]
            target = target[keep]
            signs = signs[keep]
            true_final = true_final[keep]

        unresolved = len(prior)
        return wrong+unresolved, unresolved, work

    options = []
    for coarse, serial, bid, B, br in shortlist:
        hx, hz, n = lifted_matrices(G, A, B)
        k = n-gf2_rank(hx)-gf2_rank(hz)
        if n != 570 or k != 288:
            continue
        px = fault_mass(hz, hx, n)
        pz = fault_mass(hx, hz, n)
        # Do not let a noisy surrogate promote a code with an exact
        # singleton/pair ambiguity representing a weight-three logical.
        if px[0] or pz[0]:
            continue
        structural = (
            max(px[0], pz[0]), px[0]+pz[0],
            max(px[1], pz[1])/k, (px[1]+pz[1])/k,
            max(px[2], pz[2]), px[2]+pz[2],
            px[3]+pz[3],
            _four_cycles(hx)+_four_cycles(hz),
            coarse, bid)
        options.append((structural, serial, B, br, hx, hz, k, px, pz, bid))
    if not options:
        raise ValueError("no verified k=288 overlapping-Sidon C19 finalist")
    options.sort(key=lambda item: item[:2])

    def quality(x, z):
        return (max(x[0], z[0]), x[0]+z[0],
                max(x[1], z[1]), x[1]+z[1], x[2]+z[2])

    # Shared streams compare alternatives, not different random channels.
    # A separate sample bank decides between the two training finalists.
    seed = 190519731 + 1009*idx
    training = []
    for number, option in enumerate(options):
        structural, serial, B, br, hx, hz, k, px, pz, bid = option
        tx = two_time_probe(hz, hx, 570, seed, 24)
        tz = two_time_probe(hx, hz, 570, seed+1, 24)
        training.append((
            quality(tx, tz)+structural+(number,), number, option, tx, tz))
    training.sort(key=lambda item: item[:2])

    winner = None
    for _, number, option, tx, tz in training[:2]:
        structural, serial, B, br, hx, hz, k, px, pz, bid = option
        bx = two_time_probe(hz, hx, 570, seed+1000003, 64)
        bz = two_time_probe(hx, hz, 570, seed+1000004, 64)
        score = quality(bx, bz)+quality(tx, tz)+structural+(number,)
        if winner is None or score < winner[0]:
            winner = (score, B, br, hx, hz, k, px, pz, bid, bx, bz)

    score, B, br, hx, hz, k, px, pz, bid, bx, bz = winner
    actual_a, actual_b = _component_rows(hx, hz, q)
    assert actual_a == ar and actual_b == br
    assert gf2_rank(actual_a) == q
    assert gf2_rank(actual_b) == 2*q-1
    assert css_ok(hx, hz)
    assert lifted_k(G, A, B) == 288
    assert 0.475 <= k/570 < 0.525
    assert max(row.bit_count() for row in hx) == 12
    assert max(row.bit_count() for row in hz) <= 11
    assert max(row.bit_count() for row in hx+hz) <= W_MAX
    assert 6*sum(row.bit_count() for row in hx+hz) == 18240
    assert all(len(cell) <= CELL_MAX and len(set(cell)) == len(cell)
               for grid in (A, B) for row in grid for cell in row)
    return {"A": A, "B": B, "k": k,
            "design": (architecture, wanted),
            "Sidon_overlap_contract": overlap_card,
            "component_classes": (aid, bid),
            "A_contract": akey, "fault_mass": (px, pz),
            "stress_selection_samples": (bx, bz),
            "cycle_score": score, "cnots": 18240}


def CELL(rng, G, side, i, j, ctx):
    """Emit the complete verified overlapping-Sidon pairing."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Weight-legal rolled supports with uniform labels reliably balance the two bases: the parent had 100+ X failures in 750..33250 shots versus only 0..12 Z failures in 200000 shots.
# REFUTED: Short-kernel and cycle screening alone guarantees a measurable ASC schedule: current-parent slot 14 had w=10 and d2s=15 but its X schedule timed out, contributing zero to fitness.
# REFUTED: Dispersed doubletons and component short-kernel/cycle selection suffice to suppress collective D8 high-rate memory failures: fitness was 3.2454, measured Z failures were 117-147/250, all 240 logicals were affected, and abundant weight-four witnesses remained despite mostly first-pass settlement.
# REFUTED: Noncommuting order-eight M16 A binomials and gate-weighted pair-ambiguity selection suffice for rare collective memory failures: fitness was 3.2891, measured Z failures were 116-127/250, all 243 logicals were affected, and abundant weight-four witnesses remained despite almost complete first-pass Z settlement.
# REFUTED: Unit-pivot M16 A and two-interval noisy-syndrome selection improved the measured mean above 3.2891: the result was 3.2824, Z failed 110-134/250 shots, all 243 logicals were affected, and abundant weight-four witnesses persisted; this comparison does not establish statistical significance.
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
