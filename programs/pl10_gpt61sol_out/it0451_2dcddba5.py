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
CONCEPT = ("For AGL(1,8), order 56, shapes (1,3)x(1,3), balanced (2,3,3) components with order-seven binomials, short-word screening and noisy-syndrome holdout selection should raise measured ASC fitness above 4.8989; circuit measurement decides the protection-versus-exposure trade.")
GROUP = ("Frozen L1 GF(8) affine group, order and encoding. Keep exactly the existing construction using x^3+x+1, with translation subgroup C2^3 and slope subgroup C7. This bed measured 4.8989. Nonidentity slopes supply order-seven binomials without using translation involutions; this is a controlled filling experiment, not evidence of a group advantage.")
SHAPES = ("Frozen equal (1,3) sides give 168-variable, 56-check components, n=560, kb=224 and guaranteed rate 0.400; size factor ten permits q<=59. Require complete helper-built k=224, retaining zero k-kb. Parent weight-eight witnesses and collective flips span sector-one Acol/Bcol blocks. Data and Acol/Brow check enrichment is mostly 1.0-1.2x, without a persistent isolated repair target.")
FILLING = ("Both sides use (2,3,3): one order-seven slope binomial and two noncommuting distinct-slope trinomials. Maximum CSS check weight falls from twelve to eleven and six-round CNOTs from 24192 to 21504. The binomial necessarily supplies component weight-seven words, so larger distance is not the hypothesis. Indexed unordered quotient contracts prescribe finite fresh voltage searches, not saved-code images. Helper-derived actions screen connectivity, full rank, C4 and kernels through four; anchored five/six-word collisions, boundaries and expansion rank invariant-distinct alternatives. Higher-word counts are penalties, not hard requirements that void difficult contracts. Complete verified CSS finalists require k=224 and no quantum logical through four. Two fault intervals include one noisy intermediate syndrome, a perfect final syndrome, degree-weighted data noise and opposite-check suffix faults; independent BP holdout results lead selection. These omit actual ASC ordering, six-round history and relay/LSD and are not calibrated circuit fitness. Target broad collective flips and slot4 X=100/6000 versus Z=100/17750, whose worst X logical #133 flipped in 58 failed shots.")
FAMILY_NAME = "r040_AGL8_q56_13x13_slopebinomial7_noisysyndrome_holdout_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 56

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("AGL(1,8) requires q=56")
    # (a,t) acts by z -> a*z+t over GF(8).
    # Three-bit polynomial encoding, modulo x^3+x+1.
    def field_mul(a, b):
        result = 0
        while b:
            if b & 1:
                result ^= a
            b >>= 1
            a <<= 1
            if a & 8:
                a ^= 11
        return result & 7

    table = [[field_mul(a, b) for b in range(8)]
             for a in range(8)]
    powers = [1]
    for _ in range(6):
        powers.append(table[powers[-1]][2])
    assert len(set(powers)) == 7
    reciprocal = {a: powers[(-i) % 7]
                  for i, a in enumerate(powers)}
    els = [(a, t) for a in powers for t in range(8)]
    positions = {g: i for i, g in enumerate(els)}

    def red(g):
        return (int(g[0]) & 7, int(g[1]) & 7)

    def mul(g, h):
        a, t = red(g)
        b, u = red(h)
        return (table[a][b], t ^ table[a][u])

    def inv(g):
        a, t = red(g)
        ai = reciprocal[a]
        return (ai, table[ai][t])

    def idx(g):
        return positions[red(g)]

    return dict(elements=els, ident=(1, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


def _f_columns(rows, width):
    out = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            out[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return out


def _f_low4(rows, stabilizers, width):
    """Exact existence of a nontrivial kernel word of weight <=4."""
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

    cols = _f_columns(rows, width)
    seen = {0: 0}
    for i, syndrome in enumerate(cols):
        word = 1 << i
        if syndrome in seen:
            if nontrivial(word ^ seen[syndrome]):
                return True
        else:
            seen[syndrome] = word
    for i, x in enumerate(cols):
        for j in range(i):
            syndrome = x ^ cols[j]
            word = (1 << i) | (1 << j)
            if syndrome in seen:
                if nontrivial(word ^ seen[syndrome]):
                    return True
            else:
                seen[syndrome] = word
    return False


def _f_graph(cols, nr):
    """Exact four/six-cycle statistics when four-cycles are absent."""
    adjacency = [0] * nr
    multiplicity = {}
    for column in cols:
        neighbors = []
        while column:
            bit = column & -column
            neighbors.append(bit.bit_length() - 1)
            column ^= bit
        assert len(neighbors) == 3
        for a in range(3):
            for b in range(a):
                i, j = sorted((neighbors[a], neighbors[b]))
                multiplicity[i, j] = multiplicity.get((i, j), 0) + 1
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i
    four = sum(v * (v - 1) // 2 for v in multiplicity.values())
    if four:
        return four, None, adjacency
    triangles = 0
    for i in range(nr):
        higher = adjacency[i] & ~((1 << (i + 1)) - 1)
        while higher:
            bit = higher & -higher
            j = bit.bit_length() - 1
            triangles += (adjacency[i] & adjacency[j]
                          & ~((1 << (j + 1)) - 1)).bit_count()
            higher ^= bit
    return 0, triangles - len(cols), adjacency


def _f_radius(adjacency):
    """Power-iteration expansion heuristic, not a certified bound."""
    from math import sin, sqrt
    nr = len(adjacency)
    neighbors = []
    for word in adjacency:
        row = []
        while word:
            bit = word & -word
            row.append(bit.bit_length() - 1)
            word ^= bit
        neighbors.append(row)
    assert len({len(row) for row in neighbors}) == 1
    result = 0.0
    for seed in (1, 3, 7):
        vector = [sin((i + 1) * seed) for i in range(nr)]
        mean = sum(vector) / nr
        vector = [v - mean for v in vector]
        norm = sqrt(sum(v * v for v in vector))
        vector = [v / norm for v in vector]
        for step in range(24):
            image = [sum(vector[j] for j in row) for row in neighbors]
            mean = sum(image) / nr
            image = [v - mean for v in image]
            norm = sqrt(sum(v * v for v in image))
            if norm < 1e-12:
                break
            vector = [v / norm for v in image]
        result = max(result, norm)
    return round(result, 8)


def _f_exposure(cols, q):
    """Anchored six-word collisions and small triple boundaries.

    All columns have odd weight three, excluding odd-weight kernels.
    After weight<=4 exclusion, distinct equal-syndrome triples are
    disjoint. Counts below are anchored exposure, not an exact spectrum.
    """
    size = len(cols)
    anchors = tuple(range(0, size, q))
    aset = set(anchors)
    anchored = {}
    boundary = 0
    for a in anchors:
        for b in range(size):
            if b == a or (b in aset and b < a):
                continue
            ab = cols[a] ^ cols[b]
            for c in range(b + 1, size):
                if c == a or (c in aset and c < a):
                    continue
                syndrome = ab ^ cols[c]
                anchored[syndrome] = anchored.get(syndrome, 0) + 1
                weight = syndrome.bit_count()
                if weight <= 3:
                    boundary += (4 - weight) ** 2
    six = -sum(anchored.values())
    get = anchored.get
    for a in range(size - 2):
        x = cols[a]
        for b in range(a + 1, size - 1):
            ab = x ^ cols[b]
            for c in range(b + 1, size):
                six += get(ab ^ cols[c], 0)
    assert six >= 0
    return six, boundary


def _a8_probe(checks, stabilizers, n, seed, shots=64):
    """Joint BP surrogate, not a measurement of ASC circuit memory.

    Hook pairs come from OPPOSITE stabilizer checks, not from the
    checks detecting this error component. Their ordering is unknown.
    Measurement histories and the evaluator's relay cascade are omitted.
    """
    import numpy as np

    def neighbors(rows):
        result = []
        for row in rows:
            current = []
            while row:
                bit = row & -row
                current.append(bit.bit_length() - 1)
                row ^= bit
            result.append(current)
        return result

    ns = neighbors(checks)
    hooks = neighbors(stabilizers)
    weight = len(ns[0])
    assert all(len(row) == weight for row in ns)
    incidence = np.asarray(ns, dtype=np.int32)
    flat = incidence.ravel()
    nr, edges = len(ns), len(flat)
    degrees = np.bincount(flat, minlength=n)
    opposite = np.bincount(
        np.asarray(hooks, dtype=np.int32).ravel(), minlength=n)
    total_degree = degrees + opposite
    variable_edges = np.full(
        (n, int(degrees.max())), edges, dtype=np.int32)
    used = np.zeros(n, dtype=np.int32)
    for edge, variable in enumerate(flat):
        variable_edges[variable, used[variable]] = edge
        used[variable] += 1

    channel = np.random.default_rng(seed)
    levels = np.asarray((0.008, 0.016, 0.028))
    rates = levels[np.arange(shots) % 3, None] * total_degree[None, :] / 6
    error = channel.random((shots, n)) < rates
    hook_rows = channel.integers(len(hooks), size=shots)
    active = channel.random(shots) < 0.30
    for shot in np.flatnonzero(active):
        row = hooks[int(hook_rows[shot])]
        a = int(channel.integers(len(row)))
        b = int(channel.integers(len(row) - 1))
        b += b >= a
        error[shot, row[a]] ^= True
        error[shot, row[b]] ^= True

    prior_rates = np.minimum(
        rates + 0.6 * opposite[None, :] / sum(opposite), 0.30)
    prior = np.log((1 - prior_rates) / prior_rates)
    target = (
        error[:, flat].reshape(shots, nr, weight).sum(axis=2) & 1
    ).astype(bool)
    target_sign = np.where(target, -1.0, 1.0)
    messages = prior[:, flat].copy()

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

    failed = iterations = 0
    for iteration in range(1, 33):
        count = len(error)
        shaped = messages.reshape(count, nr, weight)
        signs = np.where(shaped < 0, -1.0, 1.0)
        logs = np.log(np.clip(
            np.tanh(np.abs(shaped) / 2), 1e-12, 1 - 1e-7))
        product = np.minimum(np.exp(
            logs.sum(axis=2, keepdims=True) - logs), 1 - 1e-7)
        magnitude = np.log1p(product) - np.log1p(-product)
        outgoing = (
            magnitude * signs * signs.prod(axis=2, keepdims=True)
            * target_sign[:, :, None]
        ).reshape(count, edges)

        padded = np.zeros((count, edges + 1), dtype=np.float64)
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
            iterations += iteration

        keep = ~settled
        if not np.any(keep):
            return failed, 0, iterations
        messages = np.clip(
            0.35 * messages
            + 0.65 * (posterior[:, flat] - outgoing), -24, 24)[keep]
        prior = prior[keep]
        target = target[keep]
        target_sign = target_sign[keep]
        error = error[keep]

    unresolved = len(error)
    return failed + unresolved, unresolved, iterations + 33 * unresolved


def NEW_CTX(rng, G, idx):
    """Order-seven light contracts and noisy-syndrome holdout selection."""
    from itertools import combinations, combinations_with_replacement, product
    from collections import Counter
    import hashlib

    q = G.q
    assert q == 56 and A_SHAPE == B_SHAPE == (1, 3)
    width, mask = 3*q, (1 << q)-1
    supports = [[2, 3, 3]]
    assert _no_dead_lines(supports, supports)
    assert _grid_weights(supports, supports) == 11
    assert 8+3 <= min(12, W_MAX)

    def normalized(values):
        return min(tuple(sorted((v-origin) % 7 for v in values))
                   for origin in values)

    # Cell slope translations and component-column exchange are
    # conservatively quotiented. Fixing the binomial slope to two
    # anchors the Frobenius orientation; do not independently reflect
    # or multiply the two trinomial briefs.
    types = sorted({normalized(values)
                    for values in combinations(range(7), 3)})
    light = list(combinations_with_replacement(types, 2))
    contracts = list(combinations_with_replacement(range(len(light)), 2))
    assert len(contracts) == 120
    if not 0 <= idx < len(contracts):
        raise ValueError("light AGL8 quotient contract book exhausted")
    ia, ib = contracts[(17*idx) % len(contracts)]
    brief_a, brief_b = light[ia], light[ib]

    # Exact linear composition of helper-derived singleton blocks.
    probe, zero = [[[g] for g in G.elements]], [[[]]]
    ax, _, _ = lifted_matrices(G, probe, zero)
    _, bz, _ = lifted_matrices(G, zero, probe)
    actions = {"A": {}, "B": {}}
    for t, g in enumerate(G.elements):
        actions["A"][G.idx(g)] = tuple(
            (row >> (t*q)) & mask for row in ax)
        actions["B"][G.idx(g)] = tuple(
            (row >> (t*q)) & mask for row in bz)

    slopes = [1]
    for _ in range(6):
        slopes.append(G.mul((slopes[-1], 0), (2, 0))[0])
    assert len(set(slopes)) == 7

    def component(grid, side):
        rows = [0]*q
        for j, cell in enumerate(grid[0]):
            for g in cell:
                for r, word in enumerate(actions[side][G.idx(g)]):
                    rows[r] ^= word << (j*q)
        return rows

    def draw(brief, voltage):
        cells = [[G.e, (2, 0)]]
        phases = ((1, voltage[0]), (voltage[1], voltage[2]))
        for triangle, (a, b) in zip(brief, phases):
            assert triangle[0] == 0
            u = (slopes[triangle[1]], a)
            v = (slopes[triangle[2]], b)
            if G.mul(u, v) == G.mul(v, u):
                return None
            cell = [G.e, u, v]
            assert len({G.idx(g) for g in cell}) == 3
            cells.append(cell)
        return [cells]

    def graph(cols):
        """Exact C4 and C6 for mixed degree-two/three components."""
        adjacency, pairs = [0]*q, set()
        stars = 0
        for column in cols:
            ns = []
            while column:
                bit = column & -column
                ns.append(bit.bit_length()-1)
                column ^= bit
            d = len(ns)
            stars += d*(d-1)*(d-2)//6
            for i, u in enumerate(ns):
                for v in ns[:i]:
                    key = (min(u, v), max(u, v))
                    if key in pairs:
                        return None
                    pairs.add(key)
                    adjacency[u] |= 1 << v
                    adjacency[v] |= 1 << u
        reached = frontier = 1
        while frontier:
            following = 0
            while frontier:
                bit = frontier & -frontier
                following |= adjacency[bit.bit_length()-1]
                frontier ^= bit
            frontier = following & ~reached
            reached |= frontier
        if reached != (1 << q)-1:
            return None
        triangles = 0
        for u in range(q):
            later = adjacency[u] & ~((1 << (u+1))-1)
            while later:
                bit = later & -later
                v = bit.bit_length()-1
                triangles += (adjacency[u] & adjacency[v]
                              & ~((1 << (v+1))-1)).bit_count()
                later ^= bit
        assert triangles >= stars
        return triangles-stars, adjacency

    def clean4(cols):
        singles = set(cols)
        if 0 in singles or len(singles) != len(cols):
            return False
        pairs = set()
        for i, x in enumerate(cols):
            for y in cols[:i]:
                value = x ^ y
                if value in singles or value in pairs:
                    return False
                pairs.add(value)
        return True

    def anchored(cols):
        """Anchored five-word collisions and triple boundaries.

        Counts are exposure statistics, not exact spectra. Regular
        component translations are transitive within each column block:
        after clean4, zero anchored exposure certifies no weight-five
        word anywhere. The same existence argument applies to six below.
        It does not assume every word has a free group orbit.
        """
        pair_values = {x ^ y for i, x in enumerate(cols)
                       for y in cols[:i]}
        anchors = (0, q, 2*q)
        aset = set(anchors)
        triples, five, boundary = {}, 0, 0
        loads = [0]*3
        for a in anchors:
            for b in range(width):
                if b == a or (b in aset and b < a):
                    continue
                ab = cols[a] ^ cols[b]
                for c in range(b+1, width):
                    if c == a or (c in aset and c < a):
                        continue
                    value = ab ^ cols[c]
                    triples[value] = triples.get(value, 0)+1
                    five += int(value in pair_values)
                    weight = value.bit_count()
                    if weight <= 4:
                        cost = (5-weight)**2
                        boundary += cost
                        loads[a//q] += cost
                        loads[b//q] += cost
                        loads[c//q] += cost
        return five, max(loads), boundary, triples

    def six_exposure(cols, triples):
        # Subtract each anchored triple's match with itself. Other
        # equal-syndrome triples are disjoint after clean4.
        six = -sum(triples.values())
        get = triples.get
        for a in range(width-2):
            for b in range(a+1, width-1):
                ab = cols[a] ^ cols[b]
                for c in range(b+1, width):
                    six += get(ab ^ cols[c], 0)
        assert six >= 0
        return six

    def fingerprint(rows, cols):
        """Rooted component invariant, not complete Tanner canonization."""
        adjacency = [[] for _ in range(q+width)]
        for c, value in enumerate(cols):
            while value:
                bit = value & -value
                r = bit.bit_length()-1
                adjacency[r].append(q+c)
                adjacency[q+c].append(r)
                value ^= bit
        colors = [0]*q + [c.bit_count() for c in cols]
        colors[0] = 20
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

    def select(side, brief):
        # Exhaust a finite voltage family instead of issuing repeated
        # random draws. Scaling fixes the first nonzero phase to one
        # while leaving all relative voltages in the family searchable.
        voltages = list(product(range(8), repeat=3))
        rng.shuffle(voltages)
        pool = []
        for trial, voltage in enumerate(voltages):
            grid = draw(brief, voltage)
            if grid is None:
                continue
            rows = component(grid, side)
            if gf2_rank(rows) != q:
                continue
            cols = _f_columns(rows, width)
            assert all(c.bit_count() == (2 if j < q else 3)
                       for j, c in enumerate(cols))
            result = graph(cols)
            if result is None or not clean4(cols):
                continue
            cycles, adjacency = result
            five, peak, boundary, triples = anchored(cols)
            pool.append((five, cycles, trial, grid, rows, cols,
                         adjacency, peak, boundary, triples))
            pool.sort(key=lambda item: item[:3])
            del pool[12:]
        if not pool:
            raise ValueError("no clean full-rank light AGL8 component")

        refined, classes = [], set()
        for five, cycles, trial, grid, rows, cols, adj, peak, bd, ts in pool:
            identity = fingerprint(rows, cols)
            if identity in classes:
                continue
            classes.add(identity)
            six = six_exposure(cols, ts)
            key = (five, six, peak, bd, _f_radius(adj), cycles)
            refined.append((key, identity, trial, grid, rows))
        refined.sort(key=lambda item: item[:3])
        return refined[:3]

    def noisy_probe(checks, stabilizers, n, seed, shots):
        """Two fault intervals, NOT ASC simulation or relay/LSD decoding.

        Equations are H*e1+m=s1 and H*(e1+e2)=s2. The intermediate
        observation is noisy; the terminal observation is perfect.
        Opposite-check proper suffixes use random, non-ASC gate orders.
        Valid residuals are classified modulo the real stabilizer span.
        """
        import numpy as np

        def sites(row):
            ns = []
            while row:
                bit = row & -row
                ns.append(bit.bit_length()-1)
                row ^= bit
            return ns

        m, variables = len(checks), 2*n+len(checks)
        augmented = [
            row | (1 << (2*n+r)) for r, row in enumerate(checks)
        ] + [row | (row << n) for row in checks]
        neighborhoods = [sites(row) for row in augmented]
        row_width = max(map(len, neighborhoods))
        incidence = np.full((2*m, row_width), variables, dtype=np.int32)
        for r, ns in enumerate(neighborhoods):
            incidence[r, :len(ns)] = ns
        flat = incidence.ravel()
        valid, edges = flat != variables, len(flat)
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
            c.bit_count() for c in _f_columns(checks, n)])
        opposite = np.asarray([
            c.bit_count() for c in _f_columns(stabilizers, n)])
        levels = np.asarray((0.35, 0.60, 0.85))[np.arange(shots) % 3]
        data_mu = 3*0.002*(2+sensed+opposite)*2/3
        measure_mu = np.asarray([
            3*0.002*(row.bit_count()+2)*2/3 for row in checks])
        data_p = 0.5*(1-np.exp(-2*levels[:, None]*data_mu[None, :]))
        measure_p = 0.5*(1-np.exp(
            -2*levels[:, None]*measure_mu[None, :]))
        channel = np.random.default_rng(seed)
        error = np.zeros((shots, variables+1), dtype=bool)
        error[:, :n] = channel.random((shots, n)) < data_p
        error[:, n:2*n] = channel.random((shots, n)) < data_p
        error[:, 2*n:variables] = (
            channel.random((shots, m)) < measure_p)

        for row in stabilizers:
            ns = sites(row)
            order = channel.permutation(ns)
            event_p = levels*3*0.002*len(ns)*2/3
            for interval in range(2):
                for shot in np.flatnonzero(channel.random(shots) < event_p):
                    cut = int(channel.integers(1, len(order)))
                    error[shot, interval*n+order[cut:]] ^= True

        # Hook marginals are a heuristic prior, not a fitted channel.
        marginal_mu = data_mu+3*0.002*opposite*0.35
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
            product_values = np.minimum(np.exp(
                logs.sum(axis=2, keepdims=True)-logs), 1-1e-8)
            outgoing = (
                (np.log1p(product_values)-np.log1p(-product_values))
                * signs * signs.prod(axis=2, keepdims=True)
                * target_sign[:, :, None]
            ).reshape(count, edges)
            outgoing[:, ~valid] = 0.0
            padded = np.zeros((count, edges+1))
            padded[:, :edges] = outgoing
            posterior = prior+padded[:, variable_edges].sum(axis=2)
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

    aa, bb = select("A", brief_a), select("B", brief_b)
    options, identities = [], set()
    for ka, fa, ta, A, ar in aa:
        for kb, fb, tb, B, br in bb:
            identity = tuple(sorted((fa, fb)))
            if identity in identities:
                continue
            identities.add(identity)
            geometry = tuple(v for a, b in zip(ka, kb)
                             for v in (max(a, b), a+b))
            options.append((geometry, identity, ta, tb, A, B, ar, br))
    options.sort(key=lambda item: item[:4])

    seed = 560007719+1009*idx
    training = []
    for number, entry in enumerate(options[:5]):
        geometry, identity, ta, tb, A, B, ar, br = entry
        hx, hz, n = lifted_matrices(G, A, B)
        k = n-gf2_rank(hx)-gf2_rank(hz)
        if n != 560 or k != 224:
            continue
        actual_a = [
            sum(((row >> (3*j*q)) & mask) << (j*q)
                for j in range(3)) for row in hx[:q]
        ]
        actual_b = [row & ((1 << width)-1) for row in hz[:q]]
        assert actual_a == ar and actual_b == br
        if _f_low4(hx, hz, n) or _f_low4(hz, hx, n):
            continue
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx+hz) == 11
        assert 6*sum(row.bit_count() for row in hx+hz) == 21504
        x = noisy_probe(hz, hx, n, seed, 48)
        z = noisy_probe(hx, hz, n, seed+1, 48)
        training.append((quality(x, z)+geometry+(number,),
                         number, entry, hx, hz, x, z))
    training.sort(key=lambda item: item[:2])

    finalists = []
    for _, number, entry, hx, hz, tx, tz in training[:2]:
        geometry, identity, ta, tb, A, B, ar, br = entry
        # Independent holdout streams lead the selection. Training
        # results and structural screens are only later tie-breakers.
        x = noisy_probe(hz, hx, 560, seed+1000003, 128)
        z = noisy_probe(hx, hz, 560, seed+1000004, 128)
        key = quality(x, z)+quality(tx, tz)+geometry+(number,)
        finalists.append((key, A, B, identity, x, z, tx, tz))
    if not finalists:
        raise ValueError("no complete light AGL8 code passed screening")
    finalists.sort(key=lambda item: item[0])
    key, A, B, identity, x, z, tx, tz = finalists[0]
    return {"idx": idx, "cells": {"A": A, "B": B}, "k": 224,
            "contract": (brief_a, brief_b),
            "component_pair_invariant": identity,
            "training_surrogate": (tx, tz),
            "holdout_surrogate": (x, z), "joint_screen": key}


def CELL(rng, G, side, i, j, ctx):
    """Return the already screened complete cell, without resampling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Screen d2s=4 alone implies reliable circuit memory; the parent's measured slots still had worst-basis failures of 120-144/250 shots, with broad sector-1 weight-four witnesses.
# REFUTED: Balanced dihedral trinomials with d2s=8 suffice for rare circuit failures; all five measured parent codes reached failed-shot caps, including slot 0 Z=125/1750, with flips spanning all 232 logical qubits.
# REFUTED: Four-cycle-free odd-order trinomials plus boundary/spectral selection guarantee rare memory failures: F57 fitness was 4.8676, all ten measured bases hit failed-shot caps, and slot9 X failed 103/3500 shots despite d2s=8, with flips across all 228 logical qubits.
# REFUTED: Full-rank distinct-slope AGL8 trinomials with boundary/expansion screens and perfect-syndrome opposite-check pair probes suffice for rare basis-balanced ASC failures: fitness was 4.8989, all ten measured bases hit failure caps, slot2 Z failed 101/5250 shots across all 224 logicals, and slot4 X failed 100/6000 versus Z 100/17750; this does not isolate the labels from the surrogate.
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
