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
CONCEPT = ("For Frobenius C19 semidirect C3 with action 7, order 57, shapes (1,3)x(1,3), balanced (2,3,3) components containing a noncommuting three-coset triangle should improve worse-basis ASC fitness above 4.8676 by reducing extraction load without admitting component kernels through four; measured circuit pL decides.")
GROUP = ("Frozen L1 group, order and encoding. This exact bed measured fitness 4.8676 and supplies full-rank k=228 codes. Related order-55 mixed-degree programs scored 5.0002-5.0213, motivating a controlled lower-load filling experiment without demonstrating a group advantage. Retain q=57 and n=570 byte-for-byte.")
SHAPES = ("Frozen equal (1,3) sides give 171-variable, 57-check components, n=570, kb=228 and guaranteed rate 0.400; the size factor ten permits q<=59. Require complete verified k=228, preserving zero k-kb. Parent weight-eight witnesses span sector-one Acol/Bcol blocks and failures reach all 228 logicals. Data and Acol/Brow check enrichment is mostly 1.0-1.1x, establishing no persistent isolated repair target.")
FILLING = ("Both sides use an order-19 binomial and two noncommuting trinomials, with counts (2,3,3): maximum CSS weight eleven and 21888 six-round CNOTs instead of twelve and 24624. Every component retains a repeated-coset triangle, and at least one contains a three-coset triangle. Indexed relative-step contracts quotient component-column permutations, A/B exchange and a conservative common quotient reflection; disjoint contract tranches provide fresh alternatives when a finite search misses a screen. Helper-derived component actions require rank 57, connectivity, no Tanner four-cycles and no kernels through four. Rooted Tanner fingerprints conservatively diversify local pools; exact five/six-word counts and triple-boundary loads shortlist them. Complete helper-built finalists require CSS, k=228 and no quantum logical through four, then receive common-stream six-round temporal BP probes with noisy measurements and opposite-check ancilla suffix faults. Independent holdout results lead final selection. These omit actual ASC ordering and relay/LSD and are not calibrated circuit fitness. Target collective flips, late settlement and slot9 X=103/3500 despite d2s=8.")
FAMILY_NAME = "r040_Frob57_13x13_long19_threecoset_temporalholdout_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 57

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C19 semidirect C3 requires q=57")
    # x^19=y^3=1, y*x*y^-1=x^7; 7^3=1 modulo 19.
    els = [(a, b) for b in range(3) for a in range(19)]

    def red(g):
        return (int(g[0]) % 19, int(g[1]) % 3)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + pow(7, b, 19) * c) % 19, (b + d) % 3)

    def inv(g):
        a, b = red(g)
        bi = (-b) % 3
        return ((-pow(7, bi, 19) * a) % 19, bi)

    def idx(g):
        a, b = red(g)
        return a + 19 * b

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
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


def NEW_CTX(rng, G, idx):
    """Mixed-degree architecture contracts and complete temporal selection."""
    from itertools import combinations_with_replacement
    from collections import Counter
    import hashlib

    q = G.q
    assert q == 57 and A_SHAPE == B_SHAPE == (1, 3)
    width, mask = 3*q, (1 << q)-1
    supports = [[2, 3, 3]]
    assert _no_dead_lines(supports, supports)
    assert _grid_weights(supports, supports) == 11
    assert 8+3 <= min(12, W_MAX)

    # R(d,h): repeated quotient coset, with rotation step d relative
    # to the binomial's step one. T: all three quotient cosets.
    # A component always has an R, so neither component is forced to
    # vanish on the nontrivial quotient characters.
    tags = [(0, 0)] + [(d, h) for d in range(2, 10) for h in (1, 2)]
    components = [
        (a, b) for a, b in combinations_with_replacement(tags, 2)
        if (a[0] or b[0]) and (not a[0] or a[0] != b[0])
    ]

    def reflected(brief):
        return tuple(sorted((d, 3-h) if d else (0, 0)
                            for d, h in brief))

    def canonical_contract(a, b):
        direct = tuple(sorted((a, b)))
        reverse = tuple(sorted((reflected(a), reflected(b))))
        # Conservative quotient reflection, not an asserted group
        # automorphism or complete Tanner-isomorphism classification.
        return min(direct, reverse)

    contracts = sorted({
        canonical_contract(a, b)
        for a, b in combinations_with_replacement(components, 2)
        if any(d == 0 for d, h in a+b)
    })
    budget = max(20, N_SLOTS)
    assert 0 <= idx < budget and len(contracts) >= 12*budget
    begin = idx*len(contracts)//budget
    end = (idx+1)*len(contracts)//budget

    # Two verified bulk probes supply every singleton action. These
    # are not submitted codes, and their widths need not equal n.
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

    def draw(brief):
        cells = [[G.e, (1, 0)]]
        for position, (d, h) in enumerate(brief):
            if d:
                u = ((d if rng.randrange(2) else -d) % 19, 0)
                v = (0 if position == 0 else rng.randrange(19), h)
            else:
                u = (0 if position == 0 else rng.randrange(19), 1)
                v = (rng.randrange(19), 2)
            if G.mul(u, v) == G.mul(v, u):
                return None
            cell = [G.e, u, v]
            assert len({G.idx(g) for g in cell}) == 3
            cells.append(cell)
        return [cells]

    def component(grid, side):
        rows = [0]*q
        for j, cell in enumerate(grid[0]):
            for g in cell:
                for r, word in enumerate(actions[side][G.idx(g)]):
                    rows[r] ^= word << (j*q)
        return rows

    def graph(columns):
        """Exact C4/C6 and connectivity for mixed column degrees."""
        adjacency = [0]*q
        pairs = set()
        stars = 0
        for word in columns:
            vertices = []
            while word:
                bit = word & -word
                vertices.append(bit.bit_length()-1)
                word ^= bit
            degree = len(vertices)
            stars += degree*(degree-1)*(degree-2)//6
            for i, u in enumerate(vertices):
                for v in vertices[:i]:
                    edge = (min(u, v), max(u, v))
                    if edge in pairs:
                        return None
                    pairs.add(edge)
                    adjacency[u] |= 1 << v
                    adjacency[v] |= 1 << u
        reached, todo = {0}, [0]
        while todo:
            word = adjacency[todo.pop()]
            while word:
                bit = word & -word
                v = bit.bit_length()-1
                word ^= bit
                if v not in reached:
                    reached.add(v)
                    todo.append(v)
        if len(reached) != q:
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
        return triangles-stars

    def fingerprint(columns):
        """Rooted component Tanner invariant, not full canonization."""
        adjacency = [[] for _ in range(q+width)]
        for c, word in enumerate(columns):
            while word:
                bit = word & -word
                r = bit.bit_length()-1
                adjacency[r].append(q+c)
                adjacency[q+c].append(r)
                word ^= bit
        colors = [0]*q + [c.bit_count() for c in columns]
        colors[0] = 20
        digest = hashlib.sha256()
        for _ in range(10):
            signatures = [
                (colors[v], tuple(sorted(colors[w] for w in ns)))
                for v, ns in enumerate(adjacency)
            ]
            digest.update(repr(sorted(Counter(signatures).items())).encode())
            palette = {s: i for i, s in enumerate(sorted(set(signatures)))}
            colors = [palette[s] for s in signatures]
        return digest.hexdigest()

    def census(columns):
        """Exact weight-five/six counts after exclusion through four."""
        pairs, triples = {}, {}
        for i, x in enumerate(columns):
            for y in columns[:i]:
                value = x ^ y
                pairs[value] = pairs.get(value, 0)+1
        five = six = boundary = 0
        loads = [0]*3
        for a in range(width-2):
            for b in range(a+1, width-1):
                pair = columns[a] ^ columns[b]
                for c in range(b+1, width):
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
        # <=4 exclusion makes matching supports disjoint. Each
        # five-set and six-set contributes ten relevant partitions.
        assert five % 10 == 0 and six % 10 == 0
        return five//10, six//10, max(loads), boundary

    def select(side, brief):
        pool, seen = [], set()
        for trial in range(1800):
            grid = draw(brief)
            if grid is None:
                continue
            signature = tuple(sorted(
                tuple(sorted(G.idx(g) for g in cell))
                for cell in grid[0]))
            if signature in seen:
                continue
            seen.add(signature)
            rows = component(grid, side)
            columns = _f_columns(rows, width)
            cycles = graph(columns)
            if cycles is None or gf2_rank(rows) != q:
                continue
            if _f_low4(rows, [], width):
                continue
            pool.append((cycles, trial, grid, rows, columns))
            pool.sort(key=lambda item: item[:2])
            del pool[6:]
            if len(seen) >= 240 and len(pool) == 6:
                break
        refined, identities = [], set()
        for cycles, trial, grid, rows, columns in pool:
            identity = fingerprint(columns)
            if identity in identities:
                continue
            identities.add(identity)
            key = census(columns)+(cycles,)
            refined.append((key, trial, grid, rows, identity))
        refined.sort(key=lambda item: item[:2])
        return refined[:2]

    def temporal_probe(checks, stabilizers, n, seed, shots):
        """Six-round BP surrogate, NOT ASC or relay/LSD measurement.

        Jointly decodes data increments and noisy measurements.
        Opposite-check suffix faults use random gate orders rather
        than the unknown evaluator schedule. Terminal syndrome is
        perfect; settled residuals use the actual stabilizer span.
        """
        import numpy as np

        def neighborhoods(rows):
            result = []
            for row in rows:
                ns = []
                while row:
                    bit = row & -row
                    ns.append(bit.bit_length()-1)
                    row ^= bit
                result.append(ns)
            return result

        neighbors = neighborhoods(checks)
        hooks = neighborhoods(stabilizers)
        rounds, m = 6, len(checks)
        variables = rounds*(n+m)
        dummy = variables
        row_width = max(map(len, neighbors))+2
        detectors = (rounds+1)*m
        incidence = np.full(
            (detectors, row_width), dummy, dtype=np.int32)
        for t in range(rounds+1):
            for r, ns in enumerate(neighbors):
                position = 0
                if t < rounds:
                    for c in ns:
                        incidence[t*m+r, position] = t*n+c
                        position += 1
                    incidence[t*m+r, position] = rounds*n+t*m+r
                    position += 1
                if t:
                    incidence[t*m+r, position] = rounds*n+(t-1)*m+r

        flat = incidence.ravel()
        valid = flat != dummy
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

        spatial = np.asarray(
            [c.bit_count() for c in _f_columns(checks, n)])
        opposite = np.asarray(
            [c.bit_count() for c in _f_columns(stabilizers, n)])
        data_rate = 0.002*(2+spatial+opposite)*2/3
        measurement_rate = np.asarray(
            [0.002*(2+len(ns))*2/3 for ns in neighbors])
        levels = np.asarray((1.0, 1.3, 1.6))[np.arange(shots) % 3]
        base = np.concatenate((
            np.tile(data_rate, rounds),
            np.tile(measurement_rate, rounds)))
        rates = np.minimum(levels[:, None]*base[None, :], 0.25)
        channel = np.random.default_rng(seed)
        error = np.zeros((shots, variables+1), dtype=bool)
        error[:, :variables] = channel.random(
            (shots, variables)) < rates

        # Opportunity count scales with actual opposite-check weights.
        # These suffixes are possible ancilla spreads, not certified
        # ASC hooks. The decoder is not told their sampled locations.
        hook_rate = np.asarray(
            [0.002*len(ns)*2/3 for ns in hooks])
        hook_order = [channel.permutation(ns) for ns in hooks]
        for t in range(rounds):
            active = channel.random((shots, len(hooks))) < (
                levels[:, None]*hook_rate[None, :])
            for shot, r in np.argwhere(active):
                order = hook_order[int(r)]
                cut = int(channel.integers(1, len(order)))
                error[int(shot), t*n+order[cut:]] ^= True

        target = (
            error[:, flat].reshape(
                shots, detectors, row_width).sum(axis=2) & 1
        ).astype(bool)
        target_sign = np.where(target, -1.0, 1.0)
        decoding_base = np.concatenate((
            np.tile(data_rate+0.002*opposite*0.35, rounds),
            np.tile(measurement_rate, rounds)))
        decoding_rates = np.minimum(
            levels[:, None]*decoding_base[None, :], 0.30)
        prior = np.full((shots, variables+1), 30.0)
        prior[:, :variables] = np.log(
            (1-decoding_rates)/decoding_rates)
        messages = prior[:, flat].copy()
        messages[:, ~valid] = 30.0
        true_final = np.logical_xor.reduce(
            error[:, :rounds*n].reshape(shots, rounds, n), axis=1)

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

        failed = late = work = 0
        for iteration in range(1, 33):
            count = len(prior)
            shaped = messages.reshape(count, detectors, row_width)
            signs = np.where(shaped < 0, -1.0, 1.0)
            logs = np.log(np.clip(
                np.tanh(np.abs(shaped)/2), 1e-12, 1-1e-8))
            product = np.minimum(np.exp(
                logs.sum(axis=2, keepdims=True)-logs), 1-1e-8)
            outgoing = (
                (np.log1p(product)-np.log1p(-product))
                * signs * signs.prod(axis=2, keepdims=True)
                * target_sign[:, :, None]
            ).reshape(count, edges)
            outgoing[:, ~valid] = 0.0
            padded = np.zeros((count, edges+1), dtype=np.float64)
            padded[:, :edges] = outgoing
            posterior = prior+padded[:, variable_edges].sum(axis=2)
            posterior[:, dummy] = 30.0
            hard = posterior < 0
            syndrome = (
                hard[:, flat].reshape(
                    count, detectors, row_width).sum(axis=2) & 1
            ).astype(bool)
            settled = np.all(syndrome == target, axis=1)
            final = np.logical_xor.reduce(
                hard[:, :rounds*n].reshape(count, rounds, n), axis=1)
            for local in np.flatnonzero(settled):
                word = 0
                for c in np.flatnonzero(final[local] ^ true_final[local]):
                    word |= 1 << int(c)
                failed += int(nontrivial(word))
                late += int(iteration > 12)
                work += iteration
            keep = ~settled
            if not np.any(keep):
                return failed, 0, late, work
            damping = 0.30 if iteration <= 12 else 0.50
            messages = np.clip(
                damping*messages
                + (1-damping)*(posterior[:, flat]-outgoing),
                -30, 30)[keep]
            messages[:, ~valid] = 30.0
            prior = prior[keep]
            target = target[keep]
            target_sign = target_sign[keep]
            true_final = true_final[keep]
        unresolved = len(prior)
        return (failed+unresolved, unresolved,
                late+unresolved, work+33*unresolved)

    def quality(x, z):
        return (max(x[0], z[0]), x[0]+z[0],
                max(x[1], z[1]), x[1]+z[1],
                max(x[2], z[2]), x[2]+z[2], x[3]+z[3])

    for attempt, position in enumerate(range(begin, min(end, begin+12))):
        brief_a, brief_b = contracts[position]
        aa, bb = select("A", brief_a), select("B", brief_b)
        options, identities = [], set()
        for ka, ta, A, ar, fa in aa:
            for kb, tb, B, br, fb in bb:
                identity = tuple(sorted((fa, fb)))
                if identity in identities:
                    continue
                identities.add(identity)
                joint = tuple(v for a, b in zip(ka, kb)
                              for v in (max(a, b), a+b))
                options.append((joint, ta, tb, A, B, ar, br, identity))
        options.sort(key=lambda item: item[:3])

        seed = 570019317+1009*idx+100003*attempt
        coarse = []
        for number, option in enumerate(options):
            joint, ta, tb, A, B, ar, br, identity = option
            hx, hz, n = lifted_matrices(G, A, B)
            k = n-gf2_rank(hx)-gf2_rank(hz)
            if n != 570 or k != 228:
                continue
            actual_a = [
                sum(((row >> (3*j*q)) & mask) << (j*q)
                    for j in range(3))
                for row in hx[:q]]
            actual_b = [row & ((1 << width)-1) for row in hz[:q]]
            assert actual_a == ar and actual_b == br
            if _f_low4(hx, hz, n) or _f_low4(hz, hx, n):
                continue
            assert css_ok(hx, hz)
            assert max(row.bit_count() for row in hx+hz) == 11
            assert 6*sum(row.bit_count() for row in hx+hz) == 21888
            x = temporal_probe(hz, hx, n, seed, 32)
            z = temporal_probe(hx, hz, n, seed+1, 32)
            coarse.append((quality(x, z)+joint+(number,),
                           number, A, B, hx, hz, joint, identity, x, z))
        coarse.sort(key=lambda item: item[:2])

        finalists = []
        for _, number, A, B, hx, hz, joint, identity, cx, cz in coarse[:2]:
            # Independent streams lead the decision; training samples
            # are not pooled into the holdout failure counts.
            x = temporal_probe(hz, hx, 570, seed+1000003, 96)
            z = temporal_probe(hx, hz, 570, seed+1000004, 96)
            score = quality(x, z)+quality(cx, cz)+joint+(number,)
            finalists.append((score, A, B, identity, x, z, cx, cz))
        if finalists:
            finalists.sort(key=lambda item: item[0])
            score, A, B, identity, x, z, cx, cz = finalists[0]
            return {"idx": idx, "cells": {"A": A, "B": B}, "k": 228,
                    "contract": (position, brief_a, brief_b),
                    "component_identity": identity,
                    "training_surrogate": (cx, cz),
                    "holdout_surrogate": (x, z), "screen": score}
    raise ValueError("no long19 three-coset contract passed verified screens")


def CELL(rng, G, side, i, j, ctx):
    """Emit the complete holdout-selected cell without further draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Screen d2s=4 alone implies reliable circuit memory; the parent's measured slots still had worst-basis failures of 120-144/250 shots, with broad sector-1 weight-four witnesses.
# REFUTED: Balanced dihedral trinomials with d2s=8 suffice for rare circuit failures; all five measured parent codes reached failed-shot caps, including slot 0 Z=125/1750, with flips spanning all 232 logical qubits.
# REFUTED: Independently selected full-rank four-cycle-free F57 trinomials with short-boundary and spectral screening suffice for rare, balanced ASC memory: fitness was 4.8676, all ten measured bases reached failed-shot caps, and slot9 X failed 103/3500 shots despite d2s=8, with flips across all 228 logicals.
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
