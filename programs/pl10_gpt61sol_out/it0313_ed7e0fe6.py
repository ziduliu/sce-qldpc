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
CONCEPT = ("For square-slope affine C11:C5 over F11, order 55, shapes (1,3)x(1,3), balanced (2,3,3) components with mixed positive/negative quotient orientations and opposite-check temporal hook screening should improve worse-basis ASC fitness above the shown balanced-binomial result 5.0213; circuit measurement decides.")
GROUP = ("Frozen L1 group, order and encoding. The present all-trinomial program measured 4.8408 on this order-55 bed, while the shown balanced long-binomial program reached 5.0213. These readings motivate another controlled filling experiment, not a claim of group advantage. The fixed lift gives n=550.")
SHAPES = ("Frozen equal (1,3) sides give 165-variable, 55-check components, n=550, kb=220 and guaranteed rate 0.400; the size factor ten permits q<=59. Require complete verified k=220, retaining zero k-kb. Parent weight-eight witnesses occupy sector-one Acol/Bcol blocks broadly and failures reach all 220 logicals. Data and Acol/Brow check enrichment is mostly 1.0-1.1x, establishing no persistent isolated repair target.")
FILLING = ("Both components use counts (2,3,3), reducing maximum CSS weight from twelve to eleven and six-round CNOTs from 23760 to 21120. Indexed contracts combine positive and negative repeated-slope quotient orientations, with optional distinct-slope triangles; a conservative common-reflection and component-exchange quotient removes obvious duplicate contracts, not arbitrary Tanner equivalences. Fresh helper-derived component searches require rank 55, connectivity, no four-cycles and no kernels through four; exact five/six-word censuses and triple boundaries shortlist alternatives. Complete helper-built finalists require CSS, k=220 and no quantum logical through four. Six-round BP probes include noisy measurements, degree-weighted independent data increments and randomly ordered ancilla suffix faults from opposite stabilizer checks. Independent holdout streams choose the final filling. These omit actual ASC ordering and relay/LSD and are not calibrated circuit fitness. Target collective flips, late settlement and the parent's slot1 X=100/4250 versus Z=101/11000 despite d2s=8.")
FAMILY_NAME = "r040_AffSquare11_q55_13x13_mixedorientation_suffix_temporal_v3"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 55

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("square-slope affine F11 requires order 55")
    # (a,t) encodes [[a,t],[0,1]]; square slopes form a subgroup.
    slopes = tuple(sorted({x*x % 11 for x in range(1, 11)}))
    positions = {a: i for i, a in enumerate(slopes)}
    els = [(a, t) for a in slopes for t in range(11)]

    def red(g):
        return (int(g[0]) % 11, int(g[1]) % 11)

    def mul(g, h):
        a, t = red(g)
        c, u = red(h)
        return (a*c % 11, (t + a*u) % 11)

    def inv(g):
        a, t = red(g)
        ai = pow(a, 9, 11)
        return (ai, -ai*t % 11)

    def idx(g):
        a, t = red(g)
        return 11*positions[a] + t

    return dict(elements=els, ident=(1, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


def _af_columns(rows, width):
    columns = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return columns


def _af_rows(columns, checks):
    rows = [0] * checks
    for c, word in enumerate(columns):
        while word:
            bit = word & -word
            rows[bit.bit_length() - 1] |= 1 << c
            word ^= bit
    return rows


def _af_clean4(columns):
    """Exact component exclusion through four; all columns have odd degree."""
    if 0 in columns or len(set(columns)) != len(columns):
        return False
    seen = set()
    for i, x in enumerate(columns):
        for j in range(i):
            value = x ^ columns[j]
            if value in seen:
                return False
            seen.add(value)
    # Odd numbers of odd-weight columns cannot sum to zero.
    return True


def _af_graph(columns, checks):
    """Exact C4; exact Tanner C6 after C4 exclusion."""
    adjacency = [0] * checks
    pairs = {}
    four = 0
    for word in columns:
        vertices = []
        while word:
            bit = word & -word
            vertices.append(bit.bit_length() - 1)
            word ^= bit
        assert len(vertices) == 3
        for i, u in enumerate(vertices):
            for v in vertices[:i]:
                key = (min(u, v), max(u, v))
                old = pairs.get(key, 0)
                four += old
                pairs[key] = old + 1
                adjacency[u] |= 1 << v
                adjacency[v] |= 1 << u
    if four:
        return four, None, adjacency
    triangles = 0
    for u in range(checks):
        later = adjacency[u] & ~((1 << (u + 1)) - 1)
        while later:
            bit = later & -later
            v = bit.bit_length() - 1
            triangles += (adjacency[u] & adjacency[v]
                          & ~((1 << (v + 1)) - 1)).bit_count()
            later ^= bit
    six = triangles - len(columns)
    assert six >= 0
    return 0, six, adjacency


def _af_radius(adjacency):
    """Approximate nontrivial spectral radius, not a certified bound."""
    from math import sin, sqrt
    checks = len(adjacency)
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
    for seed in (1, 5):
        vector = [sin(seed * (i + 1)) for i in range(checks)]
        mean = sum(vector) / checks
        vector = [v - mean for v in vector]
        norm = sqrt(sum(v*v for v in vector))
        vector = [v / norm for v in vector]
        for step in range(24):
            image = [sum(vector[j] for j in row) for row in neighbors]
            mean = sum(image) / checks
            image = [v - mean for v in image]
            norm = sqrt(sum(v*v for v in image))
            if norm < 1e-12:
                break
            vector = [v / norm for v in image]
        result = max(result, norm)
    return round(result, 8)


def _af_six_boundary(columns):
    """Exact six-word count following <=4 exclusion; boundary surrogates.

    Distinct triples with equal syndrome are disjoint. Every six-set
    has ten unordered triple partitions. No group-orbit shortcut is
    used, so stabilizers of supports cannot corrupt the count.
    """
    histogram = {}
    collisions = one = three = 0
    size = len(columns)
    for a in range(size - 2):
        x = columns[a]
        for b in range(a + 1, size - 1):
            pair = x ^ columns[b]
            for c in range(b + 1, size):
                value = pair ^ columns[c]
                old = histogram.get(value, 0)
                collisions += old
                histogram[value] = old + 1
                weight = value.bit_count()
                one += int(weight == 1)
                three += int(weight == 3)
    assert collisions % 10 == 0
    return collisions // 10, one, three


def _af_logical4(checks, stabilizers, width):
    """Exact existence of a nontrivial quantum logical through weight four."""
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

    columns = _af_columns(checks, width)
    seen = {0: 0}
    for i, value in enumerate(columns):
        word = 1 << i
        if value in seen:
            if nontrivial(word ^ seen[value]):
                return True
        else:
            seen[value] = word
    for i, x in enumerate(columns):
        for j in range(i):
            value = x ^ columns[j]
            word = (1 << i) | (1 << j)
            if value in seen:
                if nontrivial(word ^ seen[value]):
                    return True
            else:
                seen[value] = word
    return False


def NEW_CTX(rng, G, idx):
    """Mixed-orientation light components and joint temporal holdout selection."""
    from itertools import combinations_with_replacement

    q = G.q
    assert q == 55 and A_SHAPE == B_SHAPE == (1, 3)
    counts = [[2, 3, 3]]
    assert _grid_weights(counts, counts) == 11
    assert _no_dead_lines(counts, counts)
    assert 8 + 3 <= min(12, W_MAX)
    mask, width = (1 << q) - 1, 3*q

    # R1,R2,R3,R4 have repeated quotient slope and exponent 1..4.
    # T1,T2 are distinct-slope triangles (0,1,2)/(0,1,3).
    # Require both signs somewhere in the complete contract. Thus the
    # contract is not obtained just by reflecting an all-positive brief.
    light = list(combinations_with_replacement(range(6), 2))
    reflected = (3, 2, 1, 0, 4, 5)

    def contract_key(a, b):
        direct = tuple(sorted((tuple(a), tuple(b))))
        reverse = tuple(sorted((
            tuple(sorted(reflected[t] for t in a)),
            tuple(sorted(reflected[t] for t in b)),
        )))
        # Reflection is a conservative book quotient, not a claim that
        # it is an automorphism of the fixed affine group.
        return min(direct, reverse)

    contracts = sorted({
        contract_key(a, b)
        for a, b in combinations_with_replacement(light, 2)
        if any(t in (0, 1) for t in a+b)
        and any(t in (2, 3) for t in a+b)
    })
    assert len(contracts) >= 60
    if idx >= len(contracts):
        raise ValueError("mixed-orientation contract book exhausted")
    brief_a, brief_b = contracts[(17*idx) % len(contracts)]

    # All singleton actions come from the verified construction.
    # The wide zero-partner probes are not submitted codes.
    probe = [[[g] for g in G.elements]]
    zero = [[[]]]
    ax, _, _ = lifted_matrices(G, probe, zero)
    _, bz, _ = lifted_matrices(G, zero, probe)
    actions = {"A": {}, "B": {}}
    for t, g in enumerate(G.elements):
        actions["A"][G.idx(g)] = _af_columns(
            [(row >> (t*q)) & mask for row in ax], q)
        actions["B"][G.idx(g)] = _af_columns(
            [(row >> (t*q)) & mask for row in bz], q)

    def draw(brief):
        cells = [[G.e, (1, 1)]]
        for position, kind in enumerate(brief):
            if kind < 4:
                displacement = rng.randrange(1, 11)
                phase = 0 if position == 0 else rng.randrange(11)
                u = (1, displacement)
                v = (pow(3, kind+1, 11), phase)
            else:
                exponent = 2 if kind == 4 else 3
                phase = 0 if position == 0 else rng.randrange(11)
                u = (3, phase)
                v = (pow(3, exponent, 11), rng.randrange(11))
            if G.mul(u, v) == G.mul(v, u):
                return None
            cell = [G.e, u, v]
            assert len({G.idx(g) for g in cell}) == 3
            cells.append(cell)
        return [cells]

    def component(grid, side):
        columns = []
        for cell in grid[0]:
            terms = [actions[side][G.idx(g)] for g in cell]
            for c in range(q):
                word = 0
                for term in terms:
                    word ^= term[c]
                columns.append(word)
        return columns

    def clean4(columns):
        """Exact component exclusion for mixed even/odd column degrees."""
        singles = set(columns)
        if 0 in singles or len(singles) != len(columns):
            return False
        seen = set()
        for i, x in enumerate(columns):
            for y in columns[:i]:
                value = x ^ y
                if value in singles or value in seen:
                    return False
                seen.add(value)
        return True

    def graph(columns):
        """Exact C4 and, after its exclusion, exact Tanner C6."""
        adjacency = [0]*q
        pairs = {}
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
                    key = (min(u, v), max(u, v))
                    if key in pairs:
                        return None
                    pairs[key] = 1
                    adjacency[u] |= 1 << v
                    adjacency[v] |= 1 << u

        reached, todo = {0}, [0]
        while todo:
            bits = adjacency[todo.pop()]
            while bits:
                bit = bits & -bits
                v = bit.bit_length()-1
                bits ^= bit
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
        six = triangles-stars
        assert six >= 0
        return six, adjacency

    def census(columns):
        """Exact weight-five/six census after kernels <=4 are excluded."""
        pairs = {}
        for i, x in enumerate(columns):
            for y in columns[:i]:
                value = x ^ y
                pairs[value] = pairs.get(value, 0)+1
        triples = {}
        five = six = boundary = 0
        loads = [0]*3
        for a in range(len(columns)-2):
            for b in range(a+1, len(columns)-1):
                pair = columns[a] ^ columns[b]
                for c in range(b+1, len(columns)):
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
        # Matching supports are disjoint after the <=4 exclusion.
        # Each five-set and six-set has ten relevant partitions.
        assert five % 10 == 0 and six % 10 == 0
        return five//10, six//10, max(loads), boundary

    def select(side, brief):
        pool, seen = [], set()
        accepted = 0
        for trial in range(2400):
            grid = draw(brief)
            if grid is None:
                continue
            identity = tuple(sorted(
                tuple(sorted(G.idx(g) for g in cell))
                for cell in grid[0]))
            if identity in seen:
                continue
            seen.add(identity)
            columns = component(grid, side)
            result = graph(columns)
            if result is None:
                continue
            cycles, adjacency = result
            rows = _af_rows(columns, q)
            if gf2_rank(rows) != q or not clean4(columns):
                continue
            accepted += 1
            pool.append((cycles, trial, grid, rows, columns, adjacency))
            pool.sort(key=lambda item: item[:2])
            del pool[6:]
            if accepted >= 64 and trial >= 128:
                break
        if not pool:
            raise ValueError("no clean mixed-orientation affine component")

        refined = []
        for cycles, trial, grid, rows, columns, adjacency in pool:
            key = census(columns)+(_af_radius(adjacency), cycles)
            refined.append((key, trial, grid, rows))
        refined.sort(key=lambda item: item[:2])
        return refined[:3]

    def temporal_probe(checks, stabilizers, n, seed, shots):
        """Six-round BP surrogate, NOT ASC or relay/LSD measurement.

        Independent data increments and noisy measurements are decoded
        jointly. Additional ancilla suffix faults use OPPOSITE checks.
        Random gate orders average unknown hook orientations; they are
        not the evaluator's ASC schedule. Terminal syndrome is perfect.
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
        detector_count = (rounds+1)*m
        incidence = np.full(
            (detector_count, row_width), dummy, dtype=np.int32)
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
            [c.bit_count() for c in _af_columns(checks, n)])
        opposite = np.asarray(
            [c.bit_count() for c in _af_columns(stabilizers, n)])
        data_rate = 0.002*(2+spatial+opposite)*2/3
        measurement_rate = np.asarray(
            [0.002*(2+len(ns))*2/3 for ns in neighbors])
        base = np.concatenate((
            np.tile(data_rate, rounds),
            np.tile(measurement_rate, rounds),
        ))
        levels = np.asarray((0.8, 1.0, 1.2))[np.arange(shots) % 3]
        rates = np.minimum(levels[:, None]*base[None, :], 0.25)
        channel = np.random.default_rng(seed)
        error = np.zeros((shots, variables+1), dtype=bool)
        error[:, :variables] = channel.random(
            (shots, variables)) < rates

        # Unlike one arbitrary pair per shot, fault opportunities scale
        # with the number and weights of opposite ancilla checks.
        # Suffixes longer than half a check remain legitimate errors:
        # stabilizer-span classification handles their complementary
        # representatives without manually altering the simulated fault.
        hook_rates = np.asarray(
            [0.002*len(ns)*2/3 for ns in hooks])
        hook_orders = [channel.permutation(ns) for ns in hooks]
        for t in range(rounds):
            active = channel.random((shots, len(hooks))) < (
                levels[:, None]*hook_rates[None, :])
            for shot, r in np.argwhere(active):
                order = hook_orders[int(r)]
                cut = int(channel.integers(1, len(order)))
                error[int(shot), t*n+order[cut:]] ^= True

        target = (
            error[:, flat].reshape(
                shots, detector_count, row_width).sum(axis=2) & 1
        ).astype(bool)
        target_sign = np.where(target, -1.0, 1.0)
        prior = np.empty((shots, variables+1), dtype=np.float64)
        # The decoder is not told the sampled hook location. A small
        # degree-weighted marginal allowance is only a heuristic prior.
        allowance = 0.002*opposite*0.35
        decoding_base = np.concatenate((
            np.tile(np.minimum(data_rate+allowance, 0.25), rounds),
            np.tile(measurement_rate, rounds),
        ))
        decoding_rates = np.minimum(
            levels[:, None]*decoding_base[None, :], 0.30)
        prior[:, :variables] = np.log(
            (1-decoding_rates)/decoding_rates)
        prior[:, dummy] = 30.0
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
            shaped = messages.reshape(count, detector_count, row_width)
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
                    count, detector_count, row_width).sum(axis=2) & 1
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

    seed = 550311007+1009*idx
    for restart in range(3):
        aa = select("A", brief_a)
        bb = select("B", brief_b)
        options = []
        for ka, ta, A, ar in aa:
            for kb, tb, B, br in bb:
                joint = tuple(v for a, b in zip(ka, kb)
                              for v in (max(a, b), a+b))
                options.append((joint, ta, tb, A, B, ar, br))
        options.sort(key=lambda item: item[:3])

        coarse = []
        for number, option in enumerate(options[:4]):
            joint, ta, tb, A, B, ar, br = option
            hx, hz, n = lifted_matrices(G, A, B)
            k = n-gf2_rank(hx)-gf2_rank(hz)
            if n != 550 or k != 220:
                continue
            actual_a = [
                sum(((row >> (3*j*q)) & mask) << (j*q)
                    for j in range(3))
                for row in hx[:q]
            ]
            actual_b = [row & ((1 << width)-1) for row in hz[:q]]
            assert actual_a == ar and actual_b == br
            if _af_logical4(hx, hz, n) or _af_logical4(hz, hx, n):
                continue
            assert css_ok(hx, hz)
            assert max(row.bit_count() for row in hx+hz) == 11
            assert 6*sum(row.bit_count() for row in hx+hz) == 21120
            x = temporal_probe(hz, hx, n, seed, 48)
            z = temporal_probe(hx, hz, n, seed+1, 48)
            coarse.append((quality(x, z)+joint+(number,),
                           number, A, B, hx, hz, joint, x, z))
        coarse.sort(key=lambda item: item[:2])

        finalists = []
        for training, number, A, B, hx, hz, joint, cx, cz in coarse[:2]:
            # Holdout results lead the decision, rather than being
            # pooled with samples that already selected these finalists.
            x = temporal_probe(hz, hx, 550, seed+1000003, 128)
            z = temporal_probe(hx, hz, 550, seed+1000004, 128)
            key = quality(x, z)+quality(cx, cz)+joint+(number,)
            finalists.append((key, A, B, x, z, cx, cz))
        if finalists:
            finalists.sort(key=lambda item: item[0])
            key, A, B, x, z, cx, cz = finalists[0]
            return {"A": A, "B": B, "k": 220,
                    "contract": (brief_a, brief_b),
                    "training_surrogate": (cx, cz),
                    "holdout_surrogate": (x, z), "screen": key}

    raise ValueError("no mixed-orientation code passed complete screening")


def CELL(rng, G, side, i, j, ctx):
    """Emit the complete holdout-selected cell without further random draws."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled uniform-label supports adequately protect this D14 (6,2)x(6,2) bed: all five measured parent codes have sector-2 weight-four witnesses and worse-basis failed-shot fractions of 0.268-0.452, despite mostly stage-1 settlement.
# REFUTED: Balanced D14 matching components screened against short dependencies suffice for rare circuit failures: fitness was 4.2963, all five measured codes retained abundant sector-2 weight-six witnesses, and worst-basis failures reached about 100 in 1500-2250 shots with flips spanning all 224 logicals.
# REFUTED: Full-rank four-cycle-free affine trinomials selected independently by exact weight-six multiplicity and expansion suffice for rare basis-balanced memory: fitness was 4.8408, every measured basis reached a failed-shot cap, and slot1 failed 100/4250 X versus 101/11000 Z shots despite k=220 and d2s=8.
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
