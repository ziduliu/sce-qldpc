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
CONCEPT = ("For C19 semidirect C3 with action seven, order 57, shapes (1,3)x(1,3), order-nineteen binomials, invariant-distinct component pairing and opposite-check temporal probes should improve worse-basis ASC fitness above the parent's 5.0213; measured circuit pL decides.")
GROUP = ("Jump from square-slope affine C11:C5 of order 55 to x^19=y^3=1, y*x*y^-1=x^7, of order 57. Since 7^3=1 modulo nineteen, this defines a valid semidirect product. The listed order-57 rate-0.400 bed reached about 4.9, not a demonstrated advantage over the parent. Test the lighter degree profile with translation cycles of length nineteen instead of eleven.")
SHAPES = ("Keep equal (1,3) sides for target cell 0.400: each component has 171 variables and 57 checks, n=570, kb=228 and guaranteed rate 0.400. The shape permits q<=59; require complete verified k=228. Parent weight-seven witnesses and collective failures span sector-one Acol/Bcol blocks broadly. Acol/Brow check enrichment is mostly near 1.0 and data enrichment mostly 1.0-1.2x, without an established persistent local repair target.")
FILLING = ("Both sides retain (2,3,3), maximum CSS weight eleven and 21888 six-round CNOTs. This is more total gates than the parent's 21120 because n increases; gates per data qubit are unchanged. Five quotient contracts include a repeated-phase trinomial, avoiding the distinct-phase quotient rank obstruction. Deterministic fresh searches retain rooted-invariant-distinct component classes, screened using verified actions for full rank, connectivity, C4 and kernels through four, then exact five/six-word censuses and triple boundaries. Reproducible prefix reservation excludes previously reserved unordered finalist pairs; fingerprints are conservative invariants, not complete Tanner canonization. Complete verified finalists require CSS, k=228 and no logical through four. Six-round temporal BP uses degree-weighted data and measurement noise and approximate pairs from opposite stabilizer checks; independent holdout results lead selection. These probes omit ASC ordering and relay/LSD and are not measured fitness. Target all ten parental basis failure caps, roughly 100 failures in 4500-22000 shots, and flips across almost every logical qubit.")
FAMILY_NAME = "r040_C19sdC3_action7_q57_13x13_reservedclasses_opposite_temporal_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 57

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C19 semidirect C3 requires order 57")
    # (a,b)=x^a*y^b, x^19=y^3=1, y*x*y^-1=x^7.
    action = (1, 7, 11)
    els = [(a,b) for b in range(3) for a in range(19)]

    def red(g):
        return (int(g[0]) % 19, int(g[1]) % 3)

    def mul(g, h):
        a,b = red(g)
        c,d = red(h)
        return ((a + action[b]*c) % 19, (b+d) % 3)

    def inv(g):
        a,b = red(g)
        bi = (-b) % 3
        return ((-action[bi]*a) % 19, bi)

    def idx(g):
        a,b = red(g)
        return a + 19*b

    return dict(elements=els, ident=(0,0), mul=mul, inv=inv,
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


def _af_rooted_identity(rows, width):
    """Rooted Tanner invariant, not complete graph canonization.

    Single-row regular components are transitive on their checks.
    Equal invariants are conservatively discarded within the bank.
    """
    from collections import Counter
    import hashlib

    nr = len(rows)
    adjacency = [[] for _ in range(nr + width)]
    for c, word in enumerate(_af_columns(rows, width)):
        while word:
            bit = word & -word
            r = bit.bit_length()-1
            adjacency[r].append(nr+c)
            adjacency[nr+c].append(r)
            word ^= bit
    colors = [0]*nr + [len(adjacency[nr+c]) for c in range(width)]
    colors[0] = 20
    digest = hashlib.sha256()
    for _ in range(12):
        signatures = [
            (colors[v], tuple(sorted(colors[w] for w in adjacency[v])))
            for v in range(len(adjacency))
        ]
        digest.update(repr(sorted(Counter(signatures).items())).encode())
        palette = {s: i for i, s in enumerate(sorted(set(signatures)))}
        colors = [palette[s] for s in signatures]
    return digest.hexdigest()


def NEW_CTX(rng, G, idx):
    """Pure structural banks, prefix reservation and temporal holdout selection."""
    from itertools import combinations_with_replacement

    q = G.q
    assert q == 57 and A_SHAPE == B_SHAPE == (1, 3)
    mask, width = (1 << q)-1, 3*q
    wa = wb = (2, 3, 3)
    sa, sb = [list(wa)], [list(wb)]
    assert _no_dead_lines(sa, sb)
    assert sum(wa)+max(wb) <= min(12, W_MAX)
    assert sum(wb)+max(wa) <= min(12, W_MAX)
    expected_weight = _grid_weights(sa, sb)
    expected_cnots = 24*q*(sum(wa)+sum(wb))

    # R1/R2: quotient (0,0,1)/(0,0,2).
    # T: quotient (0,1,2). At least one R per component:
    # its projection is a monomial, whereas two T cells share a
    # vanishing factor in F2[C3]. Actual ranks remain authoritative.
    light_book = [
        brief for brief in combinations_with_replacement(range(3), 2)
        if any(kind < 2 for kind in brief)
    ]
    assert len(light_book) == 5

    reflected = (1, 0, 2)
    locations = {brief: i for i, brief in enumerate(light_book)}

    def reflected_brief(number):
        return locations[tuple(sorted(
            reflected[kind] for kind in light_book[number]))]

    # Conservatively quotient a common orientation reversal and A/B
    # exchange at the contract level. Actual class reservation below
    # also removes equal rooted component-pair invariants.
    cards = sorted({
        min((a, b), tuple(sorted((reflected_brief(a),
                                 reflected_brief(b)))))
        for a, b in combinations_with_replacement(range(5), 2)
    })
    assert len(cards) >= 5

    # Bulk singleton probes use exactly the scaffold conventions.
    actions = {"A": {}, "B": {}}
    probe = [[[g] for g in G.elements]]
    zero = [[[]]]
    ax, _, _ = lifted_matrices(G, probe, zero)
    _, bz, _ = lifted_matrices(G, zero, probe)
    for t, g in enumerate(G.elements):
        actions["A"][G.idx(g)] = _af_columns(
            [(row >> (t*q)) & mask for row in ax], q)
        actions["B"][G.idx(g)] = _af_columns(
            [(row >> (t*q)) & mask for row in bz], q)

    def draw(weights, brief, source):
        cells, position = [], 0
        for count in weights:
            if count == 2:
                # Translation order nineteen. Scaling normalizes its
                # displacement, without erasing relative cell voltages.
                cells.append([G.e, (1, 0)])
                continue
            kind = brief[position]
            if kind < 2:
                displacement = source.randrange(1, 19)
                phase = 0 if position == 0 else source.randrange(19)
                u, v = (displacement, 0), (phase, kind+1)
            else:
                phase = 0 if position == 0 else source.randrange(19)
                u, v = (phase, 1), (source.randrange(19), 2)
            if G.mul(u, v) == G.mul(v, u):
                return None
            cell = [G.e, u, v]
            assert len({G._key(g) for g in cell}) == 3
            cells.append(cell)
            position += 1
        return [cells]

    def component(grid, side):
        columns = []
        for cell in grid[0]:
            terms = [actions[side][G.idx(g)] for g in cell]
            for c in range(q):
                value = 0
                for term in terms:
                    value ^= term[c]
                columns.append(value)
        return columns

    def clean4(columns):
        # Unlike _af_clean4, this handles mixed even/odd degrees.
        singles = set(columns)
        if 0 in singles or len(singles) != len(columns):
            return False
        seen = set()
        for i, x in enumerate(columns):
            for j in range(i):
                value = x ^ columns[j]
                if value == 0 or value in singles or value in seen:
                    return False
                seen.add(value)
        return True

    def graph(columns):
        """Exact C4 and, after C4 exclusion, exact Tanner C6."""
        adjacency = [0]*q
        multiplicity = {}
        four = stars = 0
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
                    old = multiplicity.get(key, 0)
                    four += old
                    multiplicity[key] = old + 1
                    adjacency[u] |= 1 << v
                    adjacency[v] |= 1 << u
        if four:
            return None

        reached, todo = {0}, [0]
        while todo:
            u = todo.pop()
            bits = adjacency[u]
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
        six = triangles - stars
        assert six >= 0
        return six

    def census(columns):
        """Exact weight-five/six counts after kernels <=4 are excluded."""
        pair_hist = {}
        for i, x in enumerate(columns):
            for y in columns[:i]:
                value = x ^ y
                pair_hist[value] = pair_hist.get(value, 0) + 1
        triple_hist = {}
        five = six = burden = 0
        loads = [0]*3
        for a in range(len(columns)-2):
            for b in range(a+1, len(columns)-1):
                pair = columns[a] ^ columns[b]
                for c in range(b+1, len(columns)):
                    value = pair ^ columns[c]
                    five += pair_hist.get(value, 0)
                    six += triple_hist.get(value, 0)
                    triple_hist[value] = triple_hist.get(value, 0) + 1
                    weight = value.bit_count()
                    if weight <= 4:
                        cost = (5-weight)**2
                        burden += cost
                        loads[a//q] += cost
                        loads[b//q] += cost
                        loads[c//q] += cost
        # Matching supports must be disjoint after clean4. Each
        # five-set has ten pair/triple partitions, and each six-set
        # has ten unordered triple/triple partitions.
        assert five % 10 == 0 and six % 10 == 0
        return five//10, six//10, max(loads), burden

    def select(side, weights, brief):
        # Same local bank in every pure slot call. Slot diversity comes
        # from reserved structural classes, not changing an orbit's seed.
        source = random.Random(
            570019317 + 1009*(brief[0]+3*brief[1]))
        classes, seen = {}, set()
        accepted = 0
        for trial in range(2400):
            grid = draw(weights, brief, source)
            if grid is None:
                continue
            signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                              for cell in grid[0])
            if signature in seen:
                continue
            seen.add(signature)
            columns = component(grid, side)
            cycles = graph(columns)
            if cycles is None:
                continue
            rows = _af_rows(columns, q)
            if gf2_rank(rows) != q or not clean4(columns):
                continue
            accepted += 1
            identity = _af_rooted_identity(rows, width)
            item = (cycles, trial, grid, rows, columns, identity)
            old = classes.get(identity)
            if old is None or item[:2] < old[:2]:
                classes[identity] = item
            if len(classes) > 16:
                retained = sorted(classes.values(),
                                  key=lambda entry: entry[:2])[:16]
                classes = {entry[-1]: entry for entry in retained}
            if len(classes) >= 12 and accepted >= 96 and trial >= 384:
                break
        if not classes:
            raise ValueError("no clean order-nineteen component class")

        refined = []
        pool = sorted(classes.values(), key=lambda entry: entry[:2])[:10]
        for cycles, trial, grid, rows, columns, identity in pool:
            key = census(columns)+(cycles,)
            refined.append((key, trial, grid, rows, identity))
        refined.sort(key=lambda item: item[:2])
        return refined[:8]

    def temporal_probe(checks, stabilizers, n, seed, shots):
        """Six-round phenomenological BP surrogate, NOT ASC simulation.

        Data increments and noisy measurements are decoded jointly.
        The final syndrome is perfect. Degree-weighted rates and
        random neighboring pairs approximate extraction exposure;
        no actual ASC gate ordering or relay/LSD cascade is claimed.
        """
        import numpy as np

        neighbors = []
        for row in checks:
            ns = []
            while row:
                bit = row & -row
                ns.append(bit.bit_length()-1)
                row ^= bit
            neighbors.append(ns)
        hooks = []
        for row in stabilizers:
            ns = []
            while row:
                bit = row & -row
                ns.append(bit.bit_length()-1)
                row ^= bit
            if len(ns) >= 2:
                hooks.append(ns)
        assert hooks
        m = len(checks)
        spatial_width = max(map(len, neighbors))
        data_dummy = n
        spatial = np.full((m, spatial_width), data_dummy,
                          dtype=np.int32)
        for r, ns in enumerate(neighbors):
            spatial[r, :len(ns)] = ns

        degrees = np.asarray(
            [c.bit_count() for c in _af_columns(checks, n)])
        other_degrees = np.asarray(
            [c.bit_count() for c in _af_columns(stabilizers, n)])
        round_data = np.minimum(
            0.002*(2 + degrees + other_degrees)*2/3, 0.12)
        round_measure = np.asarray(
            [0.002*(2 + len(ns))*2/3 for ns in neighbors])

        rounds = 6
        variables = rounds*(n+m)
        dummy = variables
        row_width = spatial_width + 2
        incidence = np.full(((rounds+1)*m, row_width), dummy,
                            dtype=np.int32)
        for t in range(rounds+1):
            for r, ns in enumerate(neighbors):
                detector = t*m+r
                position = 0
                if t < rounds:
                    for c in ns:
                        incidence[detector, position] = t*n+c
                        position += 1
                    incidence[detector, position] = rounds*n+t*m+r
                    position += 1
                if t:
                    incidence[detector, position] = (
                        rounds*n+(t-1)*m+r)

        flat = incidence.ravel()
        valid = flat != dummy
        edges = len(flat)
        variable_degree = np.bincount(
            flat[valid], minlength=variables)
        assert np.all(variable_degree > 0)
        variable_edges = np.full(
            (variables+1, int(variable_degree.max())),
            edges, dtype=np.int32)
        used = np.zeros(variables, dtype=np.int32)
        for edge in np.flatnonzero(valid):
            c = int(flat[edge])
            variable_edges[c, used[c]] = edge
            used[c] += 1

        channel = np.random.default_rng(seed)
        levels = np.asarray((1.0, 1.4, 1.8))[np.arange(shots) % 3]
        base = np.concatenate((
            np.tile(round_data, rounds),
            np.tile(round_measure, rounds),
        ))
        rates = np.minimum(levels[:, None]*base[None, :], 0.25)
        error = np.zeros((shots, variables+1), dtype=bool)
        error[:, :variables] = channel.random(
            (shots, variables)) < rates

        # Possible spread pairs belong to OPPOSITE stabilizer checks,
        # not checks sensing this error component. Their rate and
        # ordering remain an uncalibrated stress model, not ASC hooks.
        for t in range(rounds):
            hook_checks = channel.integers(len(hooks), size=shots)
            active = channel.random(shots) < 0.20
            for shot in np.flatnonzero(active):
                ns = hooks[int(hook_checks[shot])]
                a = int(channel.integers(len(ns)))
                b = int(channel.integers(len(ns)-1))
                b += b >= a
                error[shot, t*n+ns[a]] ^= True
                error[shot, t*n+ns[b]] ^= True

        target = (
            error[:, flat].reshape(
                shots, (rounds+1)*m, row_width).sum(axis=2) & 1
        ).astype(bool)
        target_sign = np.where(target, -1.0, 1.0)
        prior = np.empty((shots, variables+1), dtype=np.float64)
        prior[:, :variables] = np.log((1-rates)/rates)
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

        failures = iteration_sum = 0
        detector_count = (rounds+1)*m
        for iteration in range(1, 25):
            count = len(prior)
            shaped = messages.reshape(count, detector_count, row_width)
            signs = np.where(shaped < 0, -1.0, 1.0)
            logs = np.log(np.clip(
                np.tanh(np.abs(shaped)/2), 1e-12, 1-1e-8))
            products = np.minimum(
                np.exp(logs.sum(axis=2, keepdims=True)-logs), 1-1e-8)
            outgoing = (
                (np.log1p(products)-np.log1p(-products))
                * signs * signs.prod(axis=2, keepdims=True)
                * target_sign[:, :, None]
            ).reshape(count, edges)
            outgoing[:, ~valid] = 0.0
            padded = np.zeros((count, edges+1), dtype=np.float64)
            padded[:, :edges] = outgoing
            posterior = prior + padded[:, variable_edges].sum(axis=2)
            posterior[:, dummy] = 30.0
            hard = posterior < 0
            syndrome = (
                hard[:, flat].reshape(
                    count, detector_count, row_width).sum(axis=2) & 1
            ).astype(bool)
            settled = np.all(syndrome == target, axis=1)
            decoded_final = np.logical_xor.reduce(
                hard[:, :rounds*n].reshape(count, rounds, n), axis=1)
            for local in np.flatnonzero(settled):
                residual = decoded_final[local] ^ true_final[local]
                word = 0
                for c in np.flatnonzero(residual):
                    word |= 1 << int(c)
                failures += int(nontrivial(word))
                iteration_sum += iteration

            keep = ~settled
            if not np.any(keep):
                return failures, 0, iteration_sum
            messages = np.clip(
                0.30*messages
                + 0.70*(posterior[:, flat]-outgoing), -30, 30)[keep]
            messages[:, ~valid] = 30.0
            prior = prior[keep]
            target = target[keep]
            target_sign = target_sign[keep]
            true_final = true_final[keep]

        unresolved = len(prior)
        return (failures+unresolved, unresolved,
                iteration_sum+25*unresolved)

    # Group inversion maps the A-left component Tanner graph to the
    # B-right component graph with the same cells. Build one structural
    # bank per brief, then explicitly verify the actual B-right rows.
    banks = [select("A", wa, brief) for brief in light_book]
    catalogues = []
    for ia, ib in cards:
        unique = {}
        for ka, ta, A, ar, fa in banks[ia]:
            for kb, tb, B, _, fb in banks[ib]:
                identity = tuple(sorted((fa, fb)))
                joint = tuple(v for a, b in zip(ka, kb)
                              for v in (max(a, b), a+b))
                item = (joint, identity, ta, tb, A, B, ar)
                old = unique.get(identity)
                if old is None or item[:4] < old[:4]:
                    unique[identity] = item
        catalogues.append(sorted(unique.values(),
                                 key=lambda item: item[:4]))

    # Rebuild the same prefix locally. Reserve every finalist, including
    # unselected ones, so later slots cannot recycle its class pair.
    reserved = set()
    chosen = None
    actual_card = None
    for slot in range(idx+1):
        card = slot % len(cards)
        available = [item for item in catalogues[card]
                     if item[1] not in reserved]
        if not available:
            alternatives = [
                (item[0], item[1], c)
                for c, catalogue in enumerate(catalogues)
                for item in catalogue if item[1] not in reserved
            ]
            if not alternatives:
                raise ValueError("unused component-class pairs exhausted")
            card = min(alternatives)[2]
            available = [item for item in catalogues[card]
                         if item[1] not in reserved]
        chosen = available[:3]
        actual_card = card
        reserved.update(item[1] for item in chosen)

    options = []
    for joint, identity, ta, tb, A, B, ar in chosen:
        br = _af_rows(component(B, "B"), q)
        hx, hz, n = lifted_matrices(G, A, B)
        k = n-gf2_rank(hx)-gf2_rank(hz)
        if n != 570 or k != 228:
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
        assert max(row.bit_count() for row in hx+hz) == expected_weight
        assert 6*sum(row.bit_count() for row in hx+hz) == expected_cnots
        options.append((joint, identity, ta, tb, A, B, hx, hz))
    if not options:
        raise ValueError("no reserved order-57 pair passed complete screening")
    options.sort(key=lambda item: item[:4])

    def quality(x, z):
        return (max(x[0], z[0]), x[0]+z[0],
                max(x[1], z[1]), x[1]+z[1], x[2]+z[2])

    seed = 570019003+1009*idx
    coarse = []
    for number, option in enumerate(options):
        joint, identity, ta, tb, A, B, hx, hz = option
        x = temporal_probe(hz, hx, 570, seed, 24)
        z = temporal_probe(hx, hz, 570, seed+1, 24)
        coarse.append((quality(x, z)+joint+(number,),
                       number, option, x, z))
    coarse.sort(key=lambda item: item[:2])

    finalists = []
    for _, number, option, cx, cz in coarse[:2]:
        joint, identity, ta, tb, A, B, hx, hz = option
        # Holdout evidence leads, rather than pooling it with the
        # training patterns that already shortlisted these finalists.
        x = temporal_probe(hz, hx, 570, seed+1000003, 72)
        z = temporal_probe(hx, hz, 570, seed+1000004, 72)
        score = quality(x, z)+quality(cx, cz)+joint+(number,)
        finalists.append((score, A, B, identity, x, z, cx, cz))
    finalists.sort(key=lambda item: item[0])
    score, A, B, identity, x, z, cx, cz = finalists[0]
    ia, ib = cards[actual_card]
    return {"A": A, "B": B, "k": 228,
            "contract": (wa, wb, light_book[ia], light_book[ib]),
            "component_pair_invariant": identity,
            "training_surrogate": (cx, cz),
            "holdout_surrogate": (x, z),
            "screen": score}


def CELL(rng, G, side, i, j, ctx):
    """Return the already screened complete cell without resampling."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled uniform-label supports adequately protect this D14 (6,2)x(6,2) bed: all five measured parent codes have sector-2 weight-four witnesses and worse-basis failed-shot fractions of 0.268-0.452, despite mostly stage-1 settlement.
# REFUTED: Balanced D14 matching components screened against short dependencies suffice for rare circuit failures: fitness was 4.2963, all five measured codes retained abundant sector-2 weight-six witnesses, and worst-basis failures reached about 100 in 1500-2250 shots with flips spanning all 224 logicals.
# REFUTED: Full-rank four-cycle-free affine trinomials selected by exact weight-six multiplicity and expansion suffice for rare, basis-balanced memory failures: fitness was 4.8408, every measured basis hit a failed-shot cap, and slot1 had 100/4250 X versus 101/11000 Z failures despite k=220 and d2s=8.
# REFUTED: The order-55 long-binomial temporal surrogate suffices for rare collective ASC failures: fitness improved to 5.0213 but all ten measured bases still reached failed-shot caps, slot30 X failed 101/4500 shots, and logical flips remained broadly distributed; this does not isolate the surrogate from the labels.
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
