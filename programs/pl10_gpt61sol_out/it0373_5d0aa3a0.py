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
CONCEPT = ("For cyclic Z13 (metacyclic Z13:Z1, s=1), order 13, shapes (2,5)x(3,8), A13/A14-to-B25/B26 gate exchange with long-cycle mixed-row A columns and noisy six-round pairing probes should improve cell-0.350 worse-basis circuit fitness above 5.0793.")
GROUP = ("Frozen L1 group: preserve the exact implementation and order byte-for-byte. This Z13 bed has reached 5.1107, while the current parent scores 5.0793. Size factor 46 allows q<=13 and gives n=598. These readings motivate a filling experiment, not an inferred group advantage.")
SHAPES = ("Frozen (2,5)x(3,8): kb=195 and guaranteed rate 15/46=0.326087; actual helper-built ranks retain 195<=k<=201. Unequal components have 65 variables/26 checks and 104 variables/39 checks. Parent failures reached every logical qubit, with only diffuse 1.1-1.3x Acol/Bcol and mild Acol/Brow enrichment. No persistent isolated block defect is established.")
FILLING = ("Twenty contracts cross A13/A14, B25/B26 and five canonical B count architectures. A has two or one mixed-row degree-two columns, respectively; the verified weak incidence is one length-26 cycle or a forest. Other A columns have degree three. B has seven degree-three plus one degree-four column, or six degree-three plus two degree-four columns, with row loads at most nine. HX weights are <=11 and HZ <=12; six-round CNOT counts are 24804,25350,25662,26208 rather than 25974/26832. Fresh canonical A ratio cards extend additional harness batches. Helper-derived singleton actions accelerate component searches; rank and short-kernel guards retain alternatives. Complete verified CSS pairings are compared using six-round noisy-syndrome BP probes with degree-scaled background faults, approximate opposite-check suffix faults and independent holdouts. These omit ASC ordering and relay/LSD and are not circuit fitness. Target the parent's consistently X-limited failure caps, collective flips and slot6's 104/14750 X versus 100/30500 Z failures; do not optimize distance as fitness.")
FAMILY_NAME = "r035_m13r1s1_25x38_Alean_Bfour_temporalholdout_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 13

def custom_ops(q):
    # Same cyclic metacyclic construction; factor 46 requires q<=13.
    # Z_13 x| Z_1: y x y^-1 = x^1.
    if q != 13:
        raise ValueError("this bed requires order 13")
    m, r, s = 13, 1, 1
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 5)               # (ra, ca)
B_SHAPE = (3, 8)               # (rb, cb)


def NEW_CTX(rng, G, idx):
    """A-lean/B-strengthened contracts with independent temporal holdouts."""
    import math
    from itertools import combinations, permutations, combinations_with_replacement

    q = G.q
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    design, batch = idx % 20, idx // 20
    allocation, card = divmod(design, 5)
    a_model, budget = divmod(allocation, 2)
    assert q == 13

    a_profiles = (
        ((1, 1, 2, 2, 1), (1, 1, 1, 1, 2)),
        ((1, 2, 2, 1, 1), (1, 1, 1, 2, 2)),
    )
    sa = [list(row) for row in a_profiles[a_model]]
    three_types = sorted(
        {(1, 1, 1)} | set(permutations((2, 1, 0))))
    four_types = sorted(
        set(permutations((2, 1, 1)))
        | set(permutations((2, 2, 0)))
        | set(permutations((3, 1, 0))))

    def canonical_support(columns):
        # Removes count-card row/column orbit copies, not arbitrary GI.
        return min(
            tuple(sorted(tuple(column[i] for i in order)
                         for column in columns))
            for order in permutations(range(rb)))

    number_four = 1 + budget
    desired_loads = [8, 8, 9] if budget == 0 else [8, 9, 9]
    book = set()
    for tail in combinations_with_replacement(
            three_types, cb - number_four):
        if not 1 <= tail.count((1, 1, 1)) <= 5:
            continue
        for strong in combinations_with_replacement(
                four_types, number_four):
            columns = tail + strong
            sb_trial = [[c[i] for c in columns] for i in range(rb)]
            if sorted(map(sum, sb_trial)) != desired_loads:
                continue
            # Six undirected cyclic difference classes are available.
            # This filters an obvious internal C4 packing obstruction;
            # it does not assert that C4=0 is attainable.
            if any(sum(t * (t - 1) // 2 for t in row) > 6
                   for row in sb_trial):
                continue
            augmentation = [
                sum((t & 1) << j for j, t in enumerate(row))
                for row in sb_trial]
            if gf2_rank(augmentation) == rb:
                book.add(canonical_support(columns))
    book = sorted(book)
    assert len(book) >= 5
    b_columns = book[card * (len(book) - 1) // 4]
    sb = [[c[i] for c in b_columns] for i in range(rb)]
    counts = {"A": sa, "B": sb}

    assert sum(map(sum, sa)) == 13 + a_model
    assert sum(map(sum, sb)) == 25 + budget
    assert sorted(map(sum, b_columns)) == (
        [3] * (7 - budget) + [4] * (1 + budget))
    assert sorted(sum(row[j] for row in sa) for j in range(ca)) == (
        [2] * (2 - a_model) + [3] * (3 + a_model))
    assert max(map(sum, sa)) == 7
    assert max(map(sum, sb)) == 9
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert _no_dead_lines(sa, sb)

    def direction(value):
        value %= q
        return min(value, q - value)

    def geometry_key(steps):
        """Canonical internal A ratios; downstream full GI is authoritative."""
        return min(
            tuple(sorted(
                tuple((sa[i][j], direction(u * steps[i][j]))
                      for i in order)
                for j in range(ca)))
            for u in range(1, q)
            for order in permutations(range(ra)))

    # Extra batches change relative directions, not just voltage offsets.
    # Unlike the parent, B is not restricted to a complementary palette.
    book_rng = random.Random(132526013 + 1009 * design)
    seen_geometries = set()
    geometry = None
    number = 0
    for proposal in range(12000):
        steps = [[0] * ca for _ in range(ra)]
        for i, row in enumerate(sa):
            sites = [j for j, t in enumerate(row) if t == 2]
            choices = book_rng.sample(range(1, 7), len(sites))
            for j, d in zip(sites, choices):
                steps[i][j] = d
        key = geometry_key(steps)
        if key in seen_geometries:
            continue
        seen_geometries.add(key)
        if number == batch:
            geometry = steps
            break
        number += 1
    if geometry is None:
        raise ValueError("canonical A-lean ratio book exhausted")

    def transpose(rows, width):
        columns = [0] * width
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                columns[bit.bit_length() - 1] ^= 1 << r
                row ^= bit
        return columns

    def verified_component(side, grid):
        # Zero probes extract precisely the scaffold's left/right actions.
        rows = (lifted_matrices(G, grid, [[[]]])[0]
                if side == "A" else
                lifted_matrices(G, [[[]]], grid)[1])
        width = len(grid[0]) * q
        keep = (1 << width) - 1
        return [row & keep for row in rows]

    # Cache only helper-built singleton blocks. Linear composition is
    # checked against complete helper-built components before submission.
    actions = {"A": {}, "B": {}}
    for side in actions:
        for g in G.elements:
            actions[side][G._key(g)] = tuple(transpose(
                verified_component(side, [[[g]]]), q))

    def component_columns(side, grid):
        columns = [0] * (len(grid[0]) * q)
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                for g in cell:
                    block = actions[side][G._key(g)]
                    for t, value in enumerate(block):
                        columns[j * q + t] ^= value << (i * q)
        expected = [sum(row[j] for row in counts[side])
                    for j in range(len(grid[0]))]
        assert all(c.bit_count() == expected[j // q]
                   for j, c in enumerate(columns))
        return columns

    def weak_A_ok(columns):
        """Two matchings make one 26-cycle; one matching makes a forest."""
        weak = [c for c in columns if c.bit_count() == 2]
        expected = (2 - a_model) * q
        assert len(weak) == expected
        return gf2_rank(weak) == expected - int(a_model == 0)

    def draw_cell(side, i, j):
        t = counts[side][i][j]
        if t == 0:
            return []
        if side == "A" and t == 2:
            g = G.pick(rng)
            return [g, G.mul(g, (geometry[i][j], 0))]
        # Avoid internal arithmetic-progression C4s in B trinomials.
        # Inter-cell repetitions remain genuine search variables.
        for _ in range(64):
            cell = rng.sample(G.elements, t)
            differences = {
                direction(g[0] - h[0])
                for g, h in combinations(cell, 2)}
            if len(differences) == t * (t - 1) // 2:
                return cell
        raise ValueError("unable to draw an internal Sidon cell")

    def draw(side):
        for _ in range(128):
            grid = [[draw_cell(side, i, j) for j in range(len(row))]
                    for i, row in enumerate(counts[side])]
            if side == "B" or weak_A_ok(component_columns(side, grid)):
                return grid
        raise ValueError("mixed-row A cycle contract not realized")

    def cycle_data(cols):
        """Exact four-cycle count for either component and mixed degrees."""
        pairs = {}
        adj = [0] * max(c.bit_length() for c in cols)
        conflicts = 0
        for col in cols:
            nodes = []
            bits = col
            while bits:
                bit = bits & -bits
                nodes.append(bit.bit_length() - 1)
                bits ^= bit
            for a, u in enumerate(nodes):
                for v in nodes[a + 1:]:
                    key = (1 << u) | (1 << v)
                    old = pairs.get(key, 0)
                    conflicts += old
                    pairs[key] = old + 1
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
        return conflicts, adj

    def kernel_metric(cols, adj):
        """Short syndrome ambiguities, including odd-weight B witnesses.

        Counts are collision proxies if shorter words exist. They are
        not quantum distances and do not distinguish stabilizers.
        """
        singles = {}
        for syn in cols:
            singles[syn] = singles.get(syn, 0) + 1
        n2 = sum(v * (v - 1) // 2 for v in singles.values())
        pair_hist = {}
        n3 = n4 = 0
        nc = len(cols)
        for a in range(nc - 1):
            for b in range(a + 1, nc):
                syn = cols[a] ^ cols[b]
                n3 += singles.get(syn, 0)
                old = pair_hist.get(syn, 0)
                n4 += old
                pair_hist[syn] = old + 1

        triangles = 0
        for u in range(len(adj)):
            bits = adj[u] & ~((1 << (u + 1)) - 1)
            while bits:
                bit = bits & -bits
                v = bit.bit_length() - 1
                triangles += (
                    adj[u] & adj[v] & ~((1 << (v + 1)) - 1)
                ).bit_count()
                bits ^= bit
        stars = sum(c.bit_count() * (c.bit_count() - 1)
                    * (c.bit_count() - 2) // 6 for c in cols)
        # Exact Tanner six-cycles at C4=0; otherwise a proposal heuristic.
        n6cycles = max(0, triangles - stars)
        if n2 or n3 or n4:
            return (n2, n3, n4, 10**9, 10**9, n6cycles)

        triple_hist = {}
        n5 = n6 = 0
        for a in range(nc - 2):
            for b in range(a + 1, nc - 1):
                ab = cols[a] ^ cols[b]
                for c in range(b + 1, nc):
                    syn = ab ^ cols[c]
                    n5 += pair_hist.get(syn, 0)
                    old = triple_hist.get(syn, 0)
                    n6 += old
                    triple_hist[syn] = old + 1
        return (0, 0, 0, n5, n6, n6cycles)

    def rows_from_columns(cols):
        """Transpose the columns extracted from the verified lift."""
        rows = [0] * max(c.bit_length() for c in cols)
        for j, col in enumerate(cols):
            while col:
                bit = col & -col
                rows[bit.bit_length() - 1] |= 1 << j
                col ^= bit
        return rows

    def kernel_probe(cols, trials):
        """Random systematic bases give verified component witnesses only."""
        source = rows_from_columns(cols)
        nc = len(cols)
        minimum = nc + 1
        small = set()
        for _ in range(trials):
            rows = list(source)
            order = list(range(nc))
            rng.shuffle(order)
            pivots = []
            rank = 0
            for c in order:
                bit = 1 << c
                hit = next((t for t in range(rank, len(rows))
                            if rows[t] & bit), None)
                if hit is None:
                    continue
                rows[rank], rows[hit] = rows[hit], rows[rank]
                pivot = rows[rank]
                for t in range(len(rows)):
                    if t != rank and rows[t] & bit:
                        rows[t] ^= pivot
                pivots.append(c)
                rank += 1
                if rank == len(rows):
                    break
            occupied = set(pivots)
            for c in order:
                if c in occupied:
                    continue
                word = 1 << c
                for p, row in zip(pivots, rows[:rank]):
                    if (row >> c) & 1:
                        word |= 1 << p
                weight = word.bit_count()
                minimum = min(minimum, weight)
                if weight <= 10:
                    small.add(word)
            if minimum <= 6:
                break
        return minimum, len(small)

    def signature(grid):
        """Ignore component column gauges when retaining search finalists."""
        columns = []
        for j in range(len(grid[0])):
            anchors = [g[0] for row in grid for g in row[j]]
            columns.append(min(
                tuple(tuple(sorted((g[0] - anchor) % q
                                   for g in row[j]))
                      for row in grid)
                for anchor in anchors))
        return tuple(columns)

    def kernel78(columns, metric):
        """Exact A weight-seven/eight counts after weights <=6 vanish.

        Colliding triple/quadruple supports are then disjoint. Each
        seven-set has 35 triple/quad partitions, and each eight-set has
        35 unordered quad/quad partitions. These are component kernels,
        not stabilizer-aware quantum logical counts.
        """
        if any(metric[:-1]):
            return (10**9, 10**9)
        triples = {}
        nc = len(columns)
        for a, b, c in combinations(range(nc), 3):
            syn = columns[a] ^ columns[b] ^ columns[c]
            triples[syn] = triples.get(syn, 0) + 1
        quads = {}
        seven = eight = 0
        for a, b, c, d in combinations(range(nc), 4):
            syn = columns[a] ^ columns[b] ^ columns[c] ^ columns[d]
            seven += triples.get(syn, 0)
            old = quads.get(syn, 0)
            eight += old
            quads[syn] = old + 1
        assert seven % 35 == 0 and eight % 35 == 0
        return seven // 35, eight // 35

    def optimize(side):
        """Retain several guarded geometries, including legal C4 minima."""
        sites = [(i, j) for i, row in enumerate(counts[side])
                 for j, t in enumerate(row) if t]
        checks = len(counts[side]) * q
        augmentation = [
            sum((t & 1) << j for j, t in enumerate(row))
            for row in counts[side]]
        allowed_deficit = len(counts[side]) - gf2_rank(augmentation)
        assert allowed_deficit == 0
        pool, tested = [], set()

        def quick_short(columns):
            singles = {}
            two = three = four = 0
            for value in columns:
                two += singles.get(value, 0)
                singles[value] = singles.get(value, 0) + 1
            pairs = {}
            for i, value in enumerate(columns):
                for other in columns[:i]:
                    syndrome = value ^ other
                    three += singles.get(syndrome, 0)
                    four += pairs.get(syndrome, 0)
                    pairs[syndrome] = pairs.get(syndrome, 0) + 1
            return two, three, four

        def energy(columns, cycles, adjacency):
            triangles = 0
            for u in range(len(adjacency)):
                bits = adjacency[u] & ~((1 << (u + 1)) - 1)
                while bits:
                    bit = bits & -bits
                    v = bit.bit_length() - 1
                    triangles += (
                        adjacency[u] & adjacency[v]
                        & ~((1 << (v + 1)) - 1)).bit_count()
                    bits ^= bit
            stars = sum(
                d * (d - 1) * (d - 2) // 6
                for d in (c.bit_count() for c in columns))
            return (64 * cycles + max(0, triangles - stars) / 8) / q

        def retain(grid, columns, cycles, adjacency):
            sig = signature(grid)
            if sig in tested:
                return
            tested.add(sig)
            deficit = checks - gf2_rank(rows_from_columns(columns))
            short = quick_short(columns)
            key = (deficit,) + short + (cycles,)
            # Random ties avoid repeatedly retaining lexicographically
            # smallest voltages. They do not transform completed codes.
            pool.append((key, rng.random(), sig, grid, columns, adjacency))
            pool.sort(key=lambda item: item[:3])
            del pool[10:]

        for restart in range(3):
            current = draw(side)
            columns = component_columns(side, current)
            cycles, adjacency = cycle_data(columns)
            current_energy = energy(columns, cycles, adjacency)
            local_min = current_energy
            retain(current, columns, cycles, adjacency)

            for step in range(560):
                i, j = rng.choice(sites)
                trial = [list(row) for row in current]
                trial[i][j] = draw_cell(side, i, j)
                tc = component_columns(side, trial)
                if side == "A" and not weak_A_ok(tc):
                    continue
                tcycles, tadj = cycle_data(tc)
                proposed = energy(tc, tcycles, tadj)
                delta = proposed - current_energy
                temperature = 12 * (0.987 ** step) + 0.10
                if delta <= 0 or rng.random() < math.exp(
                        -min(700.0, delta / temperature)):
                    current, columns = trial, tc
                    cycles, adjacency = tcycles, tadj
                    current_energy = proposed
                improved = current_energy < local_min
                local_min = min(local_min, current_energy)
                if improved or step % 28 == 0:
                    retain(current, columns, cycles, adjacency)
            retain(current, columns, cycles, adjacency)

        refined = []
        for quick, tie, sig, grid, columns, adjacency in pool:
            if any(quick[:4]):
                continue
            metric = (quick[0],) + kernel_metric(columns, adjacency)
            preliminary = (
                metric[:4] + (int(metric[4] > 0), int(metric[5] > 0),
                              metric[4], metric[5], quick[-1],
                              metric[-1], tie))
            refined.append(
                (preliminary, grid, columns, metric, quick[-1], sig))
        refined.sort(key=lambda item: item[0])
        if not refined:
            raise ValueError("no full-rank component without kernels <=4")

        finalists = []
        for pre, grid, columns, metric, cycles, sig in refined[:5]:
            upper, multiplicity = kernel_probe(columns, 32)
            next_weights = (
                kernel78(columns, metric) if side == "A" else ())
            # Distance observations are capped late tie-breakers. These
            # component statistics only shortlist complete CSS pairings.
            key = pre[:-1] + next_weights + (
                max(0, 7 - upper), multiplicity,
                -min(upper, 10), sig)
            finalists.append({
                "key": key, "grid": grid, "metric": metric,
                "c4": cycles, "upper": upper})
        finalists.sort(key=lambda item: item["key"])
        return finalists[:3]

    def temporal_probe(checks, stabilizers, n, seed, shots):
        """Six-round noisy-syndrome stress test, NOT ASC simulation.

        The detector equations use the verified full check matrix:
        H*delta_e[t] + m[t] + m[t-1]. A perfect closing observation
        supplies the last measurement-error boundary. Random proper
        suffixes of opposite-check supports approximate hook exposure;
        their ordering is not asserted to be the ASC ordering.
        """
        import numpy as np

        def neighbors(rows):
            out = []
            for row in rows:
                sites = []
                while row:
                    bit = row & -row
                    sites.append(bit.bit_length() - 1)
                    row ^= bit
                out.append(sites)
            return out

        spatial = neighbors(checks)
        hooks = neighbors(stabilizers)
        m, rounds = len(checks), 6
        variables = rounds * (n + m)
        dummy = variables
        detectors = (rounds + 1) * m
        row_width = max(map(len, spatial)) + 2
        incidence = np.full(
            (detectors, row_width), dummy, dtype=np.int32)
        for t in range(rounds + 1):
            for r, sites in enumerate(spatial):
                position = 0
                if t < rounds:
                    incidence[t*m+r, :len(sites)] = [
                        t*n+v for v in sites]
                    position = len(sites)
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
            (variables + 1, int(degree.max())),
            edges, dtype=np.int32)
        used = np.zeros(variables, dtype=np.int32)
        for edge in np.flatnonzero(valid):
            v = int(flat[edge])
            variable_edges[v, used[v]] = edge
            used[v] += 1

        gate_degree = np.asarray([
            a.bit_count() + b.bit_count()
            for a, b in zip(transpose(checks, n),
                            transpose(stabilizers, n))
        ], dtype=np.float64)
        background = 0.002 * (2 + gate_degree) * (2.0 / 3.0)
        measurement = np.asarray([
            0.002 * (len(sites) + 2) * (2.0 / 3.0)
            for sites in spatial])
        hook_marginal = np.zeros(n, dtype=np.float64)
        for sites in hooks:
            if len(sites) >= 2:
                hook_marginal[sites] += 0.002 * (len(sites) + 1) / 6.0

        channel = np.random.default_rng(seed)
        levels = np.asarray((0.45, 0.70, 0.95))[
            np.arange(shots) % 3]
        background_base = np.concatenate((
            np.tile(background, rounds),
            np.tile(measurement, rounds)))
        rates = np.minimum(
            levels[:, None] * background_base[None, :], 0.22)
        error = np.zeros((shots, variables + 1), dtype=bool)
        error[:, :variables] = (
            channel.random((shots, variables)) < rates)

        # Actual opposite-check supports, but deliberately approximate
        # propagation and ordering. No simulator measurement is claimed.
        for sites in hooks:
            if len(sites) < 2:
                continue
            order = np.asarray(sites, dtype=np.int32)
            channel.shuffle(order)
            event_rate = levels * 0.002 * (len(sites) + 1) / 3.0
            for t in range(rounds):
                events = np.flatnonzero(
                    channel.random(shots) < event_rate)
                for shot in events:
                    cut = int(channel.integers(1, len(sites)))
                    error[shot, t*n+order[cut:]] ^= True

        # BP knows marginal exposure, not the correlated suffix channel.
        marginal_base = np.concatenate((
            np.tile(background + hook_marginal, rounds),
            np.tile(measurement, rounds)))
        marginal = np.clip(
            levels[:, None] * marginal_base[None, :], 0.003, 0.22)
        prior = np.empty((shots, variables + 1), dtype=np.float64)
        prior[:, :variables] = np.log((1 - marginal) / marginal)
        prior[:, dummy] = 30.0
        target = (error[:, flat].reshape(
            shots, detectors, row_width).sum(axis=2) & 1).astype(bool)
        target_sign = np.where(target, -1.0, 1.0)
        messages = prior[:, flat].copy()
        messages[:, ~valid] = 30.0
        true_final = np.logical_xor.reduce(
            error[:, :rounds*n].reshape(shots, rounds, n), axis=1)

        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = row
                    break
                row ^= old

        def nontrivial(word):
            while word:
                p = word.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    return True
                word ^= old
            return False

        wrong = effort = 0
        late = None
        for iteration in range(1, 37):
            count = len(prior)
            shaped = messages.reshape(count, detectors, row_width)
            signs = np.where(shaped < 0, -1.0, 1.0)
            logs = np.log(np.clip(
                np.tanh(np.abs(shaped) / 2), 1e-12, 1 - 1e-8))
            product = np.minimum(
                np.exp(logs.sum(axis=2, keepdims=True) - logs),
                1 - 1e-8)
            outgoing = (
                (np.log1p(product) - np.log1p(-product))
                * signs * signs.prod(axis=2, keepdims=True)
                * target_sign[:, :, None]
            ).reshape(count, edges)
            outgoing[:, ~valid] = 0.0
            padded = np.zeros((count, edges + 1), dtype=np.float64)
            padded[:, :edges] = outgoing
            posterior = prior + padded[:, variable_edges].sum(axis=2)
            posterior[:, dummy] = 30.0
            hard = posterior < 0
            observed = (hard[:, flat].reshape(
                count, detectors, row_width).sum(axis=2) & 1).astype(bool)
            settled = np.all(observed == target, axis=1)
            decoded_final = np.logical_xor.reduce(
                hard[:, :rounds*n].reshape(count, rounds, n), axis=1)
            for shot in np.flatnonzero(settled):
                residual = decoded_final[shot] ^ true_final[shot]
                word = sum(1 << int(v)
                           for v in np.flatnonzero(residual))
                wrong += int(nontrivial(word))
                effort += iteration

            keep = ~settled
            if iteration == 12:
                late = int(keep.sum())
            if not np.any(keep):
                return wrong, 0, (0 if late is None else late), effort
            damping = 0.25 if iteration <= 12 else 0.45
            messages = np.clip(
                damping * messages
                + (1 - damping) * (posterior[:, flat] - outgoing),
                -30, 30)[keep]
            messages[:, ~valid] = 30.0
            prior = prior[keep]
            target = target[keep]
            target_sign = target_sign[keep]
            true_final = true_final[keep]

        unresolved = len(prior)
        return (wrong + unresolved, unresolved,
                unresolved if late is None else late,
                effort + 37 * unresolved)

    def probe_quality(x, z):
        return (max(x[0], z[0]), x[0] + z[0],
                max(x[1], z[1]), x[1] + z[1],
                max(x[2], z[2]), x[2] + z[2],
                x[3] + z[3])

    # Retrying searches preserves the slot's count and ratio contract.
    # Only final legal cells are submitted; no invalid slot is repaired.
    for attempt in range(3):
        try:
            apool, bpool = optimize("A"), optimize("B")
        except ValueError:
            continue
        options = []
        for ia, a in enumerate(apool):
            for ib, b in enumerate(bpool):
                A, B = a["grid"], b["grid"]
                hx, hz, n = lifted_matrices(G, A, B)
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if n != 598 or not 195 <= k <= 201:
                    continue
                ma, mb = a["metric"], b["metric"]
                guard = tuple(max(x, y) for x, y in
                              zip(ma[:4], mb[:4]))
                coarse = guard + (
                    max(ma[4], mb[4]), ma[4] + mb[4],
                    max(ma[5], mb[5]), ma[5] + mb[5],
                    a["c4"] + b["c4"], ia + ib, ia, ib)
                options.append(
                    (coarse, A, B, hx, hz, n, k, guard))
        if not options:
            continue
        options.sort(key=lambda item: item[0])

        # Common streams within each contract reduce pairing-selection
        # noise. The independent holdout is never used to generate cells.
        seed = 132526598 + 1009 * idx
        coarse_probes = []
        for serial, option in enumerate(options[:3]):
            _, A, B, hx, hz, n, k, guard = option
            x = temporal_probe(hz, hx, n, seed, 24)
            z = temporal_probe(hx, hz, n, seed + 1, 24)
            key = guard + probe_quality(x, z) + (serial,)
            coarse_probes.append((key, serial, option, x, z))
        coarse_probes.sort(key=lambda item: item[:2])

        winner = None
        for _, serial, option, cx, cz in coarse_probes[:2]:
            coarse, A, B, hx, hz, n, k, guard = option
            x = temporal_probe(hz, hx, n, seed + 1000003, 48)
            z = temporal_probe(hx, hz, n, seed + 1000004, 48)
            tx = tuple(u + v for u, v in zip(cx, x))
            tz = tuple(u + v for u, v in zip(cz, z))
            key = guard + probe_quality(tx, tz) + coarse[4:] + (serial,)
            if winner is None or key < winner[0]:
                winner = (key, option, tx, tz)

        key, option, tx, tz = winner
        _, A, B, hx, hz, n, k, guard = option
        cells = {"A": A, "B": B}
        assert not any(guard)
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx) <= 11
        assert max(row.bit_count() for row in hz) <= 12
        assert max(row.bit_count() for row in hx + hz) <= W_MAX
        assert weak_A_ok(component_columns("A", A))
        for side, grid in cells.items():
            columns = component_columns(side, grid)
            assert transpose(
                verified_component(side, grid), len(columns)) == columns
            assert all(len(cell) == len({G._key(g) for g in cell})
                       for row in grid for cell in row)

        expected_cnots = 6 * q * (
            11 * sum(map(sum, sa)) + 7 * sum(map(sum, sb)))
        assert expected_cnots == 24804 + 858 * a_model + 546 * budget
        assert 6 * sum(row.bit_count() for row in hx + hz) == expected_cnots
        return {
            "idx": idx, "cells": cells, "k": k, "screen": key,
            "contract": (a_model, budget, card, batch),
            "geometry": geometry, "cnots": expected_cnots,
            "temporal_surrogate": (tx, tz)}
    raise ValueError("no guarded temporal-screened filling in target cell")


def CELL(rng, G, side, i, j, ctx):
    """Return the precomputed full cell without changing shared state."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Equal (n,k,d2s) determines circuit performance: parent Z17 slots with (578,204,5) scored 3.691-4.347, and slot 50 was strongly Z-limited despite its weight-seven Z-type witnesses.
# REFUTED: Weight-eight witnesses in both bases imply balanced memory reliability: current parent slot 20 had 100/8250 Z failures versus 100/50250 X failures, with only diffuse block enrichment.
# REFUTED: Component short-kernel screening and d2s=8 suffice for low, balanced circuit failure in this Z13 bed: all five measured codes hit failure caps; slot 1 had 103/7500 X versus 101/33000 Z failures.
# REFUTED: The parent's side-disjoint support/ratio contracts supplied twenty inequivalent designs: only eight distinct codes survived sixty slots, with extensive isomorphic duplication.
# REFUTED: Verified removal of mixed-sector four-cycles suffices for basis-balanced rare failures on this bed: all five current-parent codes remained X-limited, hit failure caps, and scored 4.984-5.135.
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
