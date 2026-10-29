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
CONCEPT = ("Q8 x C5, order 40, shapes (1,2)x(2,6): joint noisy-history decoder selection of fresh balanced unit-trinomial fillings should improve worse-basis circuit fitness beyond 5.9459 without increasing extraction work.")
GROUP = ("Frozen under L1: retain the exact quaternion direct-product group, order 40 and encoding. Trinomials project to units in F2[C5], and the nilpotent kernel ideal associated with normal Q8 lifts those units. Confirm A-block ranks with the verified helper. The current bed's measured scores 5.712..6.218 remain well below its approximately 7.72 full-sample ceiling; this motivates another filling experiment, not a group change.")
SHAPES = ("Frozen A=(1,2), B=(2,6): n=560, kb=160 and guaranteed rate 2/7 target cell 0.300. Components have 80 variables/40 checks and 240 variables/80 checks. Retain actual 160<=k<182 using full verified ranks. Parent failures span Acol/Bcol blocks and essentially all 160 logical qubits; data enrichment is mostly 1.1x and check enrichment approximately 1.0x. Neither the Acol/Brow maps nor worst-logical indices establish a persistent localized defect.")
FILLING = ("Twenty contracts cross five balanced B topologies with four noncommuting triangle/binomial policies; larger harness batches change relative spectra and marked positions. All B columns total three and both rows total nine. A cells remain trinomials, so HX/HZ weights are 9/12 and extraction remains 24480 CNOTs. Fresh A relative supports remove whole-cell origin search; B retains independently optimized inter-cell geometry. Verified component screening supplies complete-code alternatives, followed by common-noise repeated-syndrome BP probes with measurement faults and possible opposite-check spread subsets. Settled residuals are classified modulo actual stabilizers; unresolved shots measure decoder difficulty. Independent six-round validation limits training-sample selection. These probes omit ASC ordering, relay BP and LSD and are not measured pL. Parent slots4/6 shared n,k,d2s,w and depth, yet Z failures were 100/66750 versus 53/200000; target collective decodability rather than distance alone.")
FAMILY_NAME = "r030_Q8xC5_q40_12x26_joint_history_validation_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 40

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Q8 x C5 requires order 40")
    # (a,b,c) = i^a j^b z^c.
    # i^4=1, j^2=i^2, j*i*j^-1=i^-1; z is central of order five.
    els = [(a, b, c) for c in range(5)
           for b in range(2) for a in range(4)]

    def red(g):
        return (int(g[0]) % 4, int(g[1]) % 2, int(g[2]) % 5)

    def mul(g, h):
        a, b, c = red(g)
        d, e, f = red(h)
        return ((a + (d if b == 0 else -d) + 2*b*e) % 4,
                (b + e) % 2, (c + f) % 5)

    def inv(g):
        a, b, c = red(g)
        return (((-a) % 4 if b == 0 else (a + 2) % 4),
                b, (-c) % 5)

    def idx(g):
        a, b, c = red(g)
        return a + 4*b + 8*c

    return dict(elements=els, ident=(0, 0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (2, 6)               # (rb, cb)


# Five architectural contracts, each with B-row totals (9,9).
# The degree-two/four controls retain the same total gate count.
_B_PROFILES = (
    ((2,1),(2,1),(2,1),(1,2),(1,2),(1,2)),
    ((3,0),(0,3),(2,1),(2,1),(1,2),(1,2)),
    ((3,0),(3,0),(0,3),(0,3),(2,1),(1,2)),
    ((1,1),(2,2),(2,1),(2,1),(1,2),(1,2)),
    ((0,2),(2,0),(2,2),(2,2),(2,1),(1,2)),
)


def _component_rows(G, side, grid):
    """Extract the component action from the verified construction.

    The opposite side is a zero 1x1 probe, not a submitted component.
    This preserves the evaluator's left/right conventions exactly.
    """
    if side == "A":
        rows = lifted_matrices(G, grid, [[[]]])[0]
    else:
        rows = lifted_matrices(G, [[[]]], grid)[1]
    mask = (1 << (len(grid[0]) * G.q)) - 1
    return [row & mask for row in rows]


def _cycle_counts(rows):
    """Exact component Tanner four- and six-cycle counts."""
    size = len(rows)
    intersections = [[0] * size for _ in range(size)]
    counts = [[0] * size for _ in range(size)]
    four = 0
    for i in range(size):
        for j in range(i + 1, size):
            common = rows[i] & rows[j]
            number = common.bit_count()
            intersections[i][j] = common
            counts[i][j] = number
            four += number * (number - 1) // 2
    six = 0
    for i in range(size):
        for j in range(i + 1, size):
            a = counts[i][j]
            if not a:
                continue
            for k in range(j + 1, size):
                b, c = counts[i][k], counts[j][k]
                if b and c:
                    common = (intersections[i][j] & rows[k]).bit_count()
                    # Count distinct variable choices, excluding stars.
                    six += a*b*c - common*(a+b+c) + 2*common
    return four, six


def _short_collision_counts(rows, columns):
    """Cheap component-codeword screen through weight four.

    Pair collisions with overlapping supports can overcount when
    weight-one/two words exist; those are already penalized first.
    """
    syndromes = [0] * columns
    for check, row in enumerate(rows):
        while row:
            bit = row & -row
            syndromes[bit.bit_length() - 1] |= 1 << check
            row ^= bit
    singles = {}
    one = two = 0
    for syndrome in syndromes:
        one += int(syndrome == 0)
        two += singles.get(syndrome, 0)
        singles[syndrome] = singles.get(syndrome, 0) + 1
    pairs = {}
    three = four = 0
    for i in range(columns):
        for j in range(i + 1, columns):
            syndrome = syndromes[i] ^ syndromes[j]
            three += singles.get(syndrome, 0)
            four += pairs.get(syndrome, 0)
            pairs[syndrome] = pairs.get(syndrome, 0) + 1
    return one, two, three, four


def _copy_cells(grid):
    return [[list(cell) for cell in row] for row in grid]


def _draw_design_cell(rng, G, count, model, orders, triangles,
                      double_order=20):
    """Fresh relative-support geometry, never a saved-code orbit image."""
    if count == 0:
        return []
    anchor = G.pick(rng)
    if count == 1:
        return [anchor]
    if count == 2:
        relative = [G.e, rng.choice(orders[double_order])]
    elif count == 3:
        u, v = rng.choice(triangles[model])
        relative = [G.e, u, v]
    else:
        raise ValueError("unsupported quaternion contract cell")
    cell = [G.mul(anchor, g) for g in relative]
    assert len({G._key(g) for g in cell}) == count
    return cell


def _triple_boundary_metric(rows, width):
    """Small triple boundaries and syndrome collisions.

    Collisions are a multiplicity heuristic, not an exact spectrum
    when shorter kernels exist. Boundary penalties are uncalibrated
    decoder-risk surrogates, never a replacement for circuit pL.
    """
    columns = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    histogram = {}
    collisions = boundary = zero = 0
    for i in range(width):
        for j in range(i):
            pair = columns[i] ^ columns[j]
            for h in range(j):
                syndrome = pair ^ columns[h]
                weight = syndrome.bit_count()
                zero += int(weight == 0)
                if 0 < weight < 4:
                    boundary += (4 - weight) ** 2
                previous = histogram.get(syndrome, 0)
                collisions += previous
                histogram[syndrome] = previous + 1
    return zero, boundary, collisions


def _design_component(rng, G, side, support, draw_cell, actions):
    """Optimize exact helper-derived actions inside one design contract."""
    import math

    q = G.q
    nr, nc = len(support), len(support[0])
    width = nc*q
    positions = [(i, j) for i in range(nr) for j in range(nc)
                 if support[i][j]]

    def rows_of(grid):
        result = []
        for row in grid:
            lifted = [0]*q
            for j, cell in enumerate(row):
                for g in cell:
                    block = actions[G.idx(g)]
                    for r, word in enumerate(block):
                        lifted[r] ^= word << (j*q)
            result.extend(lifted)
        return result

    def columns(rows):
        cols = [0]*width
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                cols[bit.bit_length()-1] |= 1 << r
                row ^= bit
        return cols

    def analyze(grid):
        rows = rows_of(grid)
        cols = columns(rows)
        size = len(rows)
        adjacency = [0]*size
        overlaps = {}
        stars = 0
        for col in cols:
            ns = []
            while col:
                bit = col & -col
                ns.append(bit.bit_length()-1)
                col ^= bit
            d = len(ns)
            stars += d*(d-1)*(d-2)//6
            for a in range(d):
                for b in range(a):
                    i, j = ns[b], ns[a]
                    overlaps[i, j] = overlaps.get((i, j), 0)+1
                    adjacency[i] |= 1 << j
                    adjacency[j] |= 1 << i
        four = sum(v*(v-1)//2 for v in overlaps.values())
        triangles = 0
        for i in range(size):
            later = adjacency[i] >> (i+1)
            while later:
                bit = later & -later
                j = i+1+bit.bit_length()-1
                triangles += ((adjacency[i] & adjacency[j])
                              >> (j+1)).bit_count()
                later ^= bit
        # Triangle-minus-stars is exact without four-cycles;
        # otherwise it is only a six-cycle optimization surrogate.
        six = max(0, triangles-stars)
        unseen = (1 << size)-1
        parts = 0
        while unseen:
            todo = unseen & -unseen
            unseen ^= todo
            parts += 1
            while todo:
                bit = todo & -todo
                todo ^= bit
                neighbors = adjacency[bit.bit_length()-1] & unseen
                unseen ^= neighbors
                todo |= neighbors
        energy = (80*four + six/8.0)/q + 100*(parts-1)
        return energy, rows, cols, parts, four, six

    def boundary(cols):
        """Anchored triple exposure; not an exact weight-six spectrum."""
        pair_syndromes = {0}
        pair_syndromes.update(cols)
        for i, x in enumerate(cols):
            for y in cols[:i]:
                pair_syndromes.add(x ^ y)
        matches = burden = 0
        for anchor in range(0, width, q):
            for i, x in enumerate(cols):
                if i == anchor:
                    continue
                pair = cols[anchor] ^ x
                for j in range(i):
                    if j == anchor:
                        continue
                    syndrome = pair ^ cols[j]
                    matches += int(syndrome in pair_syndromes)
                    weight = syndrome.bit_count()
                    if weight < 5:
                        burden += (5-weight)**2
        return matches/nc, burden/nc

    def kernel_sample(cols):
        """Actual sampled kernel witnesses; minimum is only an upper bound."""
        order = list(range(width))
        minimum = width+1
        small = set()
        for _ in range(40):
            rng.shuffle(order)
            pivots = {}
            basis = []
            for c in order:
                syndrome, word = cols[c], 1 << c
                while syndrome:
                    p = syndrome.bit_length()-1
                    previous = pivots.get(p)
                    if previous is None:
                        pivots[p] = syndrome, word
                        break
                    syndrome ^= previous[0]
                    word ^= previous[1]
                if not syndrome:
                    basis.append(word)
                    weight = word.bit_count()
                    minimum = min(minimum, weight)
                    if weight <= 10:
                        small.add(word)
            for i, word in enumerate(basis):
                for other in basis[:i]:
                    value = word ^ other
                    weight = value.bit_count()
                    minimum = min(minimum, weight)
                    if weight <= 10:
                        small.add(value)
            if minimum <= 4:
                break
        return minimum, len(small)/width

    proposals = []
    for restart in range(4):
        grid = [[draw_cell(i, j) for j in range(nc)]
                for i in range(nr)]
        current = analyze(grid)
        best_grid = _copy_cells(grid)
        best_energy = current[0]
        for step in range(160):
            i, j = rng.choice(positions)
            old = grid[i][j]
            grid[i][j] = draw_cell(i, j)
            proposed = analyze(grid)
            delta = proposed[0]-current[0]
            temperature = 18*(0.976**step)+0.2
            if delta <= 0 or rng.random() < math.exp(
                    -min(700.0, delta/temperature)):
                current = proposed
                if current[0] < best_energy:
                    best_grid = _copy_cells(grid)
                    best_energy = current[0]
            else:
                grid[i][j] = old
            if step in (79, 159):
                proposals.append(_copy_cells(grid))
        proposals.append(best_grid)

    pool, seen = [], set()
    for serial, grid in enumerate(proposals):
        signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                          for row in grid for cell in row)
        if signature in seen:
            continue
        seen.add(signature)
        _, rows, cols, parts, four, six = analyze(grid)
        deficit = len(rows)-gf2_rank(rows)
        short = _short_collision_counts(rows, width)
        cheap = short + (parts-1, deficit, four/q, six/q, serial)
        pool.append((cheap, grid, rows, cols))
    pool.sort(key=lambda item: item[0])

    finalists = []
    for cheap, grid, rows, cols in pool[:4]:
        matches, burden = boundary(cols)
        distance, multiplicity = kernel_sample(cols)
        key = (cheap[:6], int(distance < 10), matches, burden,
               multiplicity, cheap[6], cheap[7],
               -min(distance, 12), cheap[8])
        # Confirm cached linearity against the complete verified probe.
        assert rows == _component_rows(G, side, grid)
        finalists.append((key, grid))
    finalists.sort(key=lambda item: item[0])
    if not finalists:
        raise ValueError("quaternion component search produced no candidate")
    return finalists[:3]


def NEW_CTX(rng, G, idx):
    """Twenty algebraic/topological contracts and joint noisy-history selection."""
    from itertools import combinations
    import numpy as np

    q = G.q
    assert q == 40
    design, batch = idx % 20, idx // 20
    profile, policy = divmod(design, 4)

    # Every column has degree three, with balanced nine-term rows.
    # Include one- and three-zero architectures, not degree-two/four
    # controls that introduce weaker variable classes at the same load.
    profiles = (
        ((2,1),(2,1),(2,1),(1,2),(1,2),(1,2)),
        ((3,0),(0,3),(2,1),(2,1),(1,2),(1,2)),
        ((3,0),(3,0),(0,3),(0,3),(2,1),(1,2)),
        ((0,3),(1,2),(2,1),(2,1),(2,1),(2,1)),
        ((0,3),(0,3),(2,1),(2,1),(2,1),(3,0)),
    )
    sa = [[3, 3]]
    sb = [[column[i] for column in profiles[profile]]
          for i in range(2)]
    assert _no_dead_lines(sa, sb)
    assert all(sum(row) == 9 for row in sb)
    assert all(sum(column) == 3 for column in zip(*sb))
    assert _grid_weights(sa, sb) == 12 <= W_MAX

    element_orders = {}
    for g in G.elements:
        value = G.e
        for order in range(1, q+1):
            value = G.mul(value, g)
            if value == G.e:
                element_orders[G.idx(g)] = order
                break
        assert G.idx(g) in element_orders
    orders = {
        order: [g for g in G.elements
                if element_orders[G.idx(g)] == order]
        for order in (4, 5, 10, 20)
    }
    assert all(orders.values())

    triangles = {0: [], 1: []}
    nonidentity = [g for g in G.elements if g != G.e]
    for u, v in combinations(nonidentity, 2):
        if G.mul(u, v) == G.mul(v, u):
            continue
        relative = G.mul(G.inv(u), v)
        spectrum = sorted((element_orders[G.idx(u)],
                           element_orders[G.idx(v)],
                           element_orders[G.idx(relative)]))
        if spectrum == [20, 20, 20]:
            triangles[0].append((u, v))
        elif spectrum == [4, 20, 20]:
            triangles[1].append((u, v))
    assert all(triangles.values())

    a_types = ((0,0), (0,1), (1,1), (0,1))[policy]
    if batch % 3 == 1:
        a_types = (1, a_types[1])
    elif batch % 3 == 2:
        a_types = (a_types[0], 1)

    bin_sites = [(i, j) for i in range(2) for j in range(6)
                 if sb[i][j] == 2]
    shift = (profile+2*batch) % len(bin_sites)
    sites = bin_sites[shift:] + bin_sites[:shift]
    schedule = {site: 20 for site in sites}
    if policy == 1:
        for site in sites[:2]:
            schedule[site] = 10
    elif policy == 2:
        for site in sites[:2]:
            schedule[site] = 4
    elif policy == 3:
        schedule[sites[0]] = 5
        schedule[sites[-1]] = 10

    def draw_a(i, j):
        # Search internal supports, not whole-cell origin gauges.
        # No completed code is produced by transforming an earlier code.
        u, v = rng.choice(triangles[a_types[j]])
        return [G.e, u, v]

    def draw_b(i, j):
        count = sb[i][j]
        model = int((policy+i+j+batch) % 3 == 0)
        return _draw_design_cell(
            rng, G, count, model, orders, triangles,
            schedule.get((i, j), 20))

    # Cache singleton blocks ONLY through the verified construction.
    mask = (1 << q)-1
    actions = {"A": {}, "B": {}}
    for g in G.elements:
        hx, _, _ = lifted_matrices(G, [[[g]]], [[[]]])
        _, hz, _ = lifted_matrices(G, [[[]]], [[[g]]])
        actions["A"][G.idx(g)] = tuple(row & mask for row in hx)
        actions["B"][G.idx(g)] = tuple(row & mask for row in hz)

    def columns(rows, width):
        result = [0]*width
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                result[bit.bit_length()-1] |= tag
                row ^= bit
        return result

    def quotient(stabilizers, n):
        """Linear canonical residues modulo verified stabilizer rows."""
        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = row
                    break
                row ^= old
        residues = [0]*n
        for c in range(n):
            row = pivots.get(c)
            if row is None:
                residues[c] = 1 << c
                continue
            tail, value = row ^ (1 << c), 0
            while tail:
                bit = tail & -tail
                value ^= residues[bit.bit_length()-1]
                tail ^= bit
            residues[c] = value
        return residues

    def short_logical(checks, stabilizers, n):
        """Exact non-stabilizer logical existence through weight four."""
        cols = columns(checks, n)
        residues = quotient(stabilizers, n)
        seen = {0: 0}
        for c, syndrome in enumerate(cols):
            value = residues[c]
            old = seen.get(syndrome)
            if old is not None and old != value:
                return True
            seen[syndrome] = value
        for c, syndrome in enumerate(cols):
            value = residues[c]
            for d in range(c):
                s = syndrome ^ cols[d]
                label = value ^ residues[d]
                old = seen.get(s)
                if old is not None and old != label:
                    return True
                seen[s] = label
        return False

    def connected(rows, n):
        parent = list(range(n))

        def root(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for row in rows:
            if not row:
                return False
            bit = row & -row
            anchor = bit.bit_length()-1
            row ^= bit
            while row:
                bit = row & -row
                a, b = root(anchor), root(bit.bit_length()-1)
                if a != b:
                    parent[b] = a
                row ^= bit
        return len({root(c) for c in range(n)}) == 1

    def sites_of(row):
        result = []
        while row:
            bit = row & -row
            result.append(bit.bit_length()-1)
            row ^= bit
        return result

    def history_probe(checks, stabilizers, n, seed, rounds, shots):
        """Noisy-history min-sum surrogate, NOT an ASC memory experiment.

        Variables are incremental data faults and measurement faults.
        Detector rows implement H*e_t + m_t + m_(t-1), followed by
        a perfect final syndrome boundary. Possible opposite-check
        spread subsets are sampled without claiming an ASC hook order.
        Settled final data residuals are classified modulo stabilizers.
        Relay BP, LSD and the scored circuit ordering are not modeled.
        """
        channel = np.random.default_rng(seed)
        spatial = [sites_of(row) for row in checks]
        spread_rows = [sites_of(row) for row in stabilizers]
        nr = len(spatial)
        data_width = rounds*n
        nv = rounds*(n+nr)

        neighborhoods = []
        for t in range(rounds):
            for r, row in enumerate(spatial):
                ns = [t*n+c for c in row]
                ns.append(data_width+t*nr+r)
                if t:
                    ns.append(data_width+(t-1)*nr+r)
                neighborhoods.append(ns)
        neighborhoods.extend(
            [[data_width+(rounds-1)*nr+r] for r in range(nr)])

        nc = len(neighborhoods)
        ne = sum(map(len, neighborhoods))
        max_check = max(map(len, neighborhoods))
        check_edges = np.full((nc, max_check), ne, dtype=np.int32)
        check_sites = np.zeros((nc, max_check), dtype=np.int32)
        valid = np.zeros((nc, max_check), dtype=bool)
        edge_variables = []
        offset = 0
        for r, row in enumerate(neighborhoods):
            degree = len(row)
            check_edges[r, :degree] = np.arange(offset, offset+degree)
            check_sites[r, :degree] = row
            valid[r, :degree] = True
            edge_variables.extend(row)
            offset += degree
        edge_variables = np.asarray(edge_variables, dtype=np.int32)
        degrees = np.bincount(edge_variables, minlength=nv)
        assert np.all(degrees > 0)
        variable_edges = np.full(
            (nv, int(degrees.max())), ne, dtype=np.int32)
        used = np.zeros(nv, dtype=np.int32)
        for edge, c in enumerate(edge_variables):
            variable_edges[c, used[c]] = edge
            used[c] += 1

        sensed = columns(checks, n)
        spread = columns(stabilizers, n)
        gates = np.asarray(
            [a.bit_count()+b.bit_count()
             for a, b in zip(sensed, spread)], dtype=float)
        hook_exposure = np.zeros(n, dtype=float)
        for row in spread_rows:
            hook_exposure[row] += len(row)

        # Elevated, mixed stresses provide decoder-difficulty signal;
        # these rates are not a calibration of the ASC circuit channel.
        levels = np.asarray((0.75, 1.0, 1.35, 1.8))
        level = levels[np.arange(shots) % len(levels), None]
        raw_data = level*(0.0085*gates[None, :]/6.0)
        data_prior = np.minimum(
            0.25, raw_data+level*0.00025*hook_exposure[None, :])
        check_weight = np.asarray(list(map(len, spatial)), dtype=float)
        measure_rate = np.minimum(
            0.20, level*(0.003+0.00065*check_weight[None, :]))

        errors = np.zeros((shots, nv), dtype=bool)
        errors[:, :data_width] = (
            channel.random((shots, data_width))
            < np.tile(raw_data, (1, rounds)))
        errors[:, data_width:] = (
            channel.random((shots, rounds*nr))
            < np.tile(measure_rate, (1, rounds)))

        # Random suffix subsets emulate potential first-order spread,
        # but are deliberately not represented as certified ASC hooks.
        hook_order = [
            channel.permutation(row).tolist() for row in spread_rows]
        hook_weight = np.asarray(list(map(len, spread_rows)), dtype=float)
        trigger_rate = np.minimum(
            0.08, level[:, :, None]*0.0005*hook_weight[None, None, :])
        triggers = (
            channel.random((shots, rounds, len(spread_rows)))
            < trigger_rate)
        for shot, t, r in zip(*np.nonzero(triggers)):
            row = hook_order[int(r)]
            if len(row) < 2:
                continue
            start = int(channel.integers(1, len(row)))
            variables = np.asarray(row[start:], dtype=np.int32)+int(t)*n
            errors[int(shot), variables] ^= True

        prior = np.concatenate(
            (np.tile(data_prior, (1, rounds)),
             np.tile(measure_rate, (1, rounds))), axis=1)
        llr = np.log((1.0-prior)/prior).astype(np.float32)
        target = np.logical_xor.reduce(
            errors[:, check_sites] & valid[None, :, :], axis=2)
        target_sign = np.where(target, -1.0, 1.0).astype(np.float32)
        messages = llr[:, edge_variables].copy()
        outgoing = np.zeros((shots, ne+1), dtype=np.float32)
        done = np.zeros(shots, dtype=bool)
        decoded_final = np.zeros((shots, n), dtype=bool)
        iterations = 48 if rounds == 6 else 36
        completion = np.full(shots, iterations+1, dtype=np.int32)
        positions = np.arange(max_check)[None, None, :]

        for iteration in range(iterations):
            if iteration == 24:
                # A second damped pass is a difficulty probe, not relay BP.
                jitter = channel.normal(
                    0.0, 0.08, messages.shape).astype(np.float32)
                messages = 0.85*llr[:, edge_variables]+jitter

            padded = np.concatenate(
                (messages, np.zeros((shots, 1), dtype=np.float32)), axis=1)
            shaped = padded[:, check_edges]
            magnitude = np.where(
                valid[None, :, :], np.abs(shaped), np.inf)
            signs = np.where(shaped < 0.0, -1.0, 1.0).astype(np.float32)
            product = signs.prod(axis=2)*target_sign
            minimum_at = magnitude.argmin(axis=2)
            smallest = np.partition(magnitude, 1, axis=2)[:, :, :2]
            size = np.where(
                positions == minimum_at[:, :, None],
                smallest[:, :, 1, None], smallest[:, :, 0, None])
            size = np.minimum(size, 24.0)
            update = 0.80*product[:, :, None]*signs*size
            update = np.where(valid[None, :, :], update, 0.0)
            outgoing[:, :ne] = update[:, valid]
            posterior = llr+outgoing[:, variable_edges].sum(axis=2)
            hard = posterior < 0.0
            parity = np.logical_xor.reduce(
                hard[:, check_sites] & valid[None, :, :], axis=2)
            newly = ~done & ~np.any(parity != target, axis=1)
            if np.any(newly):
                decoded_final[newly] = np.logical_xor.reduce(
                    hard[newly, :data_width].reshape(-1, rounds, n),
                    axis=1)
                completion[newly] = iteration+1
                done |= newly
            if np.all(done):
                break
            proposal = posterior[:, edge_variables]-outgoing[:, :ne]
            messages = (
                0.30*messages+0.70*np.clip(proposal, -24.0, 24.0))

        actual_final = np.logical_xor.reduce(
            errors[:, :data_width].reshape(shots, rounds, n), axis=1)
        residues = quotient(stabilizers, n)
        failed = ~done
        residual_weight = 0
        for shot in np.flatnonzero(done):
            difference = decoded_final[shot] ^ actual_final[shot]
            label = 0
            for c in np.flatnonzero(difference):
                label ^= residues[int(c)]
            if label:
                failed[shot] = True
                residual_weight += int(np.count_nonzero(difference))
        return (int(failed.sum()), int((~done).sum()),
                int(completion.sum()), residual_weight)

    def probe_pair(hx, hz, n, seed, rounds, shots):
        x = history_probe(hz, hx, n, seed, rounds, shots)
        z = history_probe(hx, hz, n, seed+1, rounds, shots)
        return x, z

    def probe_key(pair):
        x, z = pair
        return (max(x[0], z[0]), x[0]+z[0],
                max(x[1], z[1]), x[1]+z[1],
                max(x[2], z[2]), x[2]+z[2],
                x[3]+z[3])

    fallback = None
    for attempt in range(2):
        aa = _design_component(rng, G, "A", sa, draw_a, actions["A"])
        bb = _design_component(rng, G, "B", sb, draw_b, actions["B"])
        options = []
        for ia, (ak, A) in enumerate(aa):
            for ib, (bk, B) in enumerate(bb):
                # Protect the weaker component before aggregate rank.
                coarse = (max(ia, ib), ia+ib, ak, bk)
                options.append((coarse, ia, ib, A, B))
        options.sort(key=lambda item: item[:3])

        candidates = []
        identities = set()
        for coarse, ia, ib, A, B in options:
            signature = tuple(
                tuple(sorted(G.idx(g) for g in cell))
                for grid in (A, B) for row in grid for cell in row)
            if signature in identities:
                continue
            identities.add(signature)
            hx, hz, n = lifted_matrices(G, A, B)
            k = n-gf2_rank(hx)-gf2_rank(hz)
            if not (n == 560 and 160 <= k < 182):
                continue
            if not connected(hx+hz, n):
                continue
            assert css_ok(hx, hz)
            assert max(row.bit_count() for row in hx) == 9
            assert max(row.bit_count() for row in hz) == 12
            assert 6*sum(row.bit_count() for row in hx+hz) == 24480
            assert all(len(set(cell)) == len(cell)
                       for grid in (A, B) for row in grid for cell in row)
            for cell in A[0]:
                assert gf2_rank(
                    lifted_matrices(G, [[cell]], [[[]]])[0]) == q

            candidate = {
                "idx": idx, "cells": {"A": A, "B": B}, "k": k,
                "brief": (profile, policy, batch),
                "component_screen": coarse,
                "small_logical_safe": False,
            }
            if fallback is None:
                fallback = candidate
            if (short_logical(hx, hz, n)
                    or short_logical(hz, hx, n)):
                continue
            candidate["small_logical_safe"] = True
            candidates.append((coarse, candidate, hx, hz, n))
            if len(candidates) >= 4:
                break

        if not candidates:
            continue

        training = []
        seed = 40173029+1009*idx
        for coarse, candidate, hx, hz, n in candidates:
            result = probe_pair(hx, hz, n, seed, 3, 48)
            key = probe_key(result)
            candidate["history_training"] = result
            training.append((key, coarse, candidate, hx, hz, n))
        training.sort(key=lambda item: item[:2])

        validated = []
        for train_key, coarse, candidate, hx, hz, n in training[:2]:
            # A separate channel stream and longer history limit choosing
            # a filling solely because it won a small training sample.
            result = probe_pair(hx, hz, n, seed+104729, 6, 64)
            validation_key = probe_key(result)
            key = (validation_key[:4], train_key[:4],
                   validation_key[4:], train_key[4:], coarse)
            candidate["history_validation"] = result
            candidate["joint_screen"] = key
            validated.append((key, candidate))
        validated.sort(key=lambda item: item[0])
        return validated[0][1]

    # Optional decoder/protection preferences must not void a legal code.
    if fallback is not None:
        return fallback
    raise ValueError("no verified joint-history filling in cell 0.300")


def CELL(rng, G, side, i, j, ctx):
    """Return the already-screened complete cell, without mutation."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: A d2s=6 screen does not ensure balanced circuit protection: parent slot 56 had 107/1250 Z failures but only 53/200000 X failures.
# REFUTED: Equal n,k,d2s and symmetric component shapes fix worse-basis reliability: current parent slots 0 and 4 both had (580,180,8), yet scored 5.782 and 5.402 with opposite limiting bases.
# REFUTED: Lower-weight equal three-row C10 components suffice to improve circuit memory: measured fitness fell from 5.5627 to 5.4816, and all five measured codes reached failed-shot caps.
# REFUTED: Equal n=560,k=160,d2s=10,w=12 and ASC depth15 imply comparable circuit reliability: current slots4/6 had 100/66750 versus 53/200000 Z failures and worse-basis scores 5.721 versus 6.218.
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
