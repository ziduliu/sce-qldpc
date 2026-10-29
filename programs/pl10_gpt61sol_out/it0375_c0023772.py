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
CONCEPT = ("For Q8 x C5, order 40, shapes (1,2)x(2,5), a noncommuting trinomial/quartic A component and degree-three B columns should reduce Z-limited collective circuit failures while retaining measured rate in cell 0.250. The test is circuit pL, not a distance upper bound.")
GROUP = ("Keep the exact Q8 x C5 direct-product law, order 40 and element encoding under L2. Its current (1,2)x(2,6) program scored 6.0617. Keeping the group isolates the shape/filling experiment; the group has a unique involution and relative orders 4,10,20 without requiring involutory cell differences. No group-only performance advantage is asserted.")
SHAPES = ("Change B from (2,6) to (2,5), retaining A=(1,2). At q=40, n=480 and kb=120 give guaranteed rate 0.250; require complete verified 120<=k<=131. Components have 80 variables/40 checks and 200 variables/80 checks. A column degrees three/four and B column degree three give HX weight ten and HZ weights ten through twelve. Parent Z failures diffusely enriched Acol/Bcol and Acol/Brow blocks around 1.0-1.1x; the maps do not identify a stable isolated repair.")
FILLING = ("Twenty deliberate contracts cross four seven/eight-term B row architectures with five quartic central-phase and binomial-order policies. A weights are (3,4), with distinct quaternion axes in the quartic and noncommuting trinomials. All internal differences exclude the unique involution. Fresh internal-cell redraws search geometry rather than only single-row A origin gauges. Verified singleton actions screen connectivity, ranks, short cycles, exact component kernels through six, capped sampled witnesses and anchored triple boundaries. Full helper-built CSS matrices enforce the target rate and prefer no quantum logical through four. Common-noise irregular-check BP probes classify settled residuals modulo stabilizers; independent holdout streams refine finalists. These omit ASC scheduling, measurement histories and the relay/LSD cascade. Six-round CNOTs fall from 24480 to 22560. The target is fewer collective Z failures than the parent's 67-101 observed failures per measured code, not a larger d2s alone.")
FAMILY_NAME = "r025_Q8xC5_12x25_triquartic_holdoutbp_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 40

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Q8 x C5 requires q=40")
    # (t,s,a) represents z^t * (-1)^s * e_a.
    # z has order five; e_0=1,e_1=i,e_2=j,e_3=k in Q8.
    # i*j=k, j*k=i, k*i=j; reversed products change sign.
    neg = ((0,0,0,0), (0,1,0,1),
           (0,1,1,0), (0,0,1,1))
    els = [(t,s,a) for t in range(5)
           for s in range(2) for a in range(4)]

    def red(g):
        return (int(g[0]) % 5, int(g[1]) % 2, int(g[2]) % 4)

    def mul(g, h):
        t,s,a = red(g)
        u,v,b = red(h)
        return ((t+u) % 5, s ^ v ^ neg[a][b], a ^ b)

    def inv(g):
        t,s,a = red(g)
        return ((-t) % 5, s ^ int(a != 0), a)

    def idx(g):
        t,s,a = red(g)
        return 8*t+4*s+a

    return dict(elements=els, ident=(0,0,0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


def _columns(rows, width):
    result = [0] * width
    for r, row in enumerate(rows):
        tag = 1 << r
        while row:
            bit = row & -row
            result[bit.bit_length() - 1] |= tag
            row ^= bit
    return result


def _rows_from_columns(columns, height):
    rows = [0] * height
    for c, syndrome in enumerate(columns):
        tag = 1 << c
        while syndrome:
            bit = syndrome & -syndrome
            rows[bit.bit_length() - 1] |= tag
            syndrome ^= bit
    return rows


def _component_cycles(columns):
    """Four-cycles from shared check pairs; also return check adjacency."""
    pairs = {}
    adjacency = {}
    cycles = 0
    for syndrome in columns:
        nodes = []
        while syndrome:
            bit = syndrome & -syndrome
            nodes.append(bit.bit_length() - 1)
            syndrome ^= bit
        for a, u in enumerate(nodes):
            for v in nodes[:a]:
                pair = (v, u)
                old = pairs.get(pair, 0)
                cycles += old
                pairs[pair] = old + 1
                adjacency[u] = adjacency.get(u, 0) | (1 << v)
                adjacency[v] = adjacency.get(v, 0) | (1 << u)
    return cycles, adjacency


def _component_summary(columns, adjacency):
    """Exact weight-four kernels when columns are distinct and nonzero."""
    duplicates = len(columns) - len(set(columns)) + int(0 in columns)
    pairs = {}
    collisions = 0
    for a, x in enumerate(columns):
        for y in columns[:a]:
            syndrome = x ^ y
            old = pairs.get(syndrome, 0)
            collisions += old
            pairs[syndrome] = old + 1

    triangles = 0
    for u, neighbors in adjacency.items():
        higher = neighbors & ~((1 << (u + 1)) - 1)
        while higher:
            bit = higher & -higher
            v = bit.bit_length() - 1
            triangles += (
                neighbors & adjacency.get(v, 0)
                & ~((1 << (v + 1)) - 1)
            ).bit_count()
            higher ^= bit
    # Exact Tanner six-cycles when four-cycles are absent; otherwise
    # this is only a secondary adjacency statistic.
    six_cycles = max(0, triangles - len(columns))
    return duplicates, collisions // 3, six_cycles


def _six_kernel_count(columns):
    """Exact weight-six count, called only without weight-two/four words."""
    histogram = {}
    collisions = 0
    width = len(columns)
    for a in range(width - 2):
        x = columns[a]
        for b in range(a + 1, width - 1):
            ab = x ^ columns[b]
            for c in range(b + 1, width):
                syndrome = ab ^ columns[c]
                old = histogram.get(syndrome, 0)
                collisions += old
                histogram[syndrome] = old + 1
    # Colliding triples are disjoint under the precondition.
    return collisions // 10


def _small_logical(checks, stabilizers, n):
    """Exact existence test for a nontrivial logical of weight <=4."""
    pivots = {}
    for row in stabilizers:
        while row:
            p = row.bit_length() - 1
            old = pivots.get(p)
            if old is None:
                pivots[p] = row
                break
            row ^= old

    # Linear quotient representatives of unit vectors. Each pivot row
    # only uses lower-numbered coordinates, so ascending substitution
    # computes their full stabilizer-row-space remainders.
    residues = [1 << c for c in range(n)]
    for p in sorted(pivots):
        tail = pivots[p] ^ (1 << p)
        value = 0
        while tail:
            bit = tail & -tail
            value ^= residues[bit.bit_length() - 1]
            tail ^= bit
        residues[p] = value

    columns = _columns(checks, n)
    seen = {0: 0}
    for c, syndrome in enumerate(columns):
        value = residues[c]
        old = seen.get(syndrome)
        if old is not None and old != value:
            return True
        seen[syndrome] = value

    for a, x in enumerate(columns):
        ra = residues[a]
        for b in range(a):
            syndrome = x ^ columns[b]
            value = ra ^ residues[b]
            old = seen.get(syndrome)
            if old is not None and old != value:
                return True
            seen[syndrome] = value
    return False


def _legacy_order27_ctx(rng, G, idx):
    """Inactive historical order-27 search; new slots never call it."""
    import math

    q = G.q
    assert q == 27
    profiles = ((2, 2, 1, 1), (0, 2, 2, 2), (3, 0, 2, 1))
    support_pairs = ((0, 0), (0, 1), (1, 1), (0, 2))
    direction_cards = ((1, 1), (1, 2), (2, 2), (2, 3), (3, 3))
    architecture = (idx // 5) % 4
    pa, pb = support_pairs[architecture]
    counts = {}
    for side, p in (("A", pa), ("B", pb)):
        first = profiles[p]
        counts[side] = [list(first), [3 - t for t in first]]
    assert _no_dead_lines(counts["A"], counts["B"])
    assert _grid_weights(counts["A"], counts["B"]) == 9
    assert 9 <= W_MAX
    assert all(sum(row) == 6
               for grid in counts.values() for row in grid)

    elements = list(G.elements)

    def element_order(g):
        value = G.e
        for t in range(1, q + 1):
            value = G.mul(value, g)
            if value == G.e:
                return t
        raise ValueError("element order failed")

    orders = {g: element_order(g) for g in elements}
    nine = [g for g in elements if orders[g] == 9]
    three = [g for g in elements if orders[g] == 3]
    assert len(nine) == 18 and len(three) == 8

    def direction(g):
        # Projective direction in G/Z(G)=C3 x C3.
        # Order-nine elements have nonzero first quotient coordinate.
        return g[1] * pow(g[0] % 3, -1, 3) % 3

    by_direction = {
        d: [g for g in nine if direction(g) == d] for d in range(3)
    }
    triangles = [
        (u, v) for u in nine for v in nine
        if orders[G.mul(G.inv(u), v)] == 9
        and G.mul(u, v) != G.mul(v, u)
    ]
    assert triangles

    # Enumerate automorphisms from images of the presentation generators.
    # These maps classify the geometry book, not a substitute lift.
    automorphisms = set()
    for u in nine:
        up = [G.e]
        for _ in range(8):
            up.append(G.mul(up[-1], u))
        for v in three:
            if G.mul(G.mul(v, u), G.inv(v)) != up[4]:
                continue
            vp = (G.e, v, G.mul(v, v))
            image = tuple(G.idx(G.mul(up[a], vp[b]))
                          for b in range(3) for a in range(9))
            if len(set(image)) == q:
                automorphisms.add(image)
    automorphisms = sorted(automorphisms)
    assert automorphisms

    def normalized(cell, image):
        if not cell:
            return ()
        values = [elements[image[G.idx(g)]] for g in cell]
        return min(tuple(sorted(G.idx(G.mul(G.inv(anchor), g))
                                for g in values))
                   for anchor in values)

    def geometry_key(grids):
        images = []
        for image in automorphisms:
            sides = []
            for grid in grids:
                columns = [
                    (normalized(grid[0][j], image),
                     normalized(grid[1][j], image))
                    for j in range(4)
                ]
                sides.append(min(tuple(sorted(columns)),
                                 tuple(sorted((b, a) for a, b in columns))))
            images.append(tuple(sorted(sides)))
        return min(images)

    # Independent of offset-search RNG. Each indexed book entry changes
    # relative supports, not merely cell origins or group labels.
    grng = random.Random(270970 + 1009 * architecture)
    wanted = 5 * (idx // 20) + idx % 5
    book, seen_geometry = [], set()
    for attempt in range(40000):
        card = len(book) % 5
        grids = []
        legal = True
        for side_index, side in enumerate(("A", "B")):
            support = counts[side]
            sites = [(i, j) for i in range(2) for j in range(4)
                     if support[i][j] == 2]
            number = min(direction_cards[card][side_index], len(sites))
            palette = grng.sample(range(3), number)
            assignments = palette + [
                grng.choice(palette) for _ in range(len(sites) - number)
            ]
            grng.shuffle(assignments)
            prescribed = dict(zip(sites, assignments))
            grid = []
            for i in range(2):
                row = []
                used = set()
                for j, weight in enumerate(support[i]):
                    if weight == 0:
                        cell = []
                    elif weight == 1:
                        cell = [G.e]
                    elif weight == 2:
                        cell = [G.e, grng.choice(
                            by_direction[prescribed[i, j]])]
                    else:
                        u, v = grng.choice(triangles)
                        cell = [G.e, u, v]
                    assert len(set(cell)) == weight
                    # Avoid repeated normalized relative differences.
                    # Actual cycles are still measured from verified rows.
                    for g in cell:
                        for h in cell:
                            if g == h:
                                continue
                            difference = G.mul(G.inv(g), h)
                            if orders[difference] != 9 or difference in used:
                                legal = False
                            used.add(difference)
                    row.append(cell)
                grid.append(row)
            grids.append(grid)
        if not legal:
            continue
        signature = geometry_key(grids)
        if signature in seen_geometry:
            continue
        seen_geometry.add(signature)
        book.append(grids)
        if len(book) > wanted:
            break
    if len(book) <= wanted:
        raise ValueError("order-nine relative geometry book exhausted")
    geometry = dict(zip(("A", "B"), book[wanted]))

    # Singleton contributions come exclusively from the verified helper.
    mask = (1 << q) - 1
    templates = {"A": [], "B": []}
    for g in elements:
        hx, _, _ = lifted_matrices(G, [[[g]]], [[[]]])
        _, hz, _ = lifted_matrices(G, [[[]]], [[[g]]])
        templates["A"].append(_columns([row & mask for row in hx], q))
        templates["B"].append(_columns([row & mask for row in hz], q))

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def component_columns(side, grid):
        result = [0] * (4 * q)
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                for g in cell:
                    block = templates[side][G.idx(g)]
                    for t, syndrome in enumerate(block):
                        result[j * q + t] ^= syndrome << (i * q)
        assert all(s.bit_count() == 3 for s in result)
        return result

    def connected_component(adjacency):
        seen, todo = 0, 1
        while todo:
            bit = todo & -todo
            todo ^= bit
            if seen & bit:
                continue
            seen |= bit
            todo |= adjacency.get(bit.bit_length() - 1, 0) & ~seen
        return seen.bit_count() == 2 * q

    def sampled_kernel(columns, rows):
        """Common-stream kernel witnesses; upper bounds, not circuit pL."""
        srng = random.Random(271331 + 1009 * idx)
        order = list(range(len(columns)))
        best = len(columns) + 1
        short = set()

        def offer(word):
            nonlocal best
            if word:
                weight = word.bit_count()
                best = min(best, weight)
                if weight <= 10:
                    short.add(word)

        for _ in range(40):
            srng.shuffle(order)
            pivots, basis = {}, []
            for c in order:
                syndrome, word = columns[c], 1 << c
                while syndrome:
                    p = syndrome.bit_length() - 1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = syndrome, word
                        break
                    syndrome ^= old[0]
                    word ^= old[1]
                if not syndrome and word:
                    offer(word)
                    basis.append(word)
            srng.shuffle(basis)
            for i, word in enumerate(basis):
                weight = word.bit_count()
                for previous in basis[:i]:
                    proposal = word ^ previous
                    offer(proposal)
                    pw = proposal.bit_count()
                    if proposal and pw < weight:
                        word, weight = proposal, pw
                basis[i] = word
            if best <= 6:
                break

        assert all(all((row & word).bit_count() % 2 == 0 for row in rows)
                   for word in short)
        loads = [sum(bool((word >> (j * q)) & mask) for word in short)
                 for j in range(4)]
        return best, len(short), max(loads, default=0)

    def quality(key, columns):
        duplicates, n4, four, six_graph, deficit = key
        if duplicates or n4:
            return (1, 1, 10**9, 10**9, 10**9,
                    10**9, 10**9, four, six_graph, deficit)
        # Degree-three columns exclude odd-weight component words.
        # Without weight-two/four words, triple collisions count each
        # weight-six word ten times.
        histogram = {}
        collisions = boundary = 0
        loads = [0] * 4
        for a in range(len(columns)):
            for b in range(a):
                pair = columns[a] ^ columns[b]
                for c in range(b):
                    syndrome = pair ^ columns[c]
                    collisions += histogram.get(syndrome, 0)
                    histogram[syndrome] = histogram.get(syndrome, 0) + 1
                    weight = syndrome.bit_count()
                    if weight < 5:
                        penalty = (5 - weight) ** 2
                        boundary += penalty
                        for site in (a, b, c):
                            loads[site // q] += penalty
        assert collisions % 10 == 0
        n6 = collisions // 10
        del histogram
        rows = _rows_from_columns(columns, 2 * q)
        distance, short, peak = sampled_kernel(columns, rows)
        # No reward for sampled upper bounds above twelve.
        return (int(n6 > 0), int(distance < 12), n6, short, peak,
                max(loads), boundary, four, six_graph, deficit)

    def optimize(side):
        support = counts[side]
        # A lone nonzero cell in a column has only an origin gauge;
        # spend offset mutations on columns incident to both row orbits.
        sites = [(i, j) for i in range(2) for j in range(4)
                 if support[i][j] and support[1 - i][j]]
        pool, seen = [], set()

        def remember(grid, columns, four, adjacency, serial):
            if not connected_component(adjacency):
                return
            signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                              for row in grid for cell in row)
            if signature in seen:
                return
            seen.add(signature)
            rows = _rows_from_columns(columns, 2 * q)
            deficit = len(rows) - gf2_rank(rows)
            # Some balanced zero/trinomial profiles have an unavoidable
            # augmentation deficit. Let complete CSS ranks decide rate.
            if deficit > 2:
                return
            duplicates, n4, six_graph = _component_summary(
                columns, adjacency)
            key = (duplicates, n4, four, six_graph, deficit)
            pool.append((key, serial, clone(grid), list(columns)))
            pool.sort(key=lambda entry: (entry[0], entry[1]))
            del pool[6:]

        for restart in range(4):
            current = []
            for row in geometry[side]:
                out = []
                for cell in row:
                    origin = G.pick(rng)
                    out.append([G.mul(origin, g) for g in cell])
                current.append(out)
            columns = component_columns(side, current)
            cost, adjacency = _component_cycles(columns)
            remember(current, columns, cost, adjacency, restart * 241)
            for step in range(240):
                i, j = rng.choice(sites)
                trial = clone(current)
                shift = G.pick(rng)
                trial[i][j] = [G.mul(shift, g) for g in trial[i][j]]
                tc = component_columns(side, trial)
                trial_cost, trial_adjacency = _component_cycles(tc)
                delta = (trial_cost - cost) / q
                temperature = 3.0 * (0.98 ** step) + 0.04
                if delta <= 0 or rng.random() < math.exp(
                        -min(700.0, delta / temperature)):
                    current, columns = trial, tc
                    cost, adjacency = trial_cost, trial_adjacency
                if step % 12 == 0 and (cost == 0 or step % 60 == 0):
                    remember(current, columns, cost, adjacency,
                             restart * 241 + step + 1)
            remember(current, columns, cost, adjacency, 2000 + restart)

        if not pool:
            raise ValueError("no connected small-deficit component")
        finalists = [
            (quality(key, columns), serial, grid)
            for key, serial, grid, columns in pool
        ]
        finalists.sort(key=lambda entry: (entry[0], entry[1]))
        return finalists

    def full_connected(rows, width):
        parent = list(range(width))

        def root(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for row in rows:
            if not row:
                continue
            bit = row & -row
            anchor = bit.bit_length() - 1
            row ^= bit
            while row:
                bit = row & -row
                a, b = root(anchor), root(bit.bit_length() - 1)
                if a != b:
                    parent[b] = a
                row ^= bit
        return len({root(a) for a in range(width)}) == 1

    for batch in range(2):
        a_pool, b_pool = optimize("A"), optimize("B")
        options = []
        for aq, at, A in a_pool:
            for bq, bt, B in b_pool:
                score = tuple(value for a, b in zip(aq, bq)
                              for value in (max(a, b), a + b))
                options.append((score, at, bt, A, B))
        options.sort(key=lambda entry: entry[:3])

        fallback = None
        examined = 0
        for score, at, bt, A, B in options:
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (n == 540 and 108 <= k <= 120):
                continue
            if not full_connected(hx + hz, n):
                continue

            # Confirm cached components against the submitted matrices.
            ar = [
                sum(((hx[(i * 4) * q + t] >> (j * 4 * q)) & mask)
                    << (j * q) for j in range(4))
                for i in range(2) for t in range(q)
            ]
            br = [row & ((1 << (4 * q)) - 1) for row in hz[:2 * q]]
            assert ar == _rows_from_columns(
                component_columns("A", A), 2 * q)
            assert br == _rows_from_columns(
                component_columns("B", B), 2 * q)
            assert css_ok(hx, hz)
            assert max(row.bit_count() for row in hx + hz) == 9
            assert 6 * sum(row.bit_count() for row in hx + hz) == 23328
            candidate = {
                "A": A, "B": B, "design": idx % 20,
                "geometry": wanted, "k": k, "component_screen": score,
                "small_logical_safe": False,
            }
            if fallback is None:
                fallback = candidate
            if not (_small_logical(hx, hz, n)
                    or _small_logical(hz, hx, n)):
                candidate["small_logical_safe"] = True
                return candidate
            examined += 1
            if examined >= 12:
                break
        # Optional protection preferences must not void a legal slot.
        if fallback is not None:
            return fallback
    raise ValueError("no verified order-27 filling in cell 0.200")


def _retired_q40_12x26_ctx(rng, G, idx):
    """Inactive historical six-column search; new slots never call it."""
    import math
    import numpy as np

    q = G.q
    assert q == 40 and A_SHAPE == (1, 2) and B_SHAPE == (2, 6)
    architecture, policy = divmod(idx % 20, 5)
    batch = idx // 20
    # The fourth architecture also changes B's weighted support graph.
    b_profiles = ((1, 1, 1, 2, 2, 2),
                  (0, 1, 2, 1, 2, 3),
                  (0, 0, 1, 2, 3, 3))
    bp = b_profiles[(int(architecture == 3) + batch) % 3]
    counts = {
        "A": [[3, 3]],
        "B": [list(bp), [3-t for t in bp]],
    }
    assert _no_dead_lines(counts["A"], counts["B"])
    assert _grid_weights(counts["A"], counts["B"]) == 12
    assert 12 <= W_MAX
    assert sum(counts["A"][0]) == 6
    assert all(sum(row) == 9 for row in counts["B"])
    assert all(sum(col) == 3 for col in zip(*counts["B"]))

    els = list(G.elements)
    orders = {}
    for g in els:
        power = G.e
        for order in range(1, q+1):
            power = G.mul(power, g)
            if power == G.e:
                orders[g] = order
                break
        assert g in orders
    palettes = {o: [g for g in els if orders[g] == o]
                for o in (4, 10, 20)}
    assert all(palettes.values())
    assert sum(orders[g] == 2 for g in els) == 1

    Q, R = (4, 20, 20), (20, 20, 20)
    triangle_pools = {Q: [], R: []}
    eligible = palettes[4] + palettes[20]
    for u in eligible:
        for v in eligible:
            if G.mul(u, v) == G.mul(v, u):
                continue
            relative = G.mul(G.inv(u), v)
            pattern = tuple(sorted((orders[u], orders[v],
                                    orders[relative])))
            if pattern in triangle_pools:
                triangle_pools[pattern].append((u, v))
    assert all(triangle_pools.values())
    a_cards = ((Q, Q), (Q, R), (R, R), (Q, Q))
    binomial_cards = (
        (20, 20, 20, 20, 20, 20),
        (4, 20, 20, 20, 20, 20),
        (4, 4, 20, 20, 20, 20),
        (10, 10, 20, 20, 20, 20),
        (4, 10, 20, 20, 20, 20),
    )

    # All automorphisms of Q8, independently crossed with Aut(C5).
    # This canonicalizes relative supports, never substitutes a lift.
    pure = [g for g in els if g[0] == 0 and g[2] != 0]
    automorphisms = []
    for u in pure:
        for v in pure:
            if G.mul(u, v) == G.mul(v, u):
                continue
            images = (G.e, u, v, G.mul(u, v))
            for unit in (1, 2, 3, 4):
                image = []
                for t, sign, axis in els:
                    h = images[axis]
                    image.append(G.idx(
                        ((unit*t) % 5, h[1] ^ sign, h[2])))
                assert len(set(image)) == q
                automorphisms.append(tuple(image))
    automorphisms = sorted(set(automorphisms))
    assert len(automorphisms) == 96

    def normalized(cell, image):
        if not cell:
            return ()
        values = [els[image[G.idx(g)]] for g in cell]
        return min(tuple(sorted(
            G.idx(G.mul(G.inv(anchor), g)) for g in values))
            for anchor in values)

    def geometry_key(grids):
        candidates = []
        for image in automorphisms:
            a = tuple(sorted(normalized(cell, image)
                             for cell in grids["A"][0]))
            bcols = [
                (normalized(grids["B"][0][j], image),
                 normalized(grids["B"][1][j], image))
                for j in range(6)
            ]
            b = min(tuple(sorted(bcols)),
                    tuple(sorted((y, x) for x, y in bcols)))
            candidates.append((a, b))
        return min(candidates)

    # Four architecture classes times five order policies. Higher slot
    # budgets receive new book entries and changed support architectures.
    grng = random.Random(400131 + 1013*architecture
                        + 7919*(batch % 3))
    wanted = idx % 5 + 5*batch
    book, seen = [], set()
    for proposal in range(40000):
        card = len(book) % 5
        grids = {}
        legal = True
        for side in ("A", "B"):
            sites = [(i, j) for i, row in enumerate(counts[side])
                     for j, weight in enumerate(row) if weight == 2]
            schedule = dict(zip(sites, binomial_cards[card]))
            grid = []
            for i, row_counts in enumerate(counts[side]):
                row, differences = [], set()
                for j, weight in enumerate(row_counts):
                    if weight == 0:
                        cell = []
                    elif weight == 1:
                        cell = [G.e]
                    elif weight == 2:
                        cell = [G.e, grng.choice(
                            palettes[schedule[i, j]])]
                    else:
                        pattern = (a_cards[architecture][j]
                                   if side == "A" else R)
                        u, v = grng.choice(triangle_pools[pattern])
                        cell = [G.e, u, v]
                    assert len(set(cell)) == weight
                    for g in cell:
                        for h in cell:
                            if g == h:
                                continue
                            difference = G.mul(G.inv(g), h)
                            if (orders[difference] == 2
                                    or difference in differences):
                                legal = False
                            differences.add(difference)
                    row.append(cell)
                grid.append(row)
            grids[side] = grid
        if not legal:
            continue
        signature = geometry_key(grids)
        if signature in seen:
            continue
        seen.add(signature)
        book.append(grids)
        if len(book) > wanted:
            break
    if len(book) <= wanted:
        raise ValueError("quaternion relative-support book exhausted")
    geometry = book[wanted]

    # Cache only scaffold-built singleton contributions.
    mask = (1 << q)-1
    templates = {"A": [], "B": []}
    for g in els:
        hx, _, _ = lifted_matrices(G, [[[g]]], [[[]]])
        _, hz, _ = lifted_matrices(G, [[[]]], [[[g]]])
        templates["A"].append(_columns([r & mask for r in hx], q))
        templates["B"].append(_columns([r & mask for r in hz], q))

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def component(side, grid):
        nr, nc = len(grid), len(grid[0])
        columns = [0] * (nc*q)
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                for g in cell:
                    block = templates[side][G.idx(g)]
                    for t, syndrome in enumerate(block):
                        columns[j*q+t] ^= syndrome << (i*q)
        assert all(c.bit_count() == 3 for c in columns)
        return columns, nr*q

    def connected(adjacency, height):
        reached, todo = 0, 1
        while todo:
            bit = todo & -todo
            todo ^= bit
            if reached & bit:
                continue
            reached |= bit
            todo |= adjacency.get(bit.bit_length()-1, 0) & ~reached
        return reached.bit_count() == height

    def graph_tail(columns, adjacency):
        # Exact C6 only when C4 is zero; otherwise an adjacency surrogate.
        triangles = 0
        for u, neighbors in adjacency.items():
            higher = neighbors & ~((1 << (u+1))-1)
            while higher:
                bit = higher & -higher
                v = bit.bit_length()-1
                triangles += (
                    neighbors & adjacency.get(v, 0)
                    & ~((1 << (v+1))-1)).bit_count()
                higher ^= bit
        return max(0, triangles-len(columns))

    def sampled_kernel(columns, side):
        """Actual kernel witnesses; a large minimum is only an upper bound."""
        probe = random.Random(400919 + 1013*idx + (side == "B"))
        permutation = list(range(len(columns)))
        minimum = len(columns)+1
        short = set()

        def offer(word):
            nonlocal minimum
            if word:
                weight = word.bit_count()
                minimum = min(minimum, weight)
                if weight <= 8:
                    short.add(word)

        for trial in range(24 if side == "A" else 16):
            probe.shuffle(permutation)
            pivots, basis = {}, []
            for c in permutation:
                syndrome, word = columns[c], 1 << c
                while syndrome:
                    p = syndrome.bit_length()-1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = syndrome, word
                        break
                    syndrome ^= old[0]
                    word ^= old[1]
                if not syndrome and word:
                    basis.append(word)
                    offer(word)
            probe.shuffle(basis)
            for i, word in enumerate(basis):
                weight = word.bit_count()
                for other in basis[:i]:
                    candidate = word ^ other
                    offer(candidate)
                    cw = candidate.bit_count()
                    if candidate and cw < weight:
                        word, weight = candidate, cw
                basis[i] = word
            if minimum <= 4:
                break
        loads = [
            sum(bool((word >> (j*q)) & mask) for word in short)
            for j in range(len(columns)//q)
        ]
        return minimum, len(short), max(loads, default=0)

    def boundary(columns):
        """Worst-block anchored triple boundaries, not an exact trap spectrum."""
        loads = []
        for anchor in range(0, len(columns), q):
            cost = 0
            for i, x in enumerate(columns):
                if i == anchor:
                    continue
                pair = columns[anchor] ^ x
                for j in range(i):
                    if j == anchor:
                        continue
                    weight = (pair ^ columns[j]).bit_count()
                    if weight <= 3:
                        cost += (5-weight)**2
            loads.append(cost)
        return max(loads), sum(loads)

    def optimize(side):
        sites = [(i, j) for i, row in enumerate(counts[side])
                 for j, weight in enumerate(row) if weight]
        pool, seen_fillings = [], set()
        serial = 0

        def retain(grid, cols, four, adjacency):
            nonlocal serial
            height = len(grid)*q
            if not connected(adjacency, height):
                return
            signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                              for row in grid for cell in row)
            if signature in seen_fillings:
                return
            seen_fillings.add(signature)
            rows = _rows_from_columns(cols, height)
            deficit = height-gf2_rank(rows)
            if deficit > 4:
                return
            duplicates, words4, tail = _component_summary(cols, adjacency)
            coarse = (duplicates, words4, deficit, four, tail)
            serial += 1
            pool.append((coarse, serial, clone(grid), list(cols)))
            pool.sort(key=lambda item: item[:2])
            del pool[6:]

        for restart in range(4):
            grid = []
            for row in geometry[side]:
                out = []
                for cell in row:
                    origin = G.pick(rng)
                    out.append([G.mul(origin, g) for g in cell])
                grid.append(out)
            cols, height = component(side, grid)
            four, adjacency = _component_cycles(cols)
            tail = graph_tail(cols, adjacency)
            energy = (80*four+tail/6)/q
            retain(grid, cols, four, adjacency)
            for step in range(180):
                i, j = rng.choice(sites)
                proposal = clone(grid)
                origin = G.pick(rng)
                proposal[i][j] = [
                    G.mul(origin, g) for g in proposal[i][j]]
                pc, _ = component(side, proposal)
                pf, pa = _component_cycles(pc)
                pt = graph_tail(pc, pa)
                pe = (80*pf+pt/6)/q
                delta = pe-energy
                temperature = 4.0*(0.979**step)+0.06
                if delta <= 0 or rng.random() < math.exp(
                        -min(700.0, delta/temperature)):
                    grid, cols, four, adjacency = proposal, pc, pf, pa
                    energy = pe
                if step % 36 == 0:
                    retain(grid, cols, four, adjacency)
            retain(grid, cols, four, adjacency)

        if not pool:
            raise ValueError("no connected quaternion component portfolio")
        preliminary = []
        for coarse, number, grid, cols in pool[:4]:
            distance, short, peak = sampled_kernel(cols, side)
            bp, bt = boundary(cols)
            duplicates, words4, deficit, four, tail = coarse
            key = (duplicates, words4, max(0, 10-distance),
                   deficit, short, peak, bp, bt, four, tail)
            preliminary.append((key, number, grid, cols))
        preliminary.sort(key=lambda item: item[:2])

        refined = []
        for key, number, grid, cols in preliminary[:3]:
            duplicates, words4, floor, deficit = key[:4]
            # Odd column weight excludes odd component kernels.
            # With no weight-two/four words, this is an exact six count.
            words6 = (_six_kernel_count(cols)
                      if not (duplicates or words4) else 1 << 50)
            quality = (duplicates, words4, int(words6 != 0), floor,
                       words6) + key[4:] + (deficit,)
            refined.append((quality, number, grid))
        refined.sort(key=lambda item: item[:2])
        return refined

    def bp_probe(checks, stabilizers, n, seed):
        """Common-noise sum-product surrogate, not the scored ASC experiment.

        Opposite-check pairs model possible correlated data faults, without
        asserting that ASC realizes every such pair. Unsettled shots receive
        a penalty; settled residuals are classified modulo actual stabilizers.
        """
        def neighbors(rows):
            result = []
            for row in rows:
                ns = []
                while row:
                    bit = row & -row
                    ns.append(bit.bit_length()-1)
                    row ^= bit
                result.append(ns)
            return result

        ns = neighbors(checks)
        hooks = neighbors(stabilizers)
        weight = len(ns[0])
        assert all(len(row) == weight for row in ns)
        incidence = np.asarray(ns, dtype=np.int32)
        flat = incidence.ravel()
        nr, edges = len(ns), len(flat)
        degrees = np.bincount(flat, minlength=n)
        assert np.all(degrees > 0)
        variable_edges = np.full(
            (n, int(degrees.max())), edges, dtype=np.int32)
        used = np.zeros(n, dtype=np.int32)
        for edge, variable in enumerate(flat):
            variable_edges[variable, used[variable]] = edge
            used[variable] += 1

        shots = 32
        channel = np.random.default_rng(seed)
        levels = np.asarray((0.012, 0.024, 0.040))
        rates = levels[np.arange(shots) % 3, None] * degrees[None, :] / 3
        error = channel.random((shots, n)) < rates
        for shot in range(shots):
            row = hooks[int(channel.integers(len(hooks)))]
            a = int(channel.integers(len(row)))
            b = int(channel.integers(len(row)-1))
            b += b >= a
            if channel.random() < 0.35:
                error[shot, row[a]] ^= True
                error[shot, row[b]] ^= True

        prior_rates = np.minimum(rates+0.7/n, 0.25)
        prior = np.log((1-prior_rates)/prior_rates)
        target = (error[:, flat].reshape(shots, nr, weight)
                  .sum(axis=2) & 1).astype(bool)
        target_sign = np.where(target, -1.0, 1.0)
        messages = prior[:, flat].copy()
        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = row
                    break
                row ^= old

        def nontrivial(word):
            while word:
                p = word.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    return True
                word ^= old
            return False

        failed = iteration_sum = 0
        for iteration in range(1, 25):
            count = len(error)
            shaped = messages.reshape(count, nr, weight)
            signs = np.where(shaped < 0, -1.0, 1.0)
            logarithms = np.log(np.clip(
                np.tanh(np.abs(shaped)/2), 1e-12, 1-1e-7))
            products = np.minimum(np.exp(
                logarithms.sum(axis=2, keepdims=True)-logarithms),
                1-1e-7)
            magnitude = np.log1p(products)-np.log1p(-products)
            outgoing = (
                magnitude * signs * signs.prod(axis=2, keepdims=True)
                * target_sign[:, :, None]).reshape(count, edges)
            padded = np.zeros((count, edges+1), dtype=np.float64)
            padded[:, :edges] = outgoing
            posterior = prior+padded[:, variable_edges].sum(axis=2)
            hard = posterior < 0
            syndrome = (hard[:, flat].reshape(count, nr, weight)
                        .sum(axis=2) & 1).astype(bool)
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
                0.35*messages+0.65*(posterior[:, flat]-outgoing),
                -24, 24)[keep]
            prior, target = prior[keep], target[keep]
            target_sign, error = target_sign[keep], error[keep]
        unresolved = len(error)
        return failed+unresolved, unresolved, iteration_sum+25*unresolved

    for attempt in range(2):
        apool, bpool = optimize("A"), optimize("B")
        options = []
        for aq, at, A in apool:
            for bq, bt, B in bpool:
                structure = tuple(v for a, b in zip(aq, bq)
                                  for v in (max(a, b), a+b))
                options.append((structure, at, bt, A, B))
        options.sort(key=lambda item: item[:3])
        finalists, fallback = [], None
        for structure, at, bt, A, B in options:
            hx, hz, n = lifted_matrices(G, A, B)
            k = n-gf2_rank(hx)-gf2_rank(hz)
            if n != 560 or not 160 <= k < 182:
                continue
            assert css_ok(hx, hz)
            assert max(r.bit_count() for r in hx) == 9
            assert max(r.bit_count() for r in hz) == 12
            assert 6*sum(r.bit_count() for r in hx+hz) == 24480
            assert all(len(set(cell)) == len(cell)
                       for grid in (A, B) for row in grid for cell in row)
            # Confirm composed columns against independent verified probes.
            ar = lifted_matrices(G, A, [[[G.e]]])[0]
            br = lifted_matrices(G, [[[G.e]]], B)[1]
            ac, ah = component("A", A)
            bc, bh = component("B", B)
            assert [r & ((1 << (2*q))-1) for r in ar] == (
                _rows_from_columns(ac, ah))
            assert [r & ((1 << (6*q))-1) for r in br] == (
                _rows_from_columns(bc, bh))
            candidate = {
                "A": A, "B": B, "k": k, "idx": idx,
                "contract": (architecture, policy, batch),
                "geometry": wanted, "component_screen": structure,
                "small_logical_safe": False,
            }
            if fallback is None:
                fallback = candidate
            if _small_logical(hx, hz, n) or _small_logical(hz, hx, n):
                continue
            candidate["small_logical_safe"] = True
            seed = 400733 + 1013*idx
            x = bp_probe(hz, hx, n, seed)
            z = bp_probe(hx, hz, n, seed+1)
            quality = (
                max(x[0], z[0]), x[0]+z[0],
                max(x[1], z[1]), x[1]+z[1], x[2]+z[2],
                structure, at, bt,
            )
            candidate["bp_surrogate"] = (x, z)
            candidate["joint_screen"] = quality
            finalists.append((quality, candidate))
            if len(finalists) >= 4:
                break
        if finalists:
            finalists.sort(key=lambda item: item[0])
            return finalists[0][1]
        # A finite optional protection search must not lose a legal slot.
        if fallback is not None:
            return fallback
    raise ValueError("no quaternion-product filling retained cell 0.300")


def NEW_CTX(rng, G, idx):
    """Trinomial/quartic contracts on the shortened, verified LP bed."""
    import math
    import numpy as np

    q = G.q
    assert q == 40 and A_SHAPE == (1, 2) and B_SHAPE == (2, 5)
    architecture, policy = divmod(idx % 20, 5)
    batch = idx // 20
    first = (
        (1, 1, 1, 2, 2),
        (0, 1, 2, 2, 2),
        (0, 1, 1, 2, 3),
        (0, 0, 2, 2, 3),
    )[architecture]
    counts = {
        "A": [[3, 4]],
        "B": [list(first), [3-t for t in first]],
    }
    assert _no_dead_lines(counts["A"], counts["B"])
    assert sum(counts["A"][0]) == 7
    assert sorted(map(sum, counts["B"])) == [7, 8]
    assert all(sum(col) == 3 for col in zip(*counts["B"]))
    wx = 7 + max(map(sum, zip(*counts["B"])))
    wz = max(map(sum, counts["B"])) + 4
    assert wx == 10 and wz == _grid_weights(
        counts["A"], counts["B"]) == 12
    assert wz <= W_MAX

    els = list(G.elements)
    orders = {}
    for g in els:
        power = G.e
        for order in range(1, q+1):
            power = G.mul(power, g)
            if power == G.e:
                orders[g] = order
                break
    assert len(orders) == q
    palettes = {o: [g for g in els if orders[g] == o]
                for o in (4, 10, 20)}
    triangle_books = {(4, 20, 20): [], (20, 20, 20): []}
    for u in palettes[4] + palettes[20]:
        for v in palettes[4] + palettes[20]:
            if G.mul(u, v) == G.mul(v, u):
                continue
            signature = tuple(sorted((
                orders[u], orders[v], orders[G.mul(G.inv(u), v)])))
            if signature in triangle_books:
                triangle_books[signature].append((u, v))
    assert all(triangle_books.values())

    triangle_kind = (
        (4, 20, 20), (20, 20, 20), (4, 20, 20),
        (20, 20, 20), (20, 20, 20),
    )[policy]
    quartic_phases = (
        (1, 2, 3), (0, 1, 2), (1, 1, 2),
        (0, 0, 1), (1, 1, 1),
    )[policy]
    binomial_orders = (
        (20, 20, 20, 20, 20),
        (4, 20, 20, 20, 20),
        (10, 20, 10, 20, 20),
        (4, 20, 4, 20, 20),
        (4, 10, 20, 4, 20),
    )[policy]

    # Cache only singleton actions from the authoritative lift.
    mask = (1 << q)-1
    actions = {"A": {}, "B": {}}
    for g in els:
        hx, _, _ = lifted_matrices(G, [[[g]]], [[[]]])
        _, hz, _ = lifted_matrices(G, [[[]]], [[[g]]])
        actions["A"][g] = tuple(row & mask for row in hx)
        actions["B"][g] = tuple(row & mask for row in hz)

    def draw_cell(side, i, j):
        weight = counts[side][i][j]
        if not weight:
            return []
        origin = G.pick(rng)
        if weight == 1:
            return [origin]
        if weight == 2:
            wanted = binomial_orders[(j + batch + i) % 5]
            relative = rng.choice(palettes[wanted])
            return [origin, G.mul(origin, relative)]
        if weight == 3:
            kind = triangle_kind if side == "A" else (
                (4, 20, 20) if (i+j+batch) % 2
                else (20, 20, 20))
            u, v = rng.choice(triangle_books[kind])
            relative = [G.e, u, v]
        else:
            assert side == "A" and weight == 4
            # Four different Q8 axes make every pair ratio noncentral;
            # hence none is the unique involution, regardless of C5 phase.
            axes = rng.sample((1, 2, 3), 3)
            phases = list(quartic_phases)
            rng.shuffle(phases)
            if batch % 3 == 1:
                phases[0] = (phases[0]+1) % 5
            elif batch % 3 == 2:
                phases[1] = (phases[1]+2) % 5
            relative = [G.e] + [
                (t, rng.randrange(2), a)
                for t, a in zip(phases, axes)
            ]
        cell = [G.mul(origin, g) for g in relative]
        assert len(set(cell)) == weight
        assert all(orders[G.mul(G.inv(g), h)] != 2
                   for a, g in enumerate(cell) for h in cell[:a])
        return cell

    def draw_grid(side):
        return [[draw_cell(side, i, j)
                 for j in range(len(row))]
                for i, row in enumerate(counts[side])]

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def component(side, grid):
        rows = []
        for grid_row in grid:
            out = [0]*q
            for j, cell in enumerate(grid_row):
                for g in cell:
                    for r, bits in enumerate(actions[side][g]):
                        out[r] ^= bits << (j*q)
            rows.extend(out)
        width = len(grid[0])*q
        return rows, _columns(rows, width)

    def graph_cost(rows, cols):
        four, adjacency = _component_cycles(cols)
        unseen = (1 << len(rows))-1
        parts = 0
        while unseen:
            parts += 1
            todo = unseen & -unseen
            unseen ^= todo
            while todo:
                bit = todo & -todo
                todo ^= bit
                fresh = adjacency.get(bit.bit_length()-1, 0) & unseen
                unseen ^= fresh
                todo |= fresh
        triangles = 0
        for u, neighbors in adjacency.items():
            higher = neighbors & ~((1 << (u+1))-1)
            while higher:
                bit = higher & -higher
                v = bit.bit_length()-1
                triangles += (
                    neighbors & adjacency.get(v, 0)
                    & ~((1 << (v+1))-1)).bit_count()
                higher ^= bit
        stars = sum(d*(d-1)*(d-2)//6
                    for d in map(int.bit_count, cols))
        # Exact C6 only when C4=0; otherwise a proposal surrogate.
        tail = max(0, triangles-stars)
        deficit = len(rows)-gf2_rank(rows)
        key = (parts-1, deficit, four, tail)
        energy = (70*(parts-1) + 50*deficit
                  + (100*four + tail/4)/q)
        return key, energy

    def short_kernel(cols, through_six):
        """Exact kernel existence through four/six, including mixed degrees."""
        seen = {0}
        for value in cols:
            if value in seen:
                return True
            seen.add(value)
        for i, value in enumerate(cols):
            for other in cols[:i]:
                syndrome = value ^ other
                if syndrome in seen:
                    return True
                seen.add(syndrome)
        if through_six:
            for i, value in enumerate(cols):
                for j in range(i):
                    pair = value ^ cols[j]
                    for h in range(j):
                        syndrome = pair ^ cols[h]
                        if syndrome in seen:
                            return True
                        seen.add(syndrome)
        return False

    def witness_screen(cols, side):
        """Common-stream actual kernel witnesses, not distance certificates."""
        probe = random.Random(402517 + 1009*idx + (side == "B"))
        order = list(range(len(cols)))
        minimum = len(cols)+1
        short = set()

        def offer(word):
            nonlocal minimum
            if word:
                weight = word.bit_count()
                minimum = min(minimum, weight)
                if weight <= 10 and len(short) < 4096:
                    short.add(word)

        for trial in range(32):
            probe.shuffle(order)
            pivots, basis = {}, []
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
                    offer(word)
            probe.shuffle(basis)
            for i, word in enumerate(basis):
                weight = word.bit_count()
                for other in basis[:i]:
                    candidate = word ^ other
                    offer(candidate)
                    if candidate and candidate.bit_count() < weight:
                        word, weight = candidate, candidate.bit_count()
                basis[i] = word
            if minimum <= 4:
                break
        loads = [sum(bool((word >> (j*q)) & mask) for word in short)
                 for j in range(len(cols)//q)]
        return max(0, 12-minimum), len(short), max(loads, default=0)

    def boundary(cols):
        loads = []
        for anchor in range(0, len(cols), q):
            cost = 0
            for i in range(len(cols)):
                if i == anchor:
                    continue
                pair = cols[anchor] ^ cols[i]
                for j in range(i):
                    if j == anchor:
                        continue
                    weight = (pair ^ cols[j]).bit_count()
                    if weight <= 4:
                        cost += (6-weight)**2
            loads.append(cost)
        return max(loads), sum(loads)

    def portfolio(side):
        positions = [(i, j) for i, row in enumerate(counts[side])
                     for j, weight in enumerate(row) if weight]
        pool, seen = [], set()
        serial = 0

        def retain(grid, rows, cols, coarse):
            nonlocal serial
            signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                              for row in grid for cell in row)
            if signature in seen:
                return
            seen.add(signature)
            serial += 1
            pool.append((coarse, serial, clone(grid), rows, cols))
            pool.sort(key=lambda item: item[:2])
            del pool[6:]

        for restart in range(4):
            grid = draw_grid(side)
            rows, cols = component(side, grid)
            coarse, energy = graph_cost(rows, cols)
            retain(grid, rows, cols, coarse)
            for step in range(220):
                i, j = rng.choice(positions)
                previous = grid[i][j]
                # Redraw internal supports. In particular, do not spend
                # single-row A search solely on whole-cell origin gauges.
                grid[i][j] = draw_cell(side, i, j)
                proposed_rows, proposed_cols = component(side, grid)
                proposed, pe = graph_cost(proposed_rows, proposed_cols)
                delta = pe-energy
                temperature = 12*(0.981**step)+0.10
                if delta <= 0 or rng.random() < math.exp(
                        -min(700.0, delta/temperature)):
                    rows, cols = proposed_rows, proposed_cols
                    coarse, energy = proposed, pe
                    if step % 44 == 43:
                        retain(grid, rows, cols, coarse)
                else:
                    grid[i][j] = previous
            retain(grid, rows, cols, coarse)

        finalists = []
        for coarse, serial, grid, rows, cols in pool[:4]:
            bad4 = int(short_kernel(cols, False))
            bad6 = int(short_kernel(cols, True)) if not bad4 else 1
            gap, low, peak = witness_screen(cols, side)
            bp, bt = boundary(cols)
            key = (coarse[0], bad4, bad6, coarse[1],
                   gap, low, peak, bp, bt, coarse[2], coarse[3])
            finalists.append((key, serial, grid, rows))
        finalists.sort(key=lambda item: item[:2])
        return finalists[:3]

    def connected(rows, width):
        parent = list(range(width))

        def root(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for row in rows:
            if not row:
                return False
            anchor = (row & -row).bit_length()-1
            while row:
                bit = row & -row
                a, b = root(anchor), root(bit.bit_length()-1)
                if a != b:
                    parent[b] = a
                row ^= bit
        return len({root(a) for a in range(width)}) == 1

    def bp_probe(checks, stabilizers, n, seed, shots):
        """Irregular-check capacity BP surrogate, NOT ASC memory simulation."""
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

        ns, hooks = neighborhoods(checks), neighborhoods(stabilizers)
        nr, weight = len(ns), max(map(len, ns))
        incidence = np.full((nr, weight), n, dtype=np.int32)
        for r, neighbors in enumerate(ns):
            incidence[r, :len(neighbors)] = neighbors
        flat = incidence.ravel()
        valid = flat != n
        edges = len(flat)
        degrees = np.bincount(flat[valid], minlength=n)
        other = np.asarray([c.bit_count()
                            for c in _columns(stabilizers, n)])
        assert np.all(degrees > 0)
        variable_edges = np.full(
            (n+1, int(degrees.max())), edges, dtype=np.int32)
        used = np.zeros(n, dtype=np.int32)
        for edge in np.flatnonzero(valid):
            c = int(flat[edge])
            variable_edges[c, used[c]] = edge
            used[c] += 1

        channel = np.random.default_rng(seed)
        levels = np.asarray((0.012, 0.021, 0.032))
        rates = np.minimum(
            levels[np.arange(shots) % 3, None]
            * (degrees+other)[None, :]/6, 0.20)
        error = np.zeros((shots, n+1), dtype=bool)
        error[:, :n] = channel.random((shots, n)) < rates
        for shot in range(shots):
            neighbors = hooks[int(channel.integers(len(hooks)))]
            a = int(channel.integers(len(neighbors)))
            b = int(channel.integers(len(neighbors)-1))
            b += b >= a
            if channel.random() < 0.35:
                error[shot, neighbors[a]] ^= True
                error[shot, neighbors[b]] ^= True

        prior_rates = np.minimum(rates+0.7/n, 0.25)
        prior = np.full((shots, n+1), 30.0)
        prior[:, :n] = np.log((1-prior_rates)/prior_rates)
        target = (error[:, flat].reshape(shots, nr, weight)
                  .sum(axis=2) & 1).astype(bool)
        target_sign = np.where(target, -1.0, 1.0)
        messages = prior[:, flat].copy()
        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = row
                    break
                row ^= old

        def nontrivial(word):
            while word:
                p = word.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    return True
                word ^= old
            return False

        failed = iteration_sum = 0
        for iteration in range(1, 29):
            count = len(error)
            shaped = messages.reshape(count, nr, weight)
            signs = np.where(shaped < 0, -1.0, 1.0)
            logarithms = np.log(np.clip(
                np.tanh(np.abs(shaped)/2), 1e-12, 1-1e-7))
            products = np.minimum(np.exp(
                logarithms.sum(axis=2, keepdims=True)-logarithms),
                1-1e-7)
            outgoing = (
                (np.log1p(products)-np.log1p(-products))
                * signs * signs.prod(axis=2, keepdims=True)
                * target_sign[:, :, None]).reshape(count, edges)
            outgoing[:, ~valid] = 0.0
            padded = np.zeros((count, edges+1))
            padded[:, :edges] = outgoing
            posterior = prior+padded[:, variable_edges].sum(axis=2)
            posterior[:, n] = 30.0
            hard = posterior < 0
            syndrome = (hard[:, flat].reshape(count, nr, weight)
                        .sum(axis=2) & 1).astype(bool)
            settled = np.all(syndrome == target, axis=1)
            for local in np.flatnonzero(settled):
                word = 0
                for c in np.flatnonzero(
                        error[local, :n] ^ hard[local, :n]):
                    word |= 1 << int(c)
                failed += int(nontrivial(word))
                iteration_sum += iteration
            keep = ~settled
            if not np.any(keep):
                return failed, 0, iteration_sum
            messages = np.clip(
                0.30*messages+0.70*(posterior[:, flat]-outgoing),
                -28, 28)[keep]
            messages[:, ~valid] = 30.0
            prior, error = prior[keep], error[keep]
            target, target_sign = target[keep], target_sign[keep]
        unresolved = len(error)
        return failed+unresolved, unresolved, iteration_sum+29*unresolved

    def joint(x, z):
        return (max(x[0], z[0]), x[0]+z[0],
                max(x[1], z[1]), x[1]+z[1], x[2]+z[2])

    for attempt in range(2):
        aa, bb = portfolio("A"), portfolio("B")
        options = []
        for ak, at, A, ar in aa:
            for bk, bt, B, br in bb:
                structure = tuple(v for a, b in zip(ak, bk)
                                  for v in (max(a, b), a+b))
                options.append((structure, at, bt, A, B, ar, br))
        options.sort(key=lambda item: item[:3])
        legal, safe = [], []
        for structure, at, bt, A, B, ar, br in options:
            hx, hz, n = lifted_matrices(G, A, B)
            k = n-gf2_rank(hx)-gf2_rank(hz)
            if n != 480 or not 120 <= k <= 131:
                continue
            if not connected(hx+hz, n):
                continue
            assert css_ok(hx, hz)
            assert all(r.bit_count() == 10 for r in hx)
            assert max(r.bit_count() for r in hz) == 12
            assert 6*sum(r.bit_count() for r in hx+hz) == 22560
            actual_a = [
                sum(((row >> (j*5*q)) & mask) << (j*q)
                    for j in range(2))
                for row in hx[:q]
            ]
            actual_b = [row & ((1 << (5*q))-1) for row in hz[:2*q]]
            assert actual_a == ar and actual_b == br
            candidate = {
                "A": A, "B": B, "k": k, "idx": idx,
                "contract": (architecture, policy, batch),
                "component_screen": structure,
                "small_logical_safe": False,
            }
            legal.append(candidate)
            if _small_logical(hx, hz, n) or _small_logical(hz, hx, n):
                continue
            candidate["small_logical_safe"] = True
            seed = 402791 + 1009*idx
            x = bp_probe(hz, hx, n, seed, 24)
            z = bp_probe(hx, hz, n, seed+1, 24)
            safe.append((joint(x, z), structure, at, bt,
                         candidate, hx, hz, x, z))
            if len(safe) >= 4:
                break
        if safe:
            safe.sort(key=lambda item: item[:4])
            finalists = []
            for _, structure, at, bt, candidate, hx, hz, x, z in safe[:2]:
                # Independent holdout noise, common across finalists.
                seed = 1402791 + 1009*idx
                xx = bp_probe(hz, hx, 480, seed, 64)
                zz = bp_probe(hx, hz, 480, seed+1, 64)
                total_x = tuple(a+b for a, b in zip(x, xx))
                total_z = tuple(a+b for a, b in zip(z, zz))
                quality = joint(total_x, total_z)+(structure, at, bt)
                candidate["bp_surrogate"] = (total_x, total_z)
                candidate["joint_screen"] = quality
                finalists.append((quality, candidate))
            finalists.sort(key=lambda item: item[0])
            return finalists[0][1]
        # Protection preferences must not void an otherwise legal slot.
        if legal:
            return legal[0]
    raise ValueError("no verified triquartic filling retained cell 0.250")


def CELL(rng, G, side, i, j, ctx):
    """Return the complete selected cell without further randomization."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal rolled supports and uniform labels alone did not give reliable memory on the parent D14 (6,2)x(6,2) bed: fitness 3.4511, ubiquitous weight-four sector-2 witnesses, and frequent failures in both bases.
# REFUTED: Distinct rotational A-binomial steps and raw syndrome-ambiguity screening sufficiently protect the D14 (1,4)x(2,7) bed: fitness was 3.2219, measured slots retained abundant weight-three sector-1 logicals, and X failures were 108-139 per 250 shots.
# REFUTED: The full-row-rank component walk produces a usable filling for every support card: current slots1,4,5 and many others were voided because no full-row-rank filling was found.
# REFUTED: Equal n=560,k=112,d2s=10 with component-kernel screening ensures basis-balanced reliability: slot56 had 43 X failures versus one Z failure/200000 and s=6.365, while slot0 scored 7.118.
# REFUTED: Equal n=540,k=108,w=9,d2s=12 determines circuit score: current slot0 scored 6.848 with ten Z failures, whereas slot21 scored 7.254 with two failures per basis, all at 200000 shots.
# REFUTED: Involution-free degree-three Q8 x C5 components plus joint capacity-BP selection improve the six-column bed beyond 6.1594: measured fitness was 6.0617, with 67-101 Z failures per measured code and collective flips spanning all 160 logicals.
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
