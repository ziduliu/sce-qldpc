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
CONCEPT = ("Quaternion Q8, order eight, shapes (3,8)x(3,8): equal-length full-augmentation-rank components, noncentral order-four ratios and selective degree-four reinforcement should raise worse-basis circuit fitness above the parent's 5.0850.")
GROUP = ("Jump from cyclic Z13 to nonabelian Q8, an untried group/shape bed. The cyclic Z8 (3,8)x(3,8) bed scored about 3.9, warning that lift order alone is insufficient. In F2[Q8] the augmentation ideal is nilpotent, so an invertible augmentation minor lifts to an invertible group-algebra minor. Require full component ranks and k=200 using the verified scaffold; no measured quaternion advantage is assumed.")
SHAPES = ("Target cell 0.350 with equal (3,8) components, each having 64 variables and 24 checks. n=73*8=584, kb=25*8=200 and guaranteed rate 25/73=0.342466. Full-row-rank components make the complete check matrices full-row-rank, verified explicitly. Parent failures reached all 195 logicals with diffuse Acol/Bcol enrichment around 1.1-1.2x and mild Acol/Brow check enrichment; replace unequal component geometry globally rather than claim an isolated block repair.")
FILLING = ("Twenty deliberate support-pair contracts combine four canonical balanced degree-three architectures and three one-sided degree-four reinforcements. Balanced rows load eight; reinforced rows load eight/nine, while the opposite columns remain degree three. HX and HZ therefore weigh eleven/twelve, never above twelve. Balanced extraction uses 25344 six-round CNOTs, with 528 added per reinforced term. Binomial ratios have order four and distinct inverse classes within a row. Verified singleton actions, component kernel/cycle screens and joint stabilizer-aware BP stress tests select labels. A deterministic reservation prefix discards equal exact full-Tanner invariants and reserves all finalists, including extra harness batches. The BP benchmark omits noisy measurement histories, ASC ordering and relay/LSD, so circuit measurement must decide whether the parent's 100 X failures in 9000-19750 shots and collective flips decrease.")
FAMILY_NAME = "r035_Q8_38x38_augfull_noncentral_jointbp_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 8

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Quaternion Q8 requires order eight")
    # x^4=1, y^2=x^2, y*x*y^-1=x^-1; elements x^a*y^b.
    els = [(a, b) for b in range(2) for a in range(4)]

    def red(g):
        return (int(g[0]) % 4, int(g[1]) % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (1 if b == 0 else -1)*c + 2*b*d) % 4,
                (b+d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % 4, 0) if b == 0 else ((a+2) % 4, 1)

    def idx(g):
        a, b = red(g)
        return 4*b+a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (3, 8)               # (ra, ca)
B_SHAPE = (3, 8)               # (rb, cb)


def _retired_Z13_CTX(rng, G, idx):
    """Inactive parent experiment, retained for provenance; never called."""
    import math
    from itertools import combinations, permutations, combinations_with_replacement

    q = G.q
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    design, batch = idx % 20, idx // 20
    budget, card = divmod(design, 10)
    architecture, b_model = 0, design

    a_profiles = (
        ((2, 2, 1, 1, 1), (1, 1, 2, 2, 2)),
    )

    def canonical_support(columns):
        """Exclude count-card row/column copies, not certify full Tanner GI."""
        return min(
            tuple(sorted(tuple(column[i] for i in order)
                         for column in columns))
            for order in permutations(range(rb))
        )

    two_types = sorted(set(permutations((1, 1, 0))))
    three_types = sorted(
        {(1, 1, 1)} | set(permutations((2, 1, 0)))
    )
    four_types = sorted(
        set(permutations((2, 1, 1)))
        | set(permutations((2, 2, 0)))
        | set(permutations((3, 1, 0)))
    )
    books = [set(), set()]

    def offer(which, columns):
        grid = [[c[i] for c in columns] for i in range(rb)]
        loads = sorted(map(sum, grid))
        if loads != ([7, 8, 8] if which == 0 else [8, 8, 8]):
            return
        if any(sum(t * (t - 1) // 2 for t in row) > 6
               for row in grid):
            return
        augmentation = [
            sum((t & 1) << j for j, t in enumerate(row))
            for row in grid
        ]
        if gf2_rank(augmentation) == rb:
            books[which].add(canonical_support(columns))

    # One mixed-row degree-two column is common to both budgets.
    # B24 exchanges one degree-three column for a degree-four column.
    # These are count architectures, not transformed completed codes.
    for which, number in ((0, 7), (1, 6)):
        for tail in combinations_with_replacement(three_types, number):
            if not 1 <= tail.count((1, 1, 1)) <= 5:
                continue
            for weak in two_types:
                if which == 0:
                    offer(which, (weak,) + tail)
                else:
                    for strong in four_types:
                        offer(which, (weak, strong) + tail)

    books = [sorted(book) for book in books]
    assert all(len(book) >= 10 for book in books)
    book = books[budget]
    b_columns = book[card * len(book) // 10]
    sb = [[column[i] for column in b_columns] for i in range(rb)]
    counts = {"A": [list(row) for row in a_profiles[architecture]],
              "B": sb}
    expected_degrees = ([2] + [3] * 7 if budget == 0
                        else [2] + [3] * 6 + [4])
    assert sorted(map(sum, b_columns)) == expected_degrees
    assert sorted(map(sum, sb)) == (
        [7, 8, 8] if budget == 0 else [8, 8, 8])
    assert all(sorted(c) == [0, 1, 1]
               for c in b_columns if sum(c) == 2)
    assert sum(map(sum, counts["A"])) == 15
    assert sum(map(sum, sb)) == 23 + budget
    assert gf2_rank([
        sum((t & 1) << j for j, t in enumerate(row)) for row in sb
    ]) == rb
    assert _grid_weights(counts["A"], sb) <= W_MAX
    assert _no_dead_lines(counts["A"], sb)
    mask = (1 << q) - 1

    def direction(value):
        value %= q
        return min(value, q - value)

    def internal(values):
        return {direction(a - b)
                for a, b in combinations(values, 2)}

    def normalized(values, multiplier):
        values = [multiplier * v % q for v in values]
        return min(tuple(sorted((v - a) % q for v in values))
                   for a in values) if values else ()

    def geometry_key(templates):
        """Canonicalize prescribed ratios; this is not full Tanner GI."""
        images = []
        for multiplier in range(1, q):
            sides = []
            for side in ("A", "B"):
                grid = counts[side]
                row_images = []
                for order in permutations(range(len(grid))):
                    columns = []
                    for j in range(len(grid[0])):
                        column = []
                        for i in order:
                            values = templates[side][i][j]
                            label = ((-1,) if values is None else
                                     normalized(values, multiplier))
                            column.append((grid[i][j], label))
                        columns.append(tuple(column))
                    row_images.append(tuple(sorted(columns)))
                sides.append(min(row_images))
            images.append(tuple(sides))
        return min(images)

    triple_shapes = [
        (0, a, b) for a, b in combinations(range(1, q), 2)
        if len(internal((0, a, b))) == 3
    ]
    book_rng = random.Random(131024593 + 1009 * design)
    seen_geometries = set()
    geometry = None
    number = 0
    for proposal in range(4000):
        ta = []
        possible = True
        for row in counts["A"]:
            out = [None] * ca
            occupied = set()
            # Choose larger cells first so a legal row palette remains.
            for j in sorted(range(ca), key=lambda j: -row[j]):
                t = row[j]
                if t == 3:
                    choices = [v for v in triple_shapes
                               if not (internal(v) & occupied)]
                elif t == 2:
                    choices = [(0, d) for d in range(1, 7)
                               if d not in occupied]
                else:
                    choices = [(0,)] if t else [()]
                if not choices:
                    possible = False
                    break
                out[j] = book_rng.choice(choices)
                occupied.update(internal(out[j]))
            ta.append(out)
            if not possible:
                break
        if not possible:
            continue
        if any(counts["A"][0][j] == counts["A"][1][j] == 2
               and internal(ta[0][j]) == internal(ta[1][j])
               for j in range(ca)):
            continue

        # B's support architecture is prescribed, but its internal
        # ratios and inter-cell voltages are optimized freely.
        # Extra harness batches select new canonical A ratio geometry.
        tb = [[None if t else () for t in row] for row in sb]
        templates = {"A": ta, "B": tb}
        key = geometry_key(templates)
        if key in seen_geometries:
            continue
        seen_geometries.add(key)
        if number == batch:
            geometry = templates
            break
        number += 1
    if geometry is None:
        raise ValueError("no unused A ratio card for B-reinforcement contract")

    def draw_cell(side, i, j):
        values = geometry[side][i][j]
        if values is None:
            return rng.sample(G.elements, counts[side][i][j])
        g = G.pick(rng)
        return [G.mul(g, (v, 0)) for v in values]

    def draw(side):
        return [[draw_cell(side, i, j) for j in range(len(row))]
                for i, row in enumerate(counts[side])]

    def verified_component(side, grid):
        # Zero opposite-side probes isolate the scaffold's exact actions.
        rows = (lifted_matrices(G, grid, [[[]]])[0]
                if side == "A" else
                lifted_matrices(G, [[[]]], grid)[1])
        width = len(grid[0]) * q
        keep = (1 << width) - 1
        return [row & keep for row in rows]

    def transpose(rows, width):
        cols = [0] * width
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                cols[bit.bit_length() - 1] ^= 1 << r
                row ^= bit
        return cols

    # Linear composition of helper-returned singleton blocks, not a
    # separately implemented lift. Final components are checked below.
    actions = {"A": {}, "B": {}}
    for side in actions:
        for g in G.elements:
            rows = verified_component(side, [[[g]]])
            actions[side][G._key(g)] = tuple(transpose(rows, q))

    def component_columns(side, grid):
        cols = [0] * (len(grid[0]) * q)
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                for g in cell:
                    block = actions[side][G._key(g)]
                    for t, value in enumerate(block):
                        cols[j * q + t] ^= value << (i * q)
        expected = [sum(row[j] for row in counts[side])
                    for j in range(len(grid[0]))]
        assert all(c.bit_count() == expected[j // q]
                   for j, c in enumerate(cols))
        return cols

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
        # Exact when C4=0; otherwise a nonnegative proposal heuristic.
        # Reinforced contracts may have unavoidable component C4.
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

    def decoder_burden(columns, side):
        """Fixed-bank classical component BP benchmark, not circuit pL.

        A valid wrong solution is a component kernel miscorrection.
        Unresolved patterns indicate decoder burden. Neither observation
        distinguishes quantum stabilizers or models ASC fault propagation.
        """
        import numpy as np

        nc = len(columns)
        nr = len(counts[side]) * q
        edge_variables = []
        checks = [[] for _ in range(nr)]
        variables = [[] for _ in range(nc)]
        for v, syndrome in enumerate(columns):
            while syndrome:
                bit = syndrome & -syndrome
                c = bit.bit_length() - 1
                edge = len(edge_variables)
                edge_variables.append(v)
                checks[c].append(edge)
                variables[v].append(edge)
                syndrome ^= bit

        ne = len(edge_variables)
        ev = np.asarray(edge_variables, dtype=np.int64)
        ce = np.full((nr, max(map(len, checks))), ne, dtype=np.int64)
        ve = np.full((nc, max(map(len, variables))), ne, dtype=np.int64)
        for c, edges in enumerate(checks):
            ce[c, :len(edges)] = edges
        for v, edges in enumerate(variables):
            ve[v, :len(edges)] = edges
        real = ce != ne
        check_variables = np.asarray(
            edge_variables + [nc], dtype=np.int64)[ce]

        # Every finalist of this side/slot receives the same errors.
        # Degree weighting represents unequal exposure, not gate noise.
        srng = np.random.default_rng(
            2701531 + 1009 * idx + 7919 * int(side == "B"))
        degrees = np.asarray([c.bit_count() for c in columns], dtype=float)
        probabilities = degrees / degrees.sum()
        weights = (2, 3, 4) if side == "A" else (3, 4, 5, 6)
        shots = 64 * len(weights)
        errors = np.zeros((shots, nc + 1), dtype=bool)
        for shot in range(shots):
            selected = srng.choice(
                nc, weights[shot // 64], replace=False, p=probabilities)
            errors[shot, selected] = True
        syndrome = np.logical_xor.reduce(
            errors[:, check_variables], axis=2)
        signs = 1.0 - 2.0 * syndrome.astype(float)

        p = np.clip(
            np.mean(weights) / nc * degrees / degrees.mean(),
            0.015, 0.14)
        prior = np.log((1.0 - p) / p)
        v2c = np.empty((shots, ne + 1), dtype=float)
        v2c[:, :ne] = prior[ev]
        v2c[:, ne] = 30.0
        c2v = np.zeros_like(v2c)
        active = np.ones(shots, dtype=bool)
        wrong = 0
        early = None
        ones = np.ones((shots, nr, 1), dtype=float)

        for iteration in range(32):
            t = np.tanh(np.clip(v2c[:, ce], -18.0, 18.0) / 2.0)
            t[:, ~real] = 1.0
            left = np.concatenate(
                (ones, np.cumprod(t, axis=2)[:, :, :-1]), axis=2)
            suffix = np.cumprod(t[:, :, ::-1], axis=2)[:, :, ::-1]
            right = np.concatenate((suffix[:, :, 1:], ones), axis=2)
            outgoing = 2.0 * np.arctanh(np.clip(
                signs[:, :, None] * left * right,
                -0.999999, 0.999999))
            c2v[:, ce[real]] = outgoing[:, real]
            posterior = prior[None, :] + c2v[:, ve].sum(axis=2)
            estimate = np.concatenate(
                (posterior < 0, np.zeros((shots, 1), dtype=bool)),
                axis=1)
            matched = np.logical_xor.reduce(
                estimate[:, check_variables], axis=2)
            settled = active & np.all(matched == syndrome, axis=1)
            incorrect = np.any(
                estimate[:, :nc] != errors[:, :nc], axis=1)
            wrong += int(np.count_nonzero(settled & incorrect))
            active[settled] = False
            if iteration == 11:
                early = int(active.sum())
            if not active.any():
                if early is None:
                    early = 0
                break
            damping = 0.25 if iteration < 12 else 0.50
            proposal = posterior[:, ev] - c2v[:, :ne]
            v2c[:, :ne] = (
                damping * v2c[:, :ne] + (1.0 - damping) * proposal)

        remaining = int(active.sum())
        if early is None:
            early = remaining
        return (3.0 * wrong + remaining + early / 16.0) / shots

    def optimize(side):
        sites = [(i, j) for i, row in enumerate(counts[side])
                 for j, t in enumerate(row) if t]
        number_rows = len(counts[side])
        augmentation = [
            sum((t & 1) << j for j, t in enumerate(row))
            for row in counts[side]
        ]
        allowed_deficit = number_rows - gf2_rank(augmentation)

        def rank_penalty(columns):
            deficit = number_rows * q - gf2_rank(
                rows_from_columns(columns))
            return max(0, deficit - allowed_deficit)

        def short_quality(columns):
            singles = {}
            for s in columns:
                singles[s] = singles.get(s, 0) + 1
            n2 = sum(v * (v - 1) // 2 for v in singles.values())
            pairs = {}
            n3 = n4 = 0
            for i, x in enumerate(columns):
                for j in range(i):
                    s = x ^ columns[j]
                    n3 += singles.get(s, 0)
                    n4 += pairs.get(s, 0)
                    pairs[s] = pairs.get(s, 0) + 1
            return n2, n3, n4

        def energy(columns, cost, adj):
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
            stars = sum(d * (d - 1) * (d - 2) // 6
                        for d in (c.bit_count() for c in columns))
            six_proxy = max(0, triangles - stars)
            return (60 * cost + six_proxy / 4) / q

        pool, tested = [], set()

        def retain(grid, columns, cost, adj):
            sig = signature(grid)
            if sig in tested:
                return
            tested.add(sig)
            quick = ((rank_penalty(columns),)
                     + short_quality(columns) + (cost,))
            # Neutral finalists should not be selected by smallest voltage
            # labels. This tie bank changes no completed filling by a gauge.
            pool.append((quick, (rng.random(), sig), grid, columns, adj))
            pool.sort(key=lambda item: item[:2])
            del pool[16:]

        # Retain legal nonzero-C4 minima too. The parent's C4=0-only
        # shortlist otherwise returned an entirely unscreened fallback.
        for restart in range(4):
            current = draw(side)
            cols = component_columns(side, current)
            cost, adj = cycle_data(cols)
            current_energy = energy(cols, cost, adj)
            local_min = current_energy
            retain(current, cols, cost, adj)
            steps = 520 + 40 * (batch % 3)

            for step in range(steps):
                i, j = rng.choice(sites)
                trial = [list(row) for row in current]
                # A and pure B cycle ratios are prescribed. Other B
                # doubletons may change both displacement and voltage.
                trial[i][j] = draw_cell(side, i, j)
                tc = component_columns(side, trial)
                trial_cost, ta = cycle_data(tc)
                proposed = energy(tc, trial_cost, ta)
                delta = proposed - current_energy
                temperature = 14 * (0.988 ** step) + 0.12
                if delta <= 0 or rng.random() < math.exp(
                        -min(700.0, delta / temperature)):
                    current, cols = trial, tc
                    cost, adj = trial_cost, ta
                    current_energy = proposed

                improved = current_energy < local_min
                local_min = min(local_min, current_energy)
                if improved or step % 32 == 0:
                    retain(current, cols, cost, adj)
            retain(current, cols, cost, adj)

        finalists = []
        for quick, sig, grid, columns, adj in pool:
            metric = ((quick[0],) + kernel_metric(columns, adj))
            preliminary = metric[:-1] + (quick[-1], metric[-1])
            finalists.append(
                (preliminary, sig, grid, columns, metric, quick[-1]))
        finalists.sort(key=lambda item: item[:2])

        def boundary_exposure(columns):
            """Small component boundaries, not ASC-certified trapping sets."""
            loads = []
            for anchor in range(0, len(columns), q):
                total = 0
                for a in range(len(columns)):
                    if a == anchor:
                        continue
                    pair = columns[anchor] ^ columns[a]
                    w = pair.bit_count()
                    if w <= 2:
                        total += 4 * (3 - w) ** 2
                    for b in range(a):
                        if b == anchor:
                            continue
                        w = (pair ^ columns[b]).bit_count()
                        if w <= 4:
                            total += (5 - w) ** 2
                loads.append(total)
            return max(loads), sum(loads)

        ranked = []
        for pre, sig, grid, columns, metric, c4 in finalists[:8]:
            distance_upper, multiplicity = kernel_probe(columns, 48)
            next_weights = (kernel78(columns, metric)
                            if side == "A" else ())
            boundaries = boundary_exposure(columns)
            short = metric[:4] + (
                int(metric[4] > 0), int(metric[5] > 0))
            # Component screens provide protection guards and several
            # geometries. The old component BP benchmark is not called:
            # the final choice now evaluates the complete quantum pair.
            quality = (
                short + (max(0, 8 - distance_upper),)
                + boundaries + next_weights
                + (multiplicity, c4, metric[-1], sig)
            )
            ranked.append((quality, grid))
        if not ranked:
            raise ValueError("no screened mixed-row component")
        ranked.sort(key=lambda item: item[0])
        return ranked[:3]

    def fault_burden(check_rows, stabilizer_rows, nc, basis):
        """Joint CSS stress test, NOT the ASC memory experiment.

        Perfect final syndromes are decoded after attenuated, integrated
        gate exposure. Hook supports are random proper suffixes of actual
        stabilizer rows, not an invented ASC schedule. Stabilizer-equivalent
        residuals are successes. No measurement-noise history is modeled.
        """
        import numpy as np

        nr = len(check_rows)
        columns = transpose(check_rows, nc)
        edge_variables = []
        checks = [[] for _ in range(nr)]
        variables = [[] for _ in range(nc)]
        for v, syndrome in enumerate(columns):
            while syndrome:
                bit = syndrome & -syndrome
                c = bit.bit_length() - 1
                e = len(edge_variables)
                edge_variables.append(v)
                checks[c].append(e)
                variables[v].append(e)
                syndrome ^= bit

        ne = len(edge_variables)
        ev = np.asarray(edge_variables, dtype=np.int64)
        ce = np.full((nr, max(map(len, checks))), ne, dtype=np.int64)
        ve = np.full((nc, max(map(len, variables))), ne, dtype=np.int64)
        for c, edges in enumerate(checks):
            ce[c, :len(edges)] = edges
        for v, edges in enumerate(variables):
            ve[v, :len(edges)] = edges
        real = ce != ne
        check_variables = np.asarray(
            edge_variables + [nc], dtype=np.int64)[ce]

        # Full quantum degeneracy, using the opposite verified row span.
        pivots = {}
        for row in stabilizer_rows:
            while row:
                p = row.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = row
                    break
                row ^= old

        def logical_residual(word):
            while word:
                p = word.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    return True
                word ^= old
            return False

        shots = 64
        intensity = np.repeat(np.asarray((0.25, 0.45)), shots // 2)
        srng = np.random.default_rng(
            131057903 + 1009 * idx + 7919 * basis)
        total_degree = np.asarray([
            col.bit_count()
            for col in transpose(check_rows + stabilizer_rows, nc)
        ], dtype=float)
        single_mu = 6 * 0.002 * (2.0 / 3.0) * total_degree
        single_p = 0.5 * (
            1.0 - np.exp(-2.0 * intensity[:, None] * single_mu))
        errors = np.zeros((shots, nc + 1), dtype=bool)
        errors[:, :nc] = srng.random((shots, nc)) < single_p

        hook_load = np.zeros(nc, dtype=float)
        for row in stabilizer_rows:
            sites = []
            bits = row
            while bits:
                bit = bits & -bits
                sites.append(bit.bit_length() - 1)
                bits ^= bit
            if len(sites) < 2:
                continue
            hook_load[sites] += len(sites) + 1
            order = np.asarray(sites, dtype=np.int64)
            srng.shuffle(order)
            event_p = intensity * 6 * 0.002 * (len(sites) + 1) / 3.0
            events = np.flatnonzero(srng.random(shots) < event_p)
            for shot in events:
                cut = int(srng.integers(1, len(sites)))
                errors[shot, order[cut:]] ^= True

        # Marginal priors acknowledge hook exposure, but BP still does
        # not have the true correlated fault channel or relay/LSD cascade.
        marginal_mu = single_mu + 6 * 0.002 * hook_load / 6.0
        p = np.clip(0.5 * (
            1.0 - np.exp(-2.0 * intensity[:, None] * marginal_mu)),
            0.008, 0.22)
        prior = np.log((1.0 - p) / p)
        syndrome = np.logical_xor.reduce(
            errors[:, check_variables], axis=2)
        signs = 1.0 - 2.0 * syndrome.astype(float)
        v2c = np.empty((shots, ne + 1), dtype=float)
        v2c[:, :ne] = prior[:, ev]
        v2c[:, ne] = 30.0
        c2v = np.zeros_like(v2c)
        active = np.ones(shots, dtype=bool)
        ones = np.ones((shots, nr, 1), dtype=float)
        wrong, work = 0, 0
        early = None

        for iteration in range(40):
            work += int(active.sum())
            t = np.tanh(np.clip(v2c[:, ce], -18.0, 18.0) / 2.0)
            t[:, ~real] = 1.0
            left = np.concatenate(
                (ones, np.cumprod(t, axis=2)[:, :, :-1]), axis=2)
            suffix = np.cumprod(t[:, :, ::-1], axis=2)[:, :, ::-1]
            right = np.concatenate((suffix[:, :, 1:], ones), axis=2)
            outgoing = 2.0 * np.arctanh(np.clip(
                signs[:, :, None] * left * right,
                -0.999999, 0.999999))
            c2v[:, ce[real]] = outgoing[:, real]
            posterior = prior + c2v[:, ve].sum(axis=2)
            estimate = np.concatenate(
                (posterior < 0, np.zeros((shots, 1), dtype=bool)),
                axis=1)
            matched = np.logical_xor.reduce(
                estimate[:, check_variables], axis=2)
            settled = active & np.all(matched == syndrome, axis=1)
            for shot in np.flatnonzero(settled):
                residual = estimate[shot, :nc] ^ errors[shot, :nc]
                word = sum(1 << int(v) for v in np.flatnonzero(residual))
                wrong += int(logical_residual(word))
            active[settled] = False
            if iteration == 11:
                early = int(active.sum())
            if not active.any():
                if early is None:
                    early = 0
                break
            damping = 0.25 if iteration < 12 else 0.50
            proposal = posterior[:, ev] - c2v[:, :ne]
            v2c[:, :ne] = (
                damping * v2c[:, :ne] + (1.0 - damping) * proposal)

        remaining = int(active.sum())
        if early is None:
            early = remaining
        return ((wrong + remaining) / shots, wrong / shots,
                early / shots, work / (40.0 * shots))

    # Component spectra guard protection; the complete quantum pair
    # determines the final fault-bank outcome and the actual rate.
    for attempt in range(3):
        apool, bpool = optimize("A"), optimize("B")
        prepared = []
        serial = 0
        for ka, A in apool:
            for kb, B in bpool:
                hx, hz, n = lifted_matrices(G, A, B)
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if not (195 <= k <= 201):
                    continue
                assert n == 598
                assert max(row.bit_count() for row in hx + hz) <= W_MAX
                guard = (
                    tuple(max(x, y) for x, y in zip(ka[:6], kb[:6]))
                    + tuple(x + y for x, y in zip(ka[:6], kb[:6]))
                    + (ka[6] + kb[6],)
                )
                prepared.append((guard, serial, A, B, hx, hz, n, k))
                serial += 1
        if not prepared:
            continue
        prepared.sort(key=lambda item: item[:2])

        winner = None
        # Every pairing sees the same per-basis random event stream.
        # The benchmark is evaluated only on structurally guarded pairs.
        for entry in prepared[:5]:
            guard, serial, A, B, hx, hz, n, k = entry
            sx = fault_burden(hx, hz, n, 0)
            sz = fault_burden(hz, hx, n, 1)
            quality = guard + (
                max(sx[0], sz[0]), sx[0] + sz[0],
                max(sx[1], sz[1]), max(sx[2], sz[2]),
                sx[3] + sz[3], serial)
            if winner is None or quality < winner[0]:
                winner = (quality, entry, sx, sz)

        quality, entry, sx, sz = winner
        _, _, A, B, hx, hz, n, k = entry
        cells = {"A": A, "B": B}
        if not css_ok(hx, hz):
            raise ValueError("verified CSS commutation failed")
        for side, grid in cells.items():
            columns = component_columns(side, grid)
            assert transpose(
                verified_component(side, grid), len(columns)
            ) == columns
        expected_cnots = 6 * q * (
            11 * sum(map(sum, counts["A"]))
            + 7 * sum(map(sum, counts["B"])))
        assert expected_cnots == 25428 + 546 * budget
        assert 6 * sum(row.bit_count() for row in hx + hz) == expected_cnots
        return {
            "idx": idx, "cells": cells, "k": k,
            "geometry": batch, "contract": (budget, card),
            "cnots": expected_cnots, "joint_screen": quality,
            "fault_burden": (sx, sz),
        }
    raise ValueError("no joint-screened filling in the target rate interval")


def _q8_columns(rows, width):
    columns = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            columns[bit.bit_length()-1] |= 1 << r
            row ^= bit
    return columns


def _q8_component(G, side, grid):
    """Zero partners isolate precisely the scaffold's component action."""
    rows = (lifted_matrices(G, grid, [[[]]])[0] if side == "A"
            else lifted_matrices(G, [[[]]], grid)[1])
    width = len(grid[0])*G.q
    mask = (1 << width)-1
    return [row & mask for row in rows]


def _q8_cycle_data(columns, nr):
    """Exact Tanner four/six cycles and check connectivity."""
    from itertools import combinations
    adjacency = [dict() for _ in range(nr)]
    neighborhoods = []
    parent = list(range(nr))

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for syndrome in columns:
        nodes = []
        while syndrome:
            bit = syndrome & -syndrome
            nodes.append(bit.bit_length()-1)
            syndrome ^= bit
        neighborhoods.append(nodes)
        for a in nodes[1:]:
            parent[root(a)] = root(nodes[0])
        for a, b in combinations(nodes, 2):
            adjacency[a][b] = adjacency[a].get(b, 0)+1
            adjacency[b][a] = adjacency[b].get(a, 0)+1
    four = sum(v*(v-1)//2 for a, ns in enumerate(adjacency)
               for b, v in ns.items() if a < b)
    six = 0
    for a, ns in enumerate(adjacency):
        for b, ab in ns.items():
            if b <= a:
                continue
            for c, bc in adjacency[b].items():
                if c > b and c in ns:
                    six += ab*bc*ns[c]
    for ns in neighborhoods:
        for a, b, c in combinations(ns, 3):
            six -= (adjacency[a][b]+adjacency[a][c]
                    +adjacency[b][c]-2)
    assert six >= 0
    return four, six, len({root(a) for a in range(nr)})


def _q8_short(columns):
    """Collision zero-tests through six and small triple boundaries."""
    singles = {}
    one = two = three = four = five = six = boundary = 0
    for s in columns:
        one += int(s == 0)
        two += singles.get(s, 0)
        singles[s] = singles.get(s, 0)+1
    pairs = {}
    for i, x in enumerate(columns):
        for y in columns[:i]:
            s = x ^ y
            three += singles.get(s, 0)
            four += pairs.get(s, 0)
            pairs[s] = pairs.get(s, 0)+1
    triples = {}
    for i, x in enumerate(columns):
        for j in range(i):
            pair = x ^ columns[j]
            for h in range(j):
                s = pair ^ columns[h]
                five += pairs.get(s, 0)
                six += triples.get(s, 0)
                triples[s] = triples.get(s, 0)+1
                boundary += max(0, 5-s.bit_count())**2
    return (one, two, three, four, five, six), boundary


def _q8_sample(rng, columns, trials=24):
    """Sample genuine component kernel words; this gives upper bounds."""
    order = list(range(len(columns)))
    minimum = len(columns)+1
    short = set()
    for _ in range(trials):
        rng.shuffle(order)
        pivots, basis = {}, []
        for c in order:
            syndrome, word = columns[c], 1 << c
            while syndrome:
                p = syndrome.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                basis.append(word)
        for sweep in range(2):
            rng.shuffle(basis)
            for i, word in enumerate(basis):
                minimum = min(minimum, word.bit_count())
                if word.bit_count() <= 8:
                    short.add(word)
                for other in basis[:i]:
                    proposal = word ^ other
                    if not proposal:
                        continue
                    weight = proposal.bit_count()
                    minimum = min(minimum, weight)
                    if weight <= 8:
                        short.add(proposal)
                    if weight < word.bit_count():
                        word = proposal
                basis[i] = word
        if minimum <= 4:
            break
    loads = [sum(bool((word >> (8*j)) & 255) for word in short)
             for j in range(8)]
    return minimum, len(short), max(loads, default=0)


def _q8_fingerprint(rows, columns):
    """Rooted component Tanner invariant, conservatively deduplicated.

    Regular actions are transitive within each of the three check-row
    orbits. The multiset of three rooted fingerprints therefore represents
    all 24 check roots, each repeated eight times.
    """
    from collections import Counter
    import hashlib
    nr, nc = len(rows), len(columns)
    adjacency = [[] for _ in range(nr+nc)]
    for c, syndrome in enumerate(columns):
        while syndrome:
            bit = syndrome & -syndrome
            r = bit.bit_length()-1
            adjacency[r].append(nr+c)
            adjacency[nr+c].append(r)
            syndrome ^= bit
    fingerprints = []
    for root in range(0, nr, 8):
        colors = [0]*nr+[1]*nc
        colors[root] = 2
        digest = hashlib.sha256()
        for iteration in range(12):
            signatures = [
                (colors[v], tuple(sorted(colors[w] for w in neighbors)))
                for v, neighbors in enumerate(adjacency)]
            digest.update(repr(sorted(
                Counter(signatures).items())).encode("ascii"))
            palette = {s: i for i, s in enumerate(sorted(set(signatures)))}
            colors = [palette[s] for s in signatures]
        fingerprints.append(digest.hexdigest())
    return tuple(sorted(fingerprints))


def _q8_quotient(rows, width):
    """Linear canonical remainders modulo the actual stabilizer span."""
    pivots = {}
    for row in rows:
        while row:
            p = row.bit_length()-1
            old = pivots.get(p)
            if old is None:
                pivots[p] = row
                break
            row ^= old
    residual = [0]*width
    for c in range(width):
        row = pivots.get(c)
        if row is None:
            residual[c] = 1 << c
        else:
            tail = row ^ (1 << c)
            value = 0
            while tail:
                bit = tail & -tail
                value ^= residual[bit.bit_length()-1]
                tail ^= bit
            residual[c] = value
    return residual


def _q8_low4(checks, stabilizers, width):
    """Exact stabilizer-aware logical exclusion through weight four."""
    columns = _q8_columns(checks, width)
    labels = _q8_quotient(stabilizers, width)
    seen = {0: 0}
    for c, syndrome in enumerate(columns):
        old = seen.get(syndrome)
        if old is None:
            seen[syndrome] = labels[c]
        elif old != labels[c]:
            return True
    for c, syndrome in enumerate(columns):
        for d in range(c):
            s, label = syndrome ^ columns[d], labels[c] ^ labels[d]
            old = seen.get(s)
            if old is None:
                seen[s] = label
            elif old != label:
                return True
    return False


def _q8_graph_invariant(hx, hz, width):
    """Exact uncolored combined-Tanner invariants, not a GI certificate.

    Different returned tuples prove nonisomorphism. Equal tuples are
    conservatively discarded, even when they might describe different codes.
    """
    from collections import Counter
    rows = hx+hz
    nr = len(rows)
    overlap = [[0]*nr for _ in range(nr)]
    profiles = [Counter() for _ in rows]
    histogram = Counter()
    cx = cz = 0
    split = len(hx)
    for a, row in enumerate(rows):
        for b in range(a):
            value = (row & rows[b]).bit_count()
            overlap[a][b] = overlap[b][a] = value
            histogram[value] += 1
            profiles[a][value] += 1
            profiles[b][value] += 1
            cycle = value*(value-1)//2
            if a < split:
                cx += cycle
            elif b >= split:
                cz += cycle
    check_classes = Counter(
        (row.bit_count(), tuple(sorted(p.items())))
        for row, p in zip(rows, profiles))
    data_classes = Counter()
    for syndrome in _q8_columns(rows, width):
        nodes = []
        while syndrome:
            bit = syndrome & -syndrome
            nodes.append(bit.bit_length()-1)
            syndrome ^= bit
        anchored = sum(max(0, overlap[a][b]-1)
                       for i, a in enumerate(nodes) for b in nodes[:i])
        data_classes[len(nodes), anchored] += 1
    invariant = (tuple(sorted(histogram.items())),
                 tuple(sorted(check_classes.items())),
                 tuple(sorted(data_classes.items())))
    return invariant, (cx, cz)


def _q8_bp(checks, stabilizers, width, seed):
    """Common-stream noisy-data/spread probe, NOT ASC circuit memory.

    Perfect final syndromes, damped min-sum BP and actual stabilizer
    quotient labels are used. Measurement history and relay/LSD are absent.
    """
    import numpy as np

    def sites(row):
        out = []
        while row:
            bit = row & -row
            out.append(bit.bit_length()-1)
            row ^= bit
        return out

    neighborhoods = [sites(row) for row in checks]
    opposite = [sites(row) for row in stabilizers]
    edge_variables = []
    check_edges = []
    incident = [[] for _ in range(width)]
    for ns in neighborhoods:
        es = []
        for c in ns:
            edge = len(edge_variables)
            edge_variables.append(c)
            es.append(edge)
            incident[c].append(edge)
        check_edges.append(es)
    ne = len(edge_variables)
    ce = np.full((len(checks), max(map(len, check_edges))),
                 ne, dtype=np.int32)
    ve = np.full((width, max(map(len, incident))), ne, dtype=np.int32)
    for r, es in enumerate(check_edges):
        ce[r, :len(es)] = es
    for c, es in enumerate(incident):
        ve[c, :len(es)] = es
    valid = ce != ne
    ev = np.asarray(edge_variables, dtype=np.int32)
    cv = np.asarray(edge_variables+[width], dtype=np.int32)[ce]
    shots = 48
    intensity = np.repeat(np.asarray((0.30, 0.50)), shots//2)
    channel = np.random.default_rng(seed)
    degree = np.asarray([
        c.bit_count() for c in _q8_columns(checks+stabilizers, width)])
    mu = 6*0.002*(2.0/3.0)*degree
    rates = 0.5*(1-np.exp(-2*intensity[:, None]*mu[None, :]))
    errors = np.zeros((shots, width+1), dtype=bool)
    errors[:, :width] = channel.random((shots, width)) < rates
    hook_load = np.zeros(width)
    for ns in opposite:
        hook_load[ns] += 1
        events = np.flatnonzero(channel.random(shots)
                               < intensity*6*0.002/3.0)
        if len(ns) >= 2:
            for shot in events:
                pair = channel.choice(ns, 2, replace=False)
                errors[shot, pair] ^= True
    prior_p = np.clip(rates+0.004*intensity[:, None]*hook_load,
                      0.006, 0.20)
    prior = np.log((1-prior_p)/prior_p)
    target = np.logical_xor.reduce(errors[:, cv], axis=2)
    target_sign = np.where(target, -1.0, 1.0)
    messages = prior[:, ev].copy()
    outgoing = np.zeros((shots, ne+1))
    active = np.ones(shots, dtype=bool)
    labels = _q8_quotient(stabilizers, width)
    wrong = work = 0
    early = shots
    positions = np.arange(ce.shape[1])[None, None, :]
    for iteration in range(36):
        work += int(active.sum())
        padded = np.concatenate((messages, np.zeros((shots, 1))), axis=1)
        shaped = padded[:, ce]
        magnitude = np.where(valid[None, :, :], np.abs(shaped), np.inf)
        signs = np.where(shaped < 0, -1.0, 1.0)
        product = np.prod(signs, axis=2)*target_sign
        minimum_at = np.argmin(magnitude, axis=2)
        smallest = np.partition(magnitude, 1, axis=2)[:, :, :2]
        size = np.where(positions == minimum_at[:, :, None],
                        smallest[:, :, 1, None], smallest[:, :, 0, None])
        update = 0.80*product[:, :, None]*signs*size
        update = np.where(valid[None, :, :], update, 0.0)
        outgoing[:, :ne] = update[:, valid]
        posterior = prior+outgoing[:, ve].sum(axis=2)
        hard = np.concatenate((posterior < 0,
                               np.zeros((shots, 1), dtype=bool)), axis=1)
        matched = np.logical_xor.reduce(hard[:, cv], axis=2)
        settled = active & np.all(matched == target, axis=1)
        for shot in np.flatnonzero(settled):
            label = 0
            for c in np.flatnonzero(hard[shot, :width] ^ errors[shot, :width]):
                label ^= labels[int(c)]
            wrong += int(label != 0)
        active[settled] = False
        if iteration == 11:
            early = int(active.sum())
        if not active.any():
            break
        proposal = posterior[:, ev]-outgoing[:, :ne]
        messages = 0.30*messages+0.70*np.clip(proposal, -24, 24)
    remaining = int(active.sum())
    return wrong+remaining, remaining, early, work


def NEW_CTX(rng, G, idx):
    """Pure Q8 catalogue, indexed support pairs and invariant reservation."""
    import math
    from itertools import (permutations, combinations,
                           combinations_with_replacement)

    assert G.q == 8 and A_SHAPE == B_SHAPE == (3, 8)
    q, width = 8, 64

    def augmentation(grid):
        return gf2_rank([sum((t & 1) << j for j, t in enumerate(row))
                         for row in grid])

    def canonical(grid):
        return min(tuple(sorted(tuple(grid[i][j] for i in order)
                                for j in range(8)))
                   for order in permutations(range(3)))

    types = sorted({(1, 1, 1)} | set(permutations((2, 1, 0))))
    support_book = set()
    for columns in combinations_with_replacement(types, 8):
        grid = [[c[i] for c in columns] for i in range(3)]
        if tuple(map(sum, grid)) != (8, 8, 8):
            continue
        if any(row.count(2) > 3 for row in grid):
            continue
        if augmentation(grid) == 3:
            support_book.add(canonical(grid))
    support_book = sorted(support_book)
    assert len(support_book) >= 4

    def from_columns(columns):
        return [[c[i] for c in columns] for i in range(3)]

    base = [from_columns(support_book[
        j*(len(support_book)-1)//3]) for j in range(4)]
    reinforced = []
    for extra in (1, 2, 3):
        choices = {}
        for grid in base:
            zeros = [(i, j) for i in range(3) for j in range(8)
                     if grid[i][j] == 0]
            for selected in combinations(zeros, extra):
                if len({i for i, j in selected}) != extra:
                    continue
                if len({j for i, j in selected}) != extra:
                    continue
                trial = [list(row) for row in grid]
                for i, j in selected:
                    trial[i][j] = 1
                if augmentation(trial) == 3:
                    choices[canonical(trial)] = trial
        if not choices:
            raise ValueError("no full-augmentation reinforcement contract")
        reinforced.append(choices[min(choices)])
    supports = base+reinforced

    # Ten unordered balanced pairs and ten one-sided reinforcement pairs.
    # Additional batches consume unused class pairs from these same briefs.
    plans = list(combinations_with_replacement(range(4), 2))
    plans += [(a, b) for a in range(4, 7) for b in range(4)]
    plans = plans[:20]
    assert len(plans) == 20

    # Cache only helper-returned singleton blocks; no independent lift.
    actions = {}
    for g in G.elements:
        rows = _q8_component(G, "A", [[[g]]])
        actions[G.idx(g)] = _q8_columns(rows, q)

    def compose(grid):
        columns = [0]*width
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                for g in cell:
                    block = actions[G.idx(g)]
                    for t, value in enumerate(block):
                        columns[j*q+t] ^= value << (i*q)
        return columns

    def transpose(columns):
        rows = [0]*24
        for c, syndrome in enumerate(columns):
            while syndrome:
                bit = syndrome & -syndrome
                rows[bit.bit_length()-1] |= 1 << c
                syndrome ^= bit
        return rows

    # The three inverse classes of noncentral order-four elements.
    classes = (((1, 0), (3, 0)),
               ((0, 1), (2, 1)),
               ((1, 1), (3, 1)))

    def bank(number, counts):
        rr = random.Random(80382031+104729*number)
        sites = [(i, j) for i, row in enumerate(counts)
                 for j, t in enumerate(row) if t]

        def draw():
            grid = []
            for row in counts:
                directions = list(range(3))
                rr.shuffle(directions)
                out = []
                for t in row:
                    if not t:
                        out.append([])
                        continue
                    anchor = G.pick(rr)
                    if t == 1:
                        out.append([anchor])
                    else:
                        assert t == 2
                        relative = rr.choice(classes[directions.pop()])
                        out.append([anchor, G.mul(anchor, relative)])
                grid.append(out)
            return grid

        def analyze(grid):
            columns = compose(grid)
            cycles = _q8_cycle_data(columns, 24)
            energy = (80*cycles[0]+cycles[1]/8.0)/q
            return energy, columns, cycles

        pool, seen = [], set()

        def retain(grid, result):
            energy, columns, cycles = result
            signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                              for row in grid for cell in row)
            if signature in seen:
                return
            seen.add(signature)
            pool.append(((cycles[2]-1, cycles[0], cycles[1]),
                         rr.random(), grid, columns))
            pool.sort(key=lambda item: item[:2])
            del pool[32:]

        for trial in range(48):
            grid = draw()
            retain(grid, analyze(grid))
        starts = list(pool[:3])
        for coarse, tie, start, columns in starts:
            current = [[list(c) for c in row] for row in start]
            result = analyze(current)
            for step in range(260):
                proposal = [[list(c) for c in row] for row in current]
                i, j = rr.choice(sites)
                # Preserve row inverse classes but change cell voltages.
                shift = G.pick(rr)
                proposal[i][j] = [G.mul(shift, g)
                                  for g in proposal[i][j]]
                proposed = analyze(proposal)
                delta = proposed[0]-result[0]
                temperature = 10*(0.98**step)+0.10
                if delta <= 0 or rr.random() < math.exp(
                        -min(700.0, delta/temperature)):
                    current, result = proposal, proposed
                if step % 16 == 0:
                    retain(current, result)
            retain(current, result)

        finalists, fingerprints = [], set()
        for coarse, tie, grid, columns in pool:
            rows = transpose(columns)
            if coarse[0] or gf2_rank(rows) != 24:
                continue
            assert rows == _q8_component(G, "A", grid)
            identity = _q8_fingerprint(rows, columns)
            if identity in fingerprints:
                continue
            fingerprints.add(identity)
            short, boundary = _q8_short(columns)
            distance, vulnerable, peak = _q8_sample(rr, columns)
            quality = (int(any(short[:4])), int(short[4] > 0),
                       int(short[5] > 0), max(0, 8-distance),
                       vulnerable, peak, boundary, coarse[1], coarse[2])
            finalists.append((quality, identity, grid))
        finalists.sort(key=lambda item: item[:2])
        if not finalists:
            raise ValueError("no connected full-rank quaternion component")
        return finalists[:12]

    banks = [bank(i, counts) for i, counts in enumerate(supports)]

    def guard(a, b):
        return tuple(v for x, y in zip(a[:4], b[:4])
                     for v in (max(x, y), x+y))

    # Rebuild the same reservation prefix in every pure call. Reserving
    # every finalist means the final BP choice cannot reappear later.
    reserved = set()
    chosen = None
    for slot in range(idx+1):
        pa, pb = plans[slot % 20]
        alternatives = []
        for ia, a in enumerate(banks[pa]):
            for ib, b in enumerate(banks[pb]):
                if pa == pb and ia >= ib:
                    continue
                coarse = guard(a[0], b[0]) + (
                    max(a[0][4], b[0][4]), a[0][4]+b[0][4],
                    max(a[0][7], b[0][7]), a[0][7]+b[0][7])
                alternatives.append((coarse, a[1], b[1], a, b))
        alternatives.sort(key=lambda item: item[:3])
        available, local = [], set()
        for coarse, aid, bid, a, b in alternatives:
            A, B = a[2], b[2]
            hx, hz, n = lifted_matrices(G, A, B)
            k = n-gf2_rank(hx)-gf2_rank(hz)
            if n != 584 or k != 200:
                continue
            invariant, cycles = _q8_graph_invariant(hx, hz, n)
            if invariant in reserved or invariant in local:
                continue
            local.add(invariant)
            key = guard(a[0], b[0]) + (
                max(cycles), sum(cycles), a[0][4]+b[0][4],
                a[0][6]+b[0][6], aid, bid)
            available.append((key, invariant, A, B, hx, hz))
            if len(available) >= 9:
                break
        if not available:
            raise ValueError("unused invariant-distinct quaternion pair exhausted")
        available.sort(key=lambda item: item[0])
        chosen = available[:3]
        reserved.update(item[1] for item in chosen)

    winner = None
    for coarse, invariant, A, B, hx, hz in chosen:
        bad = int(_q8_low4(hx, hz, 584))+int(_q8_low4(hz, hx, 584))
        bx = _q8_bp(hx, hz, 584, 80385001+1009*idx)
        bz = _q8_bp(hz, hx, 584, 80385002+1009*idx)
        # Structural protection guards precede the uncalibrated decoder
        # test. Beyond the sampled weight-eight floor, do not reward
        # progressively larger, optimistic distance upper bounds.
        key = coarse[:8] + (bad,
            max(bx[0], bz[0]), bx[0]+bz[0],
            max(bx[1], bz[1]), bx[1]+bz[1],
            max(bx[2], bz[2]), bx[3]+bz[3]) + coarse[8:]
        if winner is None or key < winner[0]:
            winner = key, invariant, A, B, hx, hz, bx, bz
    key, invariant, A, B, hx, hz, bx, bz = winner
    sa = [[len(c) for c in row] for row in A]
    sb = [[len(c) for c in row] for row in B]
    assert augmentation(sa) == augmentation(sb) == 3
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert all(len(set(c)) == len(c)
               for grid in (A, B) for row in grid for c in row)
    assert css_ok(hx, hz)
    assert 584-gf2_rank(hx)-gf2_rank(hz) == 200
    assert max(row.bit_count() for row in hx+hz) <= W_MAX
    # B's right-lifted component is also verified, not inferred from A.
    assert gf2_rank(_q8_component(G, "B", B)) == 24
    cnots = 6*sum(row.bit_count() for row in hx+hz)
    assert cnots == 6*q*11*(sum(map(sum, sa))+sum(map(sum, sb)))
    return {"cells": {"A": A, "B": B}, "k": 200,
            "contract": plans[idx % 20], "screen": key,
            "invariant": invariant, "bp_probe": (bx, bz), "cnots": cnots}


def CELL(rng, G, side, i, j, ctx):
    """Return the precomputed full cell without changing shared state."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Equal (n,k,d2s) determines circuit performance: parent Z17 slots with (578,204,5) scored 3.691-4.347, and slot 50 was strongly Z-limited despite its weight-seven Z-type witnesses.
# REFUTED: Weight-eight witnesses in both bases imply balanced memory reliability: current parent slot 20 had 100/8250 Z failures versus 100/50250 X failures, with only diffuse block enrichment.
# REFUTED: Component short-kernel screening and d2s=8 suffice for low, balanced circuit failure in this Z13 bed: all five measured codes hit failure caps; slot 1 had 103/7500 X versus 101/33000 Z failures.
# REFUTED: The side-disjoint ratio book supplied twenty inequivalent designs: only eight distinct codes survived across sixty slots, with extensive isomorphic duplication.
# REFUTED: Removing mixed-sector four-cycles suffices to balance memory on this bed: all five measured codes remained X-limited and hit failure caps, scoring 4.984-5.135 despite exact cycle-separation checks.
# REFUTED: Primitive pure-binomial B gate exchange suffices to escape the roughly 5.1 fitness plateau: fitness was 5.0750, all five measured codes hit failure caps, and failures reached all 195 logical qubits.
# REFUTED: B27 reinforcement plus component-only BP selection improves this bed's incumbent: fitness fell to 4.7446; all measured codes had d2s=8, depth17 and 27612 CNOTs, yet X failures reached 109/3750 shots and every logical qubit.
# REFUTED: B23/B24 joint fault screening eliminates the Z13 bed's basis imbalance: fitness remained 5.0850, all five measured codes hit both failure caps, and X needed only 9000-19750 shots versus 40250-67250 Z shots despite weight-eight witnesses in both bases.
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
