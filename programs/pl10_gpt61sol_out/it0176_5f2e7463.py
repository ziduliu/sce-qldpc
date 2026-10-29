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
CONCEPT = ("For cyclic Z13 (metacyclic Z13:Z1, s=1), order 13, shapes (2,5)x(3,8), primitive pure-binomial B columns should free enough gate budget for degree-four A reinforcement to raise worse-basis circuit fitness above 5.0793.")
GROUP = ("Frozen L1 group: keep the exact Z13 implementation and order byte-for-byte. This bed scores 5.0793; size factor 46 admits q<=13 and gives n=598. The experiment isolates filling and gate-allocation effects, not a group advantage.")
SHAPES = ("Frozen (2,5)x(3,8), targeting cell 0.350: kb=195 and guaranteed rate 15/46=0.326087; final verified ranks retain 195<=k<=201. A has 65 variables/26 checks and B has 104 variables/39 checks. Parent failures affected every logical qubit, with diffuse Acol/Bcol enrichment up to 1.3x and mild Acol/Brow enrichment, not a demonstrated isolated hotspot.")
FILLING = ("Twenty count contracts cross five A reinforcement architectures with four B architectures. A has 16-18 terms and columns of degree three/four; B has 21-23 terms and one to three degree-two pure-binomial columns, at most one per check row. Other B columns have degree three and every B row loads seven/eight. Primitive doubletons individually close after thirteen lift steps, but their interactions are screened rather than assumed safe. Both CSS weights stay <=12; six-round CNOT budgets span 25194-28002, none equal the parent's 25974/26832. Canonical internal-ratio cards extend larger slot budgets without saved-code relabelings. Verified singleton actions accelerate multistart search; retain unavoidable-four-cycle minima, screen short kernels and exact A weight-seven/eight multiplicity, then small component boundaries and capped sampled witnesses. Target the parent's X failure fractions of roughly 0.7-1.0 percent versus lower Z fractions and collective logical flips; these screens do not simulate ASC or certify pL.")
FAMILY_NAME = "r035_m13r1s1_25x38_gateexchange_pureB_boundary_v3"

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
    """Twenty gate-exchange contracts; fresh canonical ratio cards per batch."""
    import math
    from itertools import combinations, permutations

    q = G.q
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    design, batch = idx % 20, idx // 20
    architecture, b_model = divmod(design, 4)

    # Balanced and skewed reinforcements have different internal loads.
    # In particular, some degree-four contracts cannot have C4=0.
    a_profiles = (
        ((2, 2, 2, 1, 1), (2, 1, 1, 2, 2)),
        ((3, 1, 2, 1, 1), (1, 2, 1, 2, 2)),
        ((2, 2, 2, 1, 1), (2, 2, 1, 2, 2)),
        ((3, 2, 1, 1, 1), (1, 2, 2, 2, 2)),
        ((3, 1, 2, 2, 1), (1, 3, 2, 1, 2)),
    )
    # Pure doubletons occur in distinct rows: two in one row would
    # create a universal weight-four commutativity word.
    b_columns = (
        ((2,0,0), (0,2,1), (2,1,0), (0,1,2),
         (1,1,1), (1,1,1), (1,1,1), (1,1,1)),
        ((2,0,0), (0,2,0), (0,1,2), (1,0,2),
         (1,1,1), (1,1,1), (1,1,1), (1,1,1)),
        ((2,0,0), (0,2,0), (0,0,2), (2,1,0),
         (0,2,1), (1,0,2), (1,1,1), (1,1,1)),
        ((2,0,0), (0,2,0), (0,0,2), (2,1,0),
         (0,1,2), (1,2,0), (1,0,2), (1,1,1)),
    )[b_model]
    sb = [[column[i] for column in b_columns] for i in range(rb)]
    counts = {"A": [list(row) for row in a_profiles[architecture]],
              "B": sb}
    pure_sites = [(i, j) for j, column in enumerate(b_columns)
                  if sum(column) == 2
                  for i, t in enumerate(column) if t == 2]
    assert len({i for i, j in pure_sites}) == len(pure_sites)
    assert all(sum(row) in (7, 8) for row in sb)
    assert sum(map(sum, sb)) in (21, 22, 23)
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
    book_rng = random.Random(130038947 + 1009 * design)
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

        # Batch cards change relative pure-cycle directions as well as
        # A's canonical geometry; no completed filling is transformed.
        tb = [[None if t else () for t in row] for row in sb]
        pure_steps = ((1, 1, 1), (1, 2, 3), (1, 3, 4))[number % 3]
        for t, (i, j) in enumerate(pure_sites):
            tb[i][j] = (0, pure_steps[t])
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
        raise ValueError("no unused primitive-cycle ratio card")

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
            pool.append((quick, sig, grid, columns, adj))
            pool.sort(key=lambda item: item[:2])
            del pool[10:]

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

        best = None
        for pre, sig, grid, columns, metric, c4 in finalists[:4]:
            distance_upper, multiplicity = kernel_probe(columns, 96)
            next_weights = (kernel78(columns, metric)
                            if side == "A" else ())
            boundaries = boundary_exposure(columns)
            # Short words and rank loss precede uncalibrated BP surrogates.
            # Cap the sampled protection reward rather than chase optimistic
            # upper bounds. Pure-binomial interactions remain screened.
            quality = (
                metric[:-1] + next_weights
                + (max(0, 8 - distance_upper),)
                + boundaries + (multiplicity, c4, metric[-1], sig)
            )
            if best is None or quality < best[0]:
                best = quality, grid
        if best is None:
            raise ValueError("no screened component for gate-exchange card")
        return best[1]

    # Retry labels within the SAME count/ratio contract. Full CSS ranks,
    # not an augmentation prediction, determine the measured rate.
    for attempt in range(3):
        cells = {"A": optimize("A"), "B": optimize("B")}
        hx, hz, n = lifted_matrices(G, cells["A"], cells["B"])
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (195 <= k <= 201):
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified CSS commutation failed")
        assert n == 598
        assert max(row.bit_count() for row in hx + hz) <= W_MAX

        def full_four_cycles(rows):
            return sum(t * (t - 1) // 2
                       for i, row in enumerate(rows)
                       for other in rows[:i]
                       for t in [(row & other).bit_count()])

        # Confirm cached singleton compositions screen precisely the
        # helper-built components used in the final construction.
        for side, grid in cells.items():
            columns = component_columns(side, grid)
            assert transpose(
                verified_component(side, grid), len(columns)
            ) == columns

        ac4 = cycle_data(component_columns("A", cells["A"]))[0]
        bc4 = cycle_data(component_columns("B", cells["B"]))[0]
        xc4, zc4 = full_four_cycles(hx), full_four_cycles(hz)
        # Mixed-sector cycles are now permitted and reported honestly;
        # side-disjoint ratios are no longer a construction requirement.
        mixed = (xc4 - cb * ac4 - ra * bc4,
                 zc4 - rb * ac4 - ca * bc4)
        assert min(mixed) >= 0
        a_terms = sum(map(sum, counts["A"]))
        b_terms = sum(map(sum, counts["B"]))
        expected_cnots = 6 * q * (11 * a_terms + 7 * b_terms)
        assert expected_cnots in (
            25194, 25740, 26286, 26052, 26598, 27144,
            26910, 27456, 28002)
        assert 6 * sum(row.bit_count() for row in hx + hz) == expected_cnots
        return {"idx": idx, "cells": cells, "k": k,
                "geometry": batch, "contract": (architecture, b_model),
                "four_cycles": (xc4, zc4), "mixed_cycles": mixed,
                "cnots": expected_cnots}
    raise ValueError("no verified filling in the target rate interval")


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
