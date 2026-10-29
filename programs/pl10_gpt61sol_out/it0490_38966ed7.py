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
CONCEPT = ("Cyclic C19, order 19, shapes (2,5)x(2,5): balanced degree-two/three components, reduced extraction exposure and independent noisy-syndrome holdouts should improve the approximately 6.2 frontier fitness in cell 0.300.")
GROUP = ("Jump from the nonabelian order-36 parent to cyclic C19. Size factor 29 allows q<=20, making 19 the largest admissible prime lift. The C19 equal-(2,5) bed has reached about 5.7, so an improvement is not assumed. Algebraically ord_19(2)=18: a nonzero polynomial of fewer than nineteen terms with odd augmentation is a unit in F2[C19]. Thus prescribed monomial/trinomial blocks are invertible, and nontrivial binomial ratios have full order nineteen.")
SHAPES = ("Equal A=B=(2,5): n=551, kb=171 and guaranteed rate 9/29=0.310345 lie inside target cell 0.300. Each component has 38 checks on 95 variables and dimension at least 57. Require full component ranks and complete helper-built k=171, retaining zero k-kb. The parent's seven observed Z failures across five measured codes flipped 227 logical qubits in total; enrichments varied among Acol/Bcol data and Acol/Brow checks, while all X measurements had zero failures. These sparse maps do not establish a persistent physical hotspot.")
FILLING = ("Canonical unordered support pairs use fourteen/fifteen terms per component, row totals seven/eight, column degrees two/three and at least three mixed-row columns. Both CSS row-weight sums are <=11; six-round CNOT counts are 22344, 23142 or 23940 versus the parent's 29592/30672. Each slot owns fresh canonical relative-support geometries, quotiented by cyclic multipliers and obvious row/column images; extra harness batches consume additional geometries rather than saved-code relabelings. Within-row difference separation removes intrinsic four-cycles. Optimize origins using helper-derived actions, then prioritize exact kernel exclusion through six, exact anchored weight-seven counts and sampled protection capped at ten. Two-interval noisy-syndrome min-sum training and independent holdouts distinguish settled miscorrections from unresolved syndromes. Complete ranks, CSS, projections and quantum-logical exclusion through four are checked before emission. The probes omit ASC ordering and relay/LSD and are not measured circuit pL; the evaluator remains authoritative for archive and Tanner equivalence.")
FAMILY_NAME = "r030_C19_25x25_degree23_temporalholdout_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 19

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C19 requires order nineteen")
    els = [(a,) for a in range(q)]

    def red(g):
        return (int(g[0]) % q,)

    def mul(g, h):
        return ((g[0] + h[0]) % q,)

    def inv(g):
        return ((-g[0]) % q,)

    def idx(g):
        return red(g)[0]

    return dict(elements=els, ident=(0,), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 5)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


# A columns have weight >=4; rows have weight <=7.
_SUPPORT_PROFILES_A = (
    (2, 2, 2, 2, 2, 2),
    (3, 2, 2, 2, 2, 2),
    (1, 2, 2, 3, 2, 2),
    (1, 2, 3, 3, 2, 1),
    (1, 1, 3, 3, 3, 1),
)

# B columns have weight >=3. Profiles 0..2 preserve column weight four;
# profiles 3..5 trade one row-weight unit for different column protection.
_SUPPORT_PROFILES_B = (
    (2, 2, 2, 2, 2, 2, 2, 2),
    (1, 3, 2, 2, 3, 1, 2, 2),
    (1, 1, 3, 3, 3, 3, 1, 1),
    (1, 2, 2, 2, 2, 1, 2, 2),
    (1, 1, 2, 3, 2, 2, 2, 1),
    (1, 2, 2, 2, 2, 1, 1, 3),
)

# Twenty deliberate ordered designs: A and B now have different shapes.
# A profile 1 has a weight-five column, so it only pairs with B rows <=7.
_PROFILE_PAIRS = (
    (0, 0), (0, 1), (0, 2),
    (2, 0), (2, 1), (2, 2),
    (3, 0), (3, 1), (3, 2),
    (4, 0), (4, 1), (4, 2),
    (0, 3), (1, 3), (2, 3), (3, 3),
    (0, 4), (1, 4), (0, 5), (1, 5),
)


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


def _small_kernel_word(rows, ncols):
    """Exact meet-in-the-middle test for a nonzero kernel word <=6."""
    columns = [0] * ncols
    for r, row in enumerate(rows):
        bit = 1 << r
        while row:
            low = row & -row
            columns[low.bit_length() - 1] |= bit
            row ^= low

    # Each subset of size <=3 is visited once. Equal syndromes of two
    # different subsets give a nonzero kernel word of weight <=6.
    seen = {0}
    for value in columns:
        if value in seen:
            return True
        seen.add(value)
    for i in range(ncols):
        for j in range(i):
            value = columns[i] ^ columns[j]
            if value in seen:
                return True
            seen.add(value)
    for i in range(ncols):
        for j in range(i):
            pair = columns[i] ^ columns[j]
            for h in range(j):
                value = pair ^ columns[h]
                if value in seen:
                    return True
                seen.add(value)
    return False


def _retired_dic9_CTX(rng, G, idx):
    """Inactive parent-bed search, retained only for provenance."""
    import math
    from itertools import combinations, combinations_with_replacement

    q = G.q
    assert q == 36
    assert A_SHAPE == (2, 3) and B_SHAPE == (2, 4)
    sa = [[2, 2, 2], [2, 2, 2]]
    b_profiles = (
        ((1, 2, 2, 2), (2, 1, 2, 2)),
        ((1, 1, 3, 2), (2, 2, 1, 2)),
        ((1, 2, 1, 3), (2, 1, 3, 1)),
        ((1, 2, 2, 2), (2, 1, 1, 2)),
    )
    mask_number, architecture = divmod(idx, len(b_profiles))
    sb = [list(row) for row in b_profiles[architecture]]
    counts = {"A": sa, "B": sb}
    assert _grid_weights(sa, sb) <= min(11, W_MAX)
    assert _no_dead_lines(sa, sb)
    assert all(0 <= w <= CELL_MAX
               for support in counts.values() for row in support for w in row)

    # A primitive rotation and an odd-y ratio generate the whole group.
    # Normalize each odd ratio to y; independent column conjugation
    # can remove its x exponent. Rotation directions remain modulo
    # inversion, global x->x^u, and component row/column permutations.
    # This catalogue is not a complete Tanner-isomorphism classifier.
    odd = G.idx((0, 1))
    rotations = [G.idx((a, b))
                 for b in (0, 2) for a in (1, 2, 4)]
    column_types = sorted(
        [(odd, r) for r in rotations] + [(r, odd) for r in rotations])

    def canonical_columns(cols):
        images = []
        for unit in (1, 2, 4, 5, 7, 8):
            image = []
            for column in cols:
                pair = []
                for r in column:
                    if r == odd:
                        pair.append(odd)
                    else:
                        a, b = G.elements[r]
                        value = (unit*a) % 9
                        pair.append(G.idx((min(value, 9-value), b)))
                image.append(tuple(pair))
            images.append(tuple(sorted(image)))
            images.append(tuple(sorted((b, a) for a, b in image)))
        return min(images)

    keys = set()
    for cols in combinations_with_replacement(column_types, 3):
        flat = [r for col in cols for r in col if r != odd]
        if {G.elements[r][1] for r in flat} != {0, 2}:
            continue
        # Both rows mix rotational and quotient-changing cells.
        if any(not 0 < sum(col[i] == odd for col in cols) < 3
               for i in range(2)):
            continue
        keys.add(canonical_columns(cols))

    masks = sorted(keys, key=lambda key: (
        -len({G.elements[r][0] for col in key for r in col if r != odd}),
        sum(G.elements[r][1] == 2
            for col in key for r in col if r != odd),
        key))
    if mask_number >= len(masks):
        raise ValueError("Unused generating-column geometry book exhausted")
    chosen = masks[mask_number]
    a_deltas = tuple(tuple(G.elements[col[i]] for col in chosen)
                     for i in range(2))

    orders = []
    for g in G.elements:
        h = G.e
        for length in range(1, q + 1):
            h = G.mul(h, g)
            if G._key(h) == G._key(G.e):
                orders.append(length)
                break
        else:
            raise ValueError("Element order missing")
    palettes = {
        length: [g for g in G.elements if orders[G.idx(g)] == length]
        for length in (4, 9, 18)
    }

    def generated_size(generators):
        reached, todo = {G.idx(G.e)}, [G.e]
        steps = list(generators) + [G.inv(g) for g in generators]
        while todo:
            g = todo.pop()
            for h in steps:
                image = G.mul(g, h)
                r = G.idx(image)
                if r not in reached:
                    reached.add(r)
                    todo.append(image)
        return len(reached)

    assert all(generated_size((a_deltas[0][j], a_deltas[1][j])) == q
               for j in range(3))

    triangle_length = 9 if (mask_number + architecture) % 2 == 0 else 18
    triangles = []
    nonidentity = [g for g in G.elements if G._key(g) != G._key(G.e)]
    for u, v in combinations(nonidentity, 2):
        if G._key(G.mul(u, v)) == G._key(G.mul(v, u)):
            continue
        signature = sorted((
            orders[G.idx(u)], orders[G.idx(v)],
            orders[G.idx(G.mul(G.inv(u), v))],
        ))
        if signature == sorted((4, 4, triangle_length)):
            triangles.append((G.e, u, v))
    assert triangles

    def draw_cell(side, i, j):
        weight = counts[side][i][j]
        if not weight:
            return []
        base = G.pick(rng)
        if weight == 1:
            return [base]
        if weight == 2:
            if side == "A":
                delta = a_deltas[i][j]
            else:
                if sb[0][j] == sb[1][j] == 2:
                    length = 4 if i == 0 else (
                        (9, 18)[(mask_number+j) % 2])
                else:
                    length = (9, 18, 4)[
                        (i+j+mask_number+architecture) % 3]
                delta = rng.choice(palettes[length])
            return [base, G.mul(base, delta)]
        return [G.mul(base, g) for g in rng.choice(triangles)]

    def draw_grid(side):
        return [[draw_cell(side, i, j) for j in range(len(row))]
                for i, row in enumerate(counts[side])]

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def columns(rows, width):
        out = [0] * width
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= tag
                row ^= bit
        return out

    # Cache only actions returned by the verified helper. Final component
    # projections are checked against the complete lift before emission.
    block_mask = (1 << q) - 1
    actions = {"A": {}, "B": {}}
    for g in G.elements:
        hx, _, _ = lifted_matrices(G, [[[g]]], [[[]]])
        _, hz, _ = lifted_matrices(G, [[[]]], [[[g]]])
        actions["A"][G.idx(g)] = tuple(row & block_mask for row in hx)
        actions["B"][G.idx(g)] = tuple(row & block_mask for row in hz)

    def component(grid, side):
        rows = []
        for grid_row in grid:
            out = [0] * q
            for j, cell in enumerate(grid_row):
                for g in cell:
                    action = actions[side][G.idx(g)]
                    for t, value in enumerate(action):
                        out[t] ^= value << (j*q)
            rows.extend(out)
        return rows

    def geometry(rows, width):
        adjacency = [0] * len(rows)
        four = 0
        for i, row in enumerate(rows):
            for j in range(i):
                overlap = (row & rows[j]).bit_count()
                four += overlap * (overlap - 1) // 2
                if overlap:
                    adjacency[i] |= 1 << j
                    adjacency[j] |= 1 << i

        reached, todo = {0}, [0]
        while todo:
            i = todo.pop()
            bits = adjacency[i]
            while bits:
                bit = bits & -bits
                j = bit.bit_length() - 1
                bits ^= bit
                if j not in reached:
                    reached.add(j)
                    todo.append(j)

        triangles_ = 0
        for i in range(len(rows)):
            lower = adjacency[i] & ((1 << i) - 1)
            while lower:
                bit = lower & -lower
                j = bit.bit_length() - 1
                triangles_ += (adjacency[i] & adjacency[j]
                               & ((1 << j) - 1)).bit_count()
                lower ^= bit
        cols = columns(rows, width)
        stars = sum(d*(d-1)*(d-2)//6 for d in map(int.bit_count, cols))
        # Exact without C4; otherwise only a proposal-energy heuristic.
        six = max(0, triangles_ - stars)
        duplicates = width - len(set(cols)) + int(0 in cols)
        return (len(rows) - len(reached), duplicates, four, six), cols

    def energy(cost):
        return (5000*cost[0] + 2000*cost[1]
                + 100*cost[2] + cost[3]/4) / q

    def capacity_stress(rows, cols, side):
        """Common-stream min-sum stress, not a circuit-pL estimator.

        Perfect syndromes isolate component iterative-decoder difficulty.
        A wrong settled kernel word and an unresolved syndrome are counted
        separately: the actual evaluator has a different decoder cascade.
        """
        width = len(cols)
        edge_columns, check_edges = [], []
        for row in rows:
            edges = []
            while row:
                bit = row & -row
                edges.append(len(edge_columns))
                edge_columns.append(bit.bit_length()-1)
                row ^= bit
            check_edges.append(edges)

        p = 0.055 if side == "A" else 0.035
        prior = math.log((1-p)/p)
        srng = random.Random(360941117 + 1009*idx + int(side == "B"))
        patterns = []
        for trial in range(48):
            if trial < 24:
                sites = [c for c in range(width) if srng.random() < p]
            else:
                weight = ((4 + trial % 5) if side == "A"
                          else (3 + trial % 4))
                sites = srng.sample(range(width), weight)
            error = syndrome = 0
            for c in sites:
                error |= 1 << c
                syndrome ^= cols[c]
            patterns.append((error, syndrome))

        wrong = unresolved = effort = 0
        for error, syndrome in patterns:
            if not syndrome:
                wrong += int(bool(error))
                continue
            messages = [prior] * len(edge_columns)
            incoming = [0.0] * len(edge_columns)
            settled = False
            for iteration in range(24):
                for r, edges in enumerate(check_edges):
                    sign = -1 if (syndrome >> r) & 1 else 1
                    first = second = 50.0
                    smallest = -1
                    for edge in edges:
                        value = messages[edge]
                        if value < 0:
                            sign = -sign
                        magnitude = abs(value)
                        if magnitude < first:
                            first, second = magnitude, first
                            smallest = edge
                        elif magnitude < second:
                            second = magnitude
                    for edge in edges:
                        magnitude = second if edge == smallest else first
                        direction = -sign if messages[edge] < 0 else sign
                        proposed = 0.80 * direction * magnitude
                        incoming[edge] = (
                            0.15*incoming[edge] + 0.85*proposed)

                beliefs = [prior] * width
                for edge, c in enumerate(edge_columns):
                    beliefs[c] += incoming[edge]
                estimate = observed = 0
                for c, value in enumerate(beliefs):
                    if value < 0:
                        estimate |= 1 << c
                        observed ^= cols[c]
                effort += 1
                if observed == syndrome:
                    wrong += int(estimate != error)
                    settled = True
                    break
                for edge, c in enumerate(edge_columns):
                    value = beliefs[c] - incoming[edge]
                    messages[edge] = max(-12.0, min(12.0, value))
            unresolved += int(not settled)
        return wrong, unresolved, effort

    def sampled_words(cols, side):
        """Actual kernel witnesses; the sampled minimum is an upper bound."""
        srng = random.Random(360913731 + 1009*idx + int(side == "B"))
        order = list(range(len(cols)))
        minimum = len(cols) + 1
        short = set()

        def offer(word):
            nonlocal minimum
            if not word:
                return
            weight = word.bit_count()
            minimum = min(minimum, weight)
            if weight <= 16 and len(short) < 8192:
                short.add(word)

        for trial in range(48):
            srng.shuffle(order)
            pivots, basis = {}, []
            for c in order:
                syndrome, word = cols[c], 1 << c
                while syndrome:
                    p = syndrome.bit_length() - 1
                    previous = pivots.get(p)
                    if previous is None:
                        pivots[p] = (syndrome, word)
                        break
                    syndrome ^= previous[0]
                    word ^= previous[1]
                if not syndrome:
                    basis.append(word)
                    offer(word)
            for i, word in enumerate(basis):
                for other in basis[:i]:
                    offer(word ^ other)
            for word in sorted(basis, key=int.bit_count)[:6]:
                for sweep in range(2):
                    best = word
                    for other in basis:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < best.bit_count():
                            best = candidate
                    if best == word:
                        break
                    word = best
                    offer(word)
            if minimum <= 6:
                break

        loads = [0] * (len(cols)//q)
        low12 = burden = 0
        for word in short:
            weight = word.bit_count()
            low12 += int(weight <= 12)
            cost = (17 - weight)**2
            burden += cost
            for j in range(len(loads)):
                loads[j] += cost * (
                    (word >> (j*q)) & block_mask).bit_count()
        return minimum, low12, burden, max(loads, default=0)

    def triple_exposure(cols):
        """Exact small-boundary statistic; not a trapping-set certificate."""
        burden = 0
        loads = [0] * (len(cols)//q)
        opportunities = [0] * len(loads)
        for i in range(len(cols)):
            for j in range(i):
                pair = cols[i] ^ cols[j]
                for h in range(j):
                    blocks = (i//q, j//q, h//q)
                    for block in blocks:
                        opportunities[block] += 1
                    weight = (pair ^ cols[h]).bit_count()
                    if weight <= 5:
                        cost = (6 - weight)**2
                        burden += cost
                        for block in blocks:
                            loads[block] += cost
        peak = max(b/max(1, o) for b, o in zip(loads, opportunities))
        return burden/max(1, sum(opportunities)//3), peak

    def optimize(side, extra=0):
        width = len(counts[side][0])*q
        sites = [(i, j) for i, row in enumerate(counts[side])
                 for j, weight in enumerate(row) if weight]
        strata, seen = {}, set()
        serial = 0

        def remember(grid, rows, cost, cols):
            nonlocal serial
            signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                              for row in grid for cell in row)
            if signature in seen:
                return
            seen.add(signature)
            serial += 1
            deficit = len(rows) - gf2_rank(rows)
            entry = (cost, serial, clone(grid), list(rows), list(cols))
            bucket = strata.setdefault(deficit, [])
            bucket.append(entry)
            bucket.sort(key=lambda item: item[:2])
            del bucket[2:]

        for restart in range(3):
            grid = draw_grid(side)
            rows = component(grid, side)
            cost, cols = geometry(rows, width)
            remember(grid, rows, cost, cols)
            for step in range(180 + extra):
                i, j = rng.choice(sites)
                old = grid[i][j]
                grid[i][j] = draw_cell(side, i, j)
                proposed_rows = component(grid, side)
                proposed, proposed_cols = geometry(proposed_rows, width)
                delta = energy(proposed) - energy(cost)
                temperature = 12.0*(0.979**step) + 0.10
                if delta <= 0 or rng.random() < math.exp(
                        -min(700.0, delta/temperature)):
                    rows, cost, cols = proposed_rows, proposed, proposed_cols
                    if step % 12 == 0 or delta < 0:
                        remember(grid, rows, cost, cols)
                else:
                    grid[i][j] = old
            remember(grid, rows, cost, cols)

        # Preserve some rank diversity: component deficits do not decide
        # the complete nonabelian CSS rate.
        candidates = [bucket[0] for bucket in strata.values()]
        candidates.sort(key=lambda item: item[:2])
        candidates = candidates[:4]
        if len(candidates) < 4:
            extras = [entry for bucket in strata.values()
                      for entry in bucket[1:]]
            extras.sort(key=lambda item: item[:2])
            candidates.extend(extras[:4-len(candidates)])

        refined = []
        target = 16 if side == "A" else 12
        for cost, number, grid, rows, cols in candidates:
            bad = int(_small_kernel_word(rows, width))
            d, low12, burden, load = sampled_words(cols, side)
            boundary, peak = triple_exposure(cols)
            stress = capacity_stress(rows, cols, side)
            item = {
                "grid": grid, "rows": rows, "cost": cost, "bad": bad,
                "d": d, "gap": max(0, target-d), "low12": low12,
                "burden": burden, "load": load,
                "boundary": boundary, "peak": peak, "stress": stress,
            }
            key = (cost[0], cost[1], bad, item["gap"], low12,
                   stress[0]+stress[1], stress[0],
                   peak, boundary, burden, load, cost[2], cost[3],
                   stress[2], number)
            refined.append((key, item))
        refined.sort(key=lambda entry: entry[0])
        return [item for _, item in refined]

    def logical4(checks, stabilizers, width):
        """Exact existence of a nontrivial quantum logical through four."""
        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                previous = pivots.get(p)
                if previous is None:
                    pivots[p] = row
                    break
                row ^= previous

        def nontrivial(word):
            while word:
                p = word.bit_length() - 1
                previous = pivots.get(p)
                if previous is None:
                    return True
                word ^= previous
            return False

        cols = columns(checks, width)
        seen = {0: 0}
        for i, value in enumerate(cols):
            word = 1 << i
            previous = seen.get(value)
            if previous is None:
                seen[value] = word
            elif nontrivial(word ^ previous):
                return True
        for i, value in enumerate(cols):
            for j in range(i):
                syndrome = value ^ cols[j]
                word = (1 << i) | (1 << j)
                previous = seen.get(syndrome)
                if previous is None:
                    seen[syndrome] = word
                elif nontrivial(word ^ previous):
                    return True
        return False

    def fault_bundles(hx, hz, width):
        """Complete-code correlated-fault opportunity surrogate.

        Inspect every pair/triple in each opposite-basis check support.
        There is no assumed ASC gate order. Count low syndrome boundaries
        and normalize block burden by inspected opportunities.
        """
        rates, peaks = [], []
        for checks, opposite in ((hx, hz), (hz, hx)):
            cols = columns(checks, width)
            burden = [0] * (width//q)
            opportunities = [0] * len(burden)
            total = inspected = 0
            for row in opposite:
                neighbors = []
                while row:
                    bit = row & -row
                    neighbors.append(bit.bit_length() - 1)
                    row ^= bit
                for size in (2, 3):
                    for bundle in combinations(neighbors, size):
                        syndrome = 0
                        for c in bundle:
                            syndrome ^= cols[c]
                            opportunities[c//q] += 1
                        inspected += 1
                        weight = syndrome.bit_count()
                        if weight <= 6:
                            cost = (7-weight)**2
                            total += cost
                            for c in bundle:
                                burden[c//q] += cost
            rates.append(total/max(1, inspected))
            peaks.append(max(
                b/max(1, o) for b, o in zip(burden, opportunities)))
        return max(peaks), max(rates), sum(rates)

    options = []
    for attempt in range(2):
        apool = optimize("A", 40*attempt)
        bpool = optimize("B", 40*attempt)
        for a in apool:
            for b in bpool:
                A, B = a["grid"], b["grid"]
                hx, hz, n = lifted_matrices(G, A, B)
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if n != 576 or not 73 <= k <= 100:
                    continue
                prefix = (
                    a["cost"][0] + b["cost"][0],
                    a["cost"][1] + b["cost"][1],
                    a["bad"] + b["bad"],
                    a["gap"], b["gap"],
                    a["low12"] + b["low12"],
                    a["stress"][0] + a["stress"][1],
                    a["stress"][0],
                    b["stress"][0] + b["stress"][1],
                    b["stress"][0],
                    max(0, 82-k), -k,
                )
                tail = (
                    max(a["peak"], b["peak"]),
                    a["boundary"] + b["boundary"],
                    a["burden"] + b["burden"],
                    max(a["load"], b["load"]),
                    a["cost"][2] + b["cost"][2],
                    a["cost"][3] + b["cost"][3],
                    a["stress"][2] + b["stress"][2],
                )
                options.append((prefix, tail, A, B, hx, hz, k,
                                a["rows"], b["rows"]))
        if options:
            break

    # If annealing misses the rate stratum, search fresh matrices under
    # the same count/order contract. Never repair emitted cells or reuse
    # a selected code by relabeling it.
    if not options:
        for trial in range(192):
            A, B = draw_grid("A"), draw_grid("B")
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if n != 576 or not 73 <= k <= 100:
                continue
            ar, br = _component_rows(hx, hz, q)
            ac, av = geometry(ar, 3*q)
            bc, bv = geometry(br, 4*q)
            prefix = (ac[0]+bc[0], ac[1]+bc[1])
            tail = (ac[2]+bc[2], ac[3]+bc[3], -k, trial)
            options.append((prefix, tail, A, B, hx, hz, k, ar, br))
            options.sort(key=lambda entry: entry[:2])
            del options[6:]
        if not options:
            raise ValueError("Mixed-order contract missed verified cell0.150")

    options.sort(key=lambda entry: entry[:2])
    finalists = []
    for number, option in enumerate(options[:6]):
        prefix, tail, A, B, hx, hz, k, ar, br = option
        if logical4(hx, hz, 576) or logical4(hz, hx, 576):
            continue
        exposure = fault_bundles(hx, hz, 576)
        key = prefix + exposure + tail + (number,)
        finalists.append((key, A, B, hx, hz, k, ar, br, exposure))
    if not finalists:
        raise ValueError("Every target-rate finalist had a logical through four")

    finalists.sort(key=lambda entry: entry[0])
    key, A, B, hx, hz, k, ar, br, exposure = finalists[0]
    actual_a, actual_b = _component_rows(hx, hz, q)
    assert actual_a == ar and actual_b == br
    assert component(A, "A") == actual_a
    assert component(B, "B") == actual_b
    # Confirm the actual isolated-column kernels using verified rows.
    # A binomial column's constant word is present; rank35 makes it
    # the only nonzero kernel word. A singleton B block is invertible.
    for j in range(3):
        isolated = [(row >> (j*q)) & block_mask for row in actual_a]
        assert gf2_rank(isolated) == q-1
        assert all(row.bit_count() % 2 == 0 for row in isolated)
        for i in range(2):
            ratio = G.mul(G.inv(A[i][j][0]), A[i][j][1])
            assert G._key(ratio) == G._key(a_deltas[i][j])
    for j in range(4):
        isolated = [(row >> (j*q)) & block_mask for row in actual_b]
        expected_rank = q if 1 in (sb[0][j], sb[1][j]) else q-1
        assert gf2_rank(isolated) == expected_rank
        if expected_rank == q-1:
            assert all(row.bit_count() % 2 == 0 for row in isolated)
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) <= min(11, W_MAX)
    expected_cnots = (72 + 5*sum(map(sum, sb)))*q*6
    assert 6*sum(row.bit_count() for row in hx + hz) == expected_cnots
    assert expected_cnots in (29592, 30672)
    assert 0.125 < k/576 < 0.175
    assert all(len(cell) == counts[side][i][j]
               and len({G._key(g) for g in cell}) == len(cell)
               for side, grid in (("A", A), ("B", B))
               for i, row in enumerate(grid) for j, cell in enumerate(row))
    return {"A": A, "B": B, "k": k, "cnots": expected_cnots,
            "contract": (chosen, architecture, triangle_length),
            "fault_bundle_surrogate": exposure, "screen": key}


def _p19_columns(rows, width):
    out = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            out[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return out


def _p19_contract(support, number, profile):
    """Deterministic relative-support catalogue, not stored code images."""
    from itertools import combinations

    q = 19

    def normal(values, unit=1):
        if not values:
            return ()
        values = [(unit * a) % q for a in values]
        return min(tuple(sorted((a - origin) % q for a in values))
                   for origin in values)

    def directions(values):
        return {min((a-b) % q, (b-a) % q)
                for a, b in combinations(values, 2)}

    triples = sorted({
        normal((0, a, b))
        for a, b in combinations(range(1, q), 2)
        if len(directions((0, a, b))) == 3
    })
    palettes = {
        0: ((),), 1: ((0,),),
        2: tuple((0, d) for d in range(1, 10)),
        3: tuple(triples),
    }

    def canonical(grid):
        images = []
        for unit in range(1, q):
            transformed = [
                [normal(cell, unit) for cell in row] for row in grid]
            cols = list(zip(*transformed))
            images.append(tuple(sorted(cols)))
            images.append(tuple(sorted((b, a) for a, b in cols)))
        return min(images)

    srng = random.Random(19030017 + 7919 * profile)
    seen = set()
    accepted = 0
    for proposal in range(20000):
        grid = []
        for row in support:
            used, cells = set(), []
            for count in row:
                available = [cell for cell in palettes[count]
                             if not directions(cell) & used]
                if not available:
                    break
                cell = srng.choice(available)
                cells.append(cell)
                used.update(directions(cell))
            if len(cells) != len(row):
                break
            grid.append(cells)
        if len(grid) != 2:
            continue
        identity = canonical(grid)
        if identity in seen:
            continue
        seen.add(identity)
        if accepted == number:
            return grid, identity
        accepted += 1
    raise ValueError("unused C19 relative-support catalogue exhausted")


def _p19_seven(cols, q):
    """Exact weight-seven census, assuming no kernel word through six.

    Translations are transitive within each column block. Each anchored
    seven-word contributes twenty ordered complementary triple matches.
    This counts incidences, without assuming free orbits of words.
    """
    anchored = []
    for anchor in range(0, len(cols), q):
        other = cols[:anchor] + cols[anchor+1:]
        triples = set()
        for i, x in enumerate(other):
            for j in range(i):
                pair = x ^ other[j]
                for h in range(j):
                    triples.add(pair ^ other[h])
        expected = len(other)*(len(other)-1)*(len(other)-2)//6
        assert len(triples) == expected
        hits = sum((s ^ cols[anchor]) in triples for s in triples)
        assert hits % 20 == 0
        anchored.append(hits // 20)
    incidences = q * sum(anchored)
    assert incidences % 7 == 0
    return incidences // 7, max(anchored), tuple(anchored)


def _p19_history_probe(rows, width, seed, shots=32):
    """Two-interval classical noisy-syndrome surrogate, NOT ASC memory.

    Equations are H*e1+m=s1 and H*(e1+e2)=s2. The final observation
    is perfect. Wrong settled final words and unresolved syndromes are
    separated. Actual gate order, six-round relay BP and LSD are absent.
    """
    import math

    nr = len(rows)
    variables = 2 * width + nr
    augmented = [
        row | (1 << (2*width+r)) for r, row in enumerate(rows)
    ] + [row | (row << width) for row in rows]
    columns = _p19_columns(augmented, variables)
    original = _p19_columns(rows, width)
    probabilities = [
        0.018 + 0.005 * original[c % width].bit_count()
        for c in range(2 * width)
    ] + [0.015] * nr
    edge_variables, check_edges = [], []
    for row in augmented:
        edges = []
        while row:
            bit = row & -row
            edges.append(len(edge_variables))
            edge_variables.append(bit.bit_length()-1)
            row ^= bit
        check_edges.append(edges)

    srng = random.Random(seed)
    wrong = unresolved = effort = 0
    mask = (1 << width) - 1
    for shot in range(shots):
        level = (0.65, 1.0, 1.35)[shot % 3]
        rates = [min(0.20, p*level) for p in probabilities]
        priors = [math.log((1-p)/p) for p in rates]
        error = syndrome = 0
        for c, p in enumerate(rates):
            if srng.random() < p:
                error |= 1 << c
                syndrome ^= columns[c]
        true_final = (error & mask) ^ ((error >> width) & mask)
        incoming = [0.0] * len(edge_variables)
        messages = [priors[c] for c in edge_variables]
        settled = False

        for iteration in range(24):
            for r, edges in enumerate(check_edges):
                sign = -1 if (syndrome >> r) & 1 else 1
                first = second = 60.0
                smallest = -1
                for edge in edges:
                    value = messages[edge]
                    if value < 0:
                        sign = -sign
                    magnitude = abs(value)
                    if magnitude < first:
                        first, second = magnitude, first
                        smallest = edge
                    elif magnitude < second:
                        second = magnitude
                for edge in edges:
                    magnitude = second if edge == smallest else first
                    direction = -sign if messages[edge] < 0 else sign
                    update = 0.80 * direction * magnitude
                    incoming[edge] = 0.20*incoming[edge] + 0.80*update

            beliefs = list(priors)
            for edge, c in enumerate(edge_variables):
                beliefs[c] += incoming[edge]
            estimate = observed = 0
            for c, value in enumerate(beliefs):
                if value < 0:
                    estimate |= 1 << c
                    observed ^= columns[c]
            effort += 1
            if observed == syndrome:
                final = (estimate & mask) ^ ((estimate >> width) & mask)
                wrong += int(final != true_final)
                settled = True
                break
            for edge, c in enumerate(edge_variables):
                proposed = beliefs[c] - incoming[edge]
                messages[edge] = max(-18.0, min(18.0, proposed))
        unresolved += int(not settled)
    return wrong, unresolved, effort


def _p19_logical4(checks, stabilizers, width):
    """Exact existence of a nontrivial quantum logical through four."""
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

    cols = _p19_columns(checks, width)
    seen = {0: 0}
    for i, syndrome in enumerate(cols):
        word = 1 << i
        old = seen.get(syndrome)
        if old is None:
            seen[syndrome] = word
        elif nontrivial(word ^ old):
            return True
    for i, syndrome in enumerate(cols):
        for j in range(i):
            value = syndrome ^ cols[j]
            word = (1 << i) | (1 << j)
            old = seen.get(value)
            if old is None:
                seen[value] = word
            elif nontrivial(word ^ old):
                return True
    return False


def NEW_CTX(rng, G, idx):
    """Canonical degree-2/3 contracts with independent history holdouts."""
    import math
    from itertools import combinations_with_replacement

    q = G.q
    assert q == 19 and A_SHAPE == B_SHAPE == (2, 5)
    width, block_mask = 5*q, (1 << q)-1

    # Remove support row/column copies before allocating slot contracts.
    strong = ((0, 3), (1, 2), (2, 1), (3, 0))
    support_keys = set()
    for budget in (14, 15):
        weak = ((1, 1),) if budget == 14 else ()
        for tail in combinations_with_replacement(strong, 5-len(weak)):
            cols = weak + tail
            totals = sorted(sum(col[i] for col in cols) for i in range(2))
            if totals != ([7, 7] if budget == 14 else [7, 8]):
                continue
            if sum(a > 0 and b > 0 for a, b in cols) < 3:
                continue
            parity = [sum((col[i] & 1) << j
                          for j, col in enumerate(cols)) for i in range(2)]
            if gf2_rank(parity) != 2:
                continue
            support_keys.add(min(
                tuple(sorted(cols)),
                tuple(sorted((b, a) for a, b in cols))))
    profile_keys = sorted(support_keys, key=lambda key: (
        sum(map(sum, key)), sum(a == 0 or b == 0 for a, b in key), key))
    profiles = [
        [[col[i] for col in key] for i in range(2)]
        for key in profile_keys
    ]
    pairs = list(combinations_with_replacement(range(len(profiles)), 2))
    assert len(pairs) >= 20
    batch, card = divmod(idx, len(pairs))
    pa, pb = pairs[card]
    supports = {
        "A": [list(row) for row in profiles[pa]],
        "B": [list(row) for row in profiles[pb]],
    }
    assert _no_dead_lines(supports["A"], supports["B"])
    assert _grid_weights(supports["A"], supports["B"]) <= min(11, W_MAX)

    relative, identities = {}, {}
    for side, profile, number in (
            ("A", pa, 2*batch), ("B", pb, 2*batch+1)):
        relative[side], identities[side] = _p19_contract(
            supports[side], number, profile)

    # Every cached singleton block comes from the verified construction.
    actions = {"A": {}, "B": {}}
    for g in G.elements:
        hx, _, _ = lifted_matrices(G, [[[g]]], [[[]]])
        _, hz, _ = lifted_matrices(G, [[[]]], [[[g]]])
        actions["A"][G.idx(g)] = tuple(row & block_mask for row in hx)
        actions["B"][G.idx(g)] = tuple(row & block_mask for row in hz)

    def component(grid, side):
        rows = [0] * (2*q)
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                for g in cell:
                    for t, bits in enumerate(actions[side][G.idx(g)]):
                        rows[i*q+t] ^= bits << (j*q)
        return rows

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def metric(rows):
        adjacency = [0] * len(rows)
        four = 0
        for i, row in enumerate(rows):
            for j in range(i):
                overlap = (row & rows[j]).bit_count()
                four += overlap*(overlap-1)//2
                if overlap:
                    adjacency[i] |= 1 << j
                    adjacency[j] |= 1 << i
        reached = frontier = 1
        while frontier:
            following = 0
            while frontier:
                bit = frontier & -frontier
                following |= adjacency[bit.bit_length()-1]
                frontier ^= bit
            frontier = following & ~reached
            reached |= frontier
        cols = _p19_columns(rows, width)
        duplicates = width-len(set(cols)) + int(0 in cols)
        triangles = 0
        for i, neighbors in enumerate(adjacency):
            lower = neighbors & ((1 << i)-1)
            while lower:
                bit = lower & -lower
                j = bit.bit_length()-1
                triangles += (
                    neighbors & adjacency[j] & ((1 << j)-1)).bit_count()
                lower ^= bit
        stars = sum(d*(d-1)*(d-2)//6 for d in map(int.bit_count, cols))
        six = max(0, triangles-stars)
        return len(rows)-reached.bit_count(), duplicates, four, six

    def energy(cost):
        return (5000*cost[0] + 2000*cost[1]
                + 100*cost[2] + cost[3]/4) / q

    def sampled(cols, side):
        srng = random.Random(19031037 + 1009*idx + int(side == "B"))
        order = list(range(width))
        minimum, short = width+1, set()

        def offer(word):
            nonlocal minimum
            if word:
                weight = word.bit_count()
                minimum = min(minimum, weight)
                if weight <= 10 and len(short) < 8192:
                    short.add(word)

        for trial in range(40):
            srng.shuffle(order)
            pivots, basis = {}, []
            for c in order:
                syndrome, word = cols[c], 1 << c
                while syndrome:
                    p = syndrome.bit_length()-1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = syndrome, word
                        break
                    syndrome ^= old[0]
                    word ^= old[1]
                if not syndrome:
                    basis.append(word)
                    offer(word)
            for i, word in enumerate(basis):
                for other in basis[:i]:
                    offer(word ^ other)
            for word in sorted(basis, key=int.bit_count)[:5]:
                for sweep in range(2):
                    best = word
                    for other in basis:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < best.bit_count():
                            best = candidate
                    if best == word:
                        break
                    word = best
                    offer(word)
        loads = [
            sum(bool((word >> (j*q)) & block_mask) for word in short)
            for j in range(5)]
        return min(minimum, 10), len(short), max(loads, default=0)

    def portfolio(side):
        support, model = supports[side], relative[side]
        mixed = [j for j in range(5) if support[0][j] and support[1][j]]
        # Column gauges fix the first live cell's origin. A row gauge
        # fixes one mixed lower origin; remaining voltages are searched.
        fixed = mixed[0]
        sites = [(1, j) for j in mixed if j != fixed]

        def cell(i, j, origin):
            return [((origin+a) % q,) for a in model[i][j]]

        def draw():
            return [[cell(i, j, rng.randrange(q)
                          if i == 1 and j in mixed and j != fixed else 0)
                     for j in range(5)] for i in range(2)]

        pool, seen = [], set()
        serial = 0

        def remember(grid, rows, cost):
            nonlocal serial
            if cost[0] or cost[1]:
                return
            if len(pool) >= 6 and cost > pool[-1][0]:
                return
            signature = tuple(tuple(sorted(G.idx(g) for g in cell_))
                              for row in grid for cell_ in row)
            if signature in seen or gf2_rank(rows) != 2*q:
                return
            seen.add(signature)
            serial += 1
            pool.append((cost, serial, clone(grid), list(rows)))
            pool.sort(key=lambda entry: entry[:2])
            del pool[6:]

        for restart in range(4):
            grid = draw()
            rows = component(grid, side)
            cost = metric(rows)
            remember(grid, rows, cost)
            for step in range(340):
                i, j = rng.choice(sites)
                old = grid[i][j]
                grid[i][j] = cell(i, j, rng.randrange(q))
                proposed_rows = component(grid, side)
                proposed = metric(proposed_rows)
                delta = energy(proposed)-energy(cost)
                temperature = 9.0*(0.986**step) + 0.08
                if delta <= 0 or rng.random() < math.exp(
                        -min(700.0, delta/temperature)):
                    rows, cost = proposed_rows, proposed
                    if delta < 0 or step % 17 == 0:
                        remember(grid, rows, cost)
                else:
                    grid[i][j] = old
            remember(grid, rows, cost)
        if not pool:
            for trial in range(256):
                grid = draw()
                rows = component(grid, side)
                remember(grid, rows, metric(rows))
        if not pool:
            raise ValueError("no connected full-rank C19 component")

        refined = []
        seed = 190312031 + 1009*idx + int(side == "B")
        for coarse, number, grid, rows in pool[:4]:
            cols = _p19_columns(rows, width)
            bad = int(_small_kernel_word(rows, width))
            seven = ((10**9, 10**9, ()) if bad
                     else _p19_seven(cols, q))
            distance, low, peak = sampled(cols, side)
            training = _p19_history_probe(rows, width, seed)
            key = (bad, int(seven[0] > 0), seven[0], seven[1],
                   max(0, 10-distance), low, peak,
                   training[0]+training[1], training[0],
                   coarse, number)
            refined.append({
                "key": key, "grid": grid, "rows": rows,
                "bad": bad, "seven": seven, "distance": distance,
                "low": low, "peak": peak, "training": training,
                "coarse": coarse,
            })
        refined.sort(key=lambda item: item["key"])
        for item in refined[:2]:
            item["holdout"] = _p19_history_probe(
                item["rows"], width, seed+1000003, 48)
        return refined[:2]

    apool, bpool = portfolio("A"), portfolio("B")
    options = []
    for a in apool:
        for b in bpool:
            A, B = a["grid"], b["grid"]
            hx, hz, n = lifted_matrices(G, A, B)
            k = n-gf2_rank(hx)-gf2_rank(hz)
            if n != 551 or k != 171:
                continue
            ah, bh = a["holdout"], b["holdout"]
            at, bt = a["training"], b["training"]
            key = (
                max(a["bad"], b["bad"]), a["bad"]+b["bad"],
                int(a["seven"][0] > 0 or b["seven"][0] > 0),
                max(a["seven"][0], b["seven"][0]),
                a["seven"][0]+b["seven"][0],
                max(0, 10-min(a["distance"], b["distance"])),
                max(a["low"], b["low"]), a["low"]+b["low"],
                max(ah[0]+ah[1], bh[0]+bh[1]),
                ah[0]+ah[1]+bh[0]+bh[1],
                max(ah[0], bh[0]), ah[0]+bh[0],
                max(at[0]+at[1], bt[0]+bt[1]),
                at[0]+at[1]+bt[0]+bt[1],
                max(a["peak"], b["peak"]),
                ah[2]+bh[2], a["coarse"], b["coarse"],
            )
            options.append((key, a, b, hx, hz))
    options.sort(key=lambda entry: entry[0])
    for key, a, b, hx, hz in options:
        if _p19_logical4(hx, hz, 551) or _p19_logical4(hz, hx, 551):
            continue
        A, B = a["grid"], b["grid"]
        actual_a, actual_b = _component_rows(hx, hz, q)
        assert actual_a == a["rows"] and actual_b == b["rows"]
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx+hz) <= min(11, W_MAX)
        terms = sum(map(sum, supports["A"])) + sum(map(sum, supports["B"]))
        cnots = 7*terms*q*6
        assert cnots in (22344, 23142, 23940)
        assert 6*sum(row.bit_count() for row in hx+hz) == cnots
        assert all(
            len(cell_) == supports[side][i][j] <= CELL_MAX
            and len({G._key(g) for g in cell_}) == len(cell_)
            for side, grid in (("A", A), ("B", B))
            for i, row in enumerate(grid) for j, cell_ in enumerate(row))
        assert 0.275 <= 171/551 < 0.325
        return {
            "A": A, "B": B, "k": 171, "cnots": cnots,
            "contract": (pa, pb, batch, identities),
            "seven_census": (a["seven"], b["seven"]),
            "training_surrogate": (a["training"], b["training"]),
            "holdout_surrogate": (a["holdout"], b["holdout"]),
            "screen": key,
        }
    raise ValueError("no exact-rate short-logical-safe C19 pairing")


def CELL(rng, G, side, i, j, ctx):
    """Emit the complete verified holdout-selected cell without redraws."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Weight-legal rolled supports with uniform labels reliably balance the two bases: the parent had 100+ X failures in 750..33250 shots versus only 0..12 Z failures in 200000 shots.
# REFUTED: Excluding component kernel words <=6 and minimizing component four/six cycles guarantees ASC measurability: current slot 14 passed these screens but its X schedule timed out after 120 seconds.
# REFUTED: Equal d2s predicts equal circuit reliability: current slots 1 and 11 both had d2s=15, but their worse bases had zero and 13 failures in 200000 shots.
# REFUTED: d2s=16 ensures near-ceiling memory on this order-36 bed: parent slot11 had45 Z failures/200000 and s=6.186, whereas slot58 had one failure per basis and s=7.225.
# REFUTED: Mixed-order fault-bundle selection guarantees zero-failure memory: current-parent slot3 had twelve Z failures/200000 and 371 logical flips, while slot47 with the same d2s15 had one failure and 42 flips; all five X measurements had zero failures.
# REFUTED: Full-column generation plus component capacity stress certifies zero-failure sampling: the current parent had seven Z failures in one million measured shots, flipping 227 logical qubits in total, while all five X measurements had zero failures.
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
