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
CONCEPT = ("C4 semidirect C4 with inversion action, order 16, shapes (3,5)x(3,5): generating even-column A supports and predominantly monomial B supports should improve worse-basis circuit fitness beyond 7.3634 in cell 0.150.")
GROUP = ("Preserve the parent's semidirect inversion construction and order-four acting factor. Size factor 34 makes order 36 illegal; m=4,r=4 gives the largest admissible order with this acting factor, q=16 and n=544. This order-16 bed reached 7.2524, including zero-failure measurements, motivating a new support and relative-geometry experiment without claiming a demonstrated group advantage.")
SHAPES = ("Equal (3,5) components each have 80 variables and 48 checks, hence dimension at least 32. At q=16, n=544 and kb=64 give guaranteed rate 2/17. Complete verified ranks must give 69<=k<=95, placing every emitted code in cell 0.150. Parent slot32 had two Z failures/200000, 54 collective logical flips, and enrichment in Acol0/Bcol2 and several Acol/Brow checks; this sparse map motivates distributed protection rather than a unique hotspot repair.")
FILLING = ("Three B support architectures cross indexed canonical generating A-ratio geometries, giving twenty deliberate initial contracts and fresh contracts for larger harness budgets. Initial A columns contain two order-four binomials and one zero cell, with column degree four and row totals six/six/eight. Each column's ratios generate all 16 elements, verified by isolated-column rank15. B is monomial except for one binomial, two in one row, or two in distinct rows and columns; every B column contains an invertible singleton. Both CSS row-weight sums are <=12 and initial six-round CNOTs are 27648/28416, versus the parent's measured 30672. A finite-search backup replaces one A column by three singletons, changing augmentation rather than repairing emitted cells. Anneal helper-derived actions while preserving ratios and retaining component rank diversity. Prefer connectivity, exact component exclusion through six, balanced sampled protection capped at fourteen, common-stream capacity min-sum stress, short-witness concentration and boundaries. Full ranks and exact quantum-logical exclusion through four select the target band; prefer k>=84 only after protection and decoder-stress priorities. These screens do not certify ASC scheduling or circuit pL.")
FAMILY_NAME = "r015_C4sdC4_35x35_evenA_monomialB_decoderstress_v3"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 16

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C4 semidirect C4 requires order 16")
    # Same inversion-semidir­ect construction, resized for factor 34.
    # x^4=y^4=1; the parity action is a homomorphism C4 -> Aut(C4).
    m, r = 4, 4
    els = [(a, b) for b in range(r) for a in range(m)]

    def red(g):
        return (g[0] % m, g[1] % r)

    def mul(g, h):
        sign = -1 if g[1] % 2 else 1
        return ((g[0] + sign * h[0]) % m,
                (g[1] + h[1]) % r)

    def inv(g):
        sign = -1 if g[1] % 2 else 1
        return ((-sign * g[0]) % m, (-g[1]) % r)

    def idx(g):
        a, b = red(g)
        return b * m + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (3, 5)               # (ra, ca)
B_SHAPE = (3, 5)               # (rb, cb)


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


def _shape_logical4(checks, stabilizers, width):
    """Exact existence of a nontrivial quantum logical through weight four."""
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

    cols = [0] * width
    for r, row in enumerate(checks):
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    seen = {0: 0}
    for i, syndrome in enumerate(cols):
        word = 1 << i
        previous = seen.get(syndrome)
        if previous is None:
            seen[syndrome] = word
        elif nontrivial(word ^ previous):
            return True
    for i, syndrome in enumerate(cols):
        for j in range(i):
            value = syndrome ^ cols[j]
            word = (1 << i) | (1 << j)
            previous = seen.get(value)
            if previous is None:
                seen[value] = word
            elif nontrivial(word ^ previous):
                return True
    return False


def NEW_CTX(rng, G, idx):
    """Generating even-column contracts with rank-diverse fresh origins."""
    import math
    from itertools import permutations

    q = G.q
    assert q == 16 and A_SHAPE == B_SHAPE == (3, 5)
    architecture, geometry_number = idx % 3, idx // 3
    even_a = [[0, 0, 2, 2, 2],
              [2, 2, 0, 0, 2],
              [2, 2, 2, 2, 0]]
    backup_a = [list(row) for row in even_a]
    for i in range(3):
        backup_a[i][0] = 1

    sb = [[1] * 5 for _ in range(3)]
    reinforced = (
        ((0, 0),),
        ((0, 0), (0, 1)),
        ((0, 0), (1, 1)),
    )[architecture]
    for i, j in reinforced:
        sb[i][j] = 2
    sa = [list(row) for row in even_a]
    counts = {"A": sa, "B": sb}
    for profile in (even_a, backup_a):
        wx = max(map(sum, profile)) + max(map(sum, zip(*sb)))
        wz = max(map(sum, sb)) + max(map(sum, zip(*profile)))
        assert wx <= 12 and wz <= 12
        assert _grid_weights(profile, sb) == max(wx, wz) <= W_MAX
        assert _no_dead_lines(profile, sb)
    assert all(0 <= w <= min(2, CELL_MAX)
               for grid in counts.values() for row in grid for w in row)

    multiplication = [
        [G.idx(G.mul(g, h)) for h in G.elements]
        for g in G.elements
    ]
    inverse = [G.idx(G.inv(g)) for g in G.elements]
    identity = G.idx(G.e)
    orders = []
    powers = []
    for g in range(q):
        power, sequence = identity, [identity]
        for order in range(1, q + 1):
            power = multiplication[power][g]
            sequence.append(power)
            if power == identity:
                orders.append(order)
                break
        else:
            raise ValueError("Element order missing")
        powers.append(sequence)
    palette = [g for g in range(q) if orders[g] == 4]
    assert palette
    generation_cache = {}

    def generated_size(a, b):
        key = tuple(sorted((min(a, inverse[a]), min(b, inverse[b]))))
        if key not in generation_cache:
            reached, todo = {identity}, [identity]
            steps = (a, inverse[a], b, inverse[b])
            while todo:
                current = todo.pop()
                for step in steps:
                    image = multiplication[current][step]
                    if image not in reached:
                        reached.add(image)
                        todo.append(image)
            generation_cache[key] = len(reached)
        return generation_cache[key]

    generating_pairs = [
        (a, b) for a in palette for b in palette
        if generated_size(a, b) == q
    ]
    assert generating_pairs

    # Enumerate actual automorphisms from images of the presentation
    # generators. This canonicalizes contracts, never selected codes.
    automorphisms = set()
    for x in palette:
        for y in palette:
            if generated_size(x, y) != q:
                continue
            conjugate = multiplication[
                multiplication[y][x]][inverse[y]]
            if conjugate != inverse[x]:
                continue
            mapping = tuple(
                multiplication[powers[x][a]][powers[y][b]]
                for a, b in G.elements
            )
            if len(set(mapping)) != q:
                continue
            if all(mapping[multiplication[a][b]]
                   == multiplication[mapping[a]][mapping[b]]
                   for a in range(q) for b in range(q)):
                automorphisms.add(mapping)
    assert automorphisms
    row_permutations = tuple(permutations(range(3)))

    def geometry_key(grid):
        images = []
        for mapping in automorphisms:
            transformed = [
                [(-1 if g < 0 else
                  min(mapping[g], mapping[inverse[g]]))
                 for g in row] for row in grid
            ]
            for p in row_permutations:
                images.append(tuple(sorted(
                    tuple(transformed[i][j] for i in p)
                    for j in range(5)
                )))
        return min(images)

    catalogue_rng = random.Random(160350731 + 1009 * architecture)
    wanted = 3 * geometry_number
    book, seen = [], set()
    for proposal in range(30000):
        grid = [[-1] * 5 for _ in range(3)]
        for j in range(5):
            active = [i for i in range(3) if even_a[i][j]]
            a, b = catalogue_rng.choice(generating_pairs)
            grid[active[0]][j], grid[active[1]][j] = a, b
        key = geometry_key(grid)
        if key in seen:
            continue
        seen.add(key)
        book.append((key, grid))
        if len(book) >= wanted + 3:
            break
    if len(book) < wanted + 3:
        raise ValueError("Unused generating even-column book exhausted")
    slot_geometries = book[wanted:wanted + 3]
    contract_key, delta_grid = slot_geometries[0]

    def draw_cell(side, i, j):
        weight = counts[side][i][j]
        if not weight:
            return []
        origin = G.pick(rng)
        if weight == 1:
            return [origin]
        delta = (delta_grid[i][j] if side == "A"
                 else rng.choice(palette))
        assert delta >= 0 and orders[delta] == 4
        cell = [origin, G.mul(origin, G.elements[delta])]
        assert len({G._key(g) for g in cell}) == 2
        return cell

    def draw_grid(side):
        return [[draw_cell(side, i, j) for j in range(5)]
                for i in range(3)]

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    # Cache only singleton actions returned by the verified helper.
    # Final full-matrix projections independently verify this assembly.
    actions = {"A": {}, "B": {}}
    block_mask = (1 << q) - 1
    for g in G.elements:
        hx, _, _ = lifted_matrices(G, [[[g]]], [[[]]])
        _, hz, _ = lifted_matrices(G, [[[]]], [[[g]]])
        actions["A"][G.idx(g)] = tuple(row & block_mask for row in hx)
        actions["B"][G.idx(g)] = tuple(row & block_mask for row in hz)

    def component(side, grid):
        rows = []
        for grid_row in grid:
            out = [0] * q
            for j, cell in enumerate(grid_row):
                for g in cell:
                    for t, bits in enumerate(actions[side][G.idx(g)]):
                        out[t] ^= bits << (j * q)
            rows.extend(out)
        return rows, len(grid[0]) * q

    def columns(rows, width):
        result = [0] * width
        for r, row in enumerate(rows):
            marker = 1 << r
            while row:
                bit = row & -row
                result[bit.bit_length() - 1] |= marker
                row ^= bit
        return result

    def connected(rows, width):
        parent = list(range(width))

        def root(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for row in rows:
            if not row:
                continue
            first = (row & -row).bit_length() - 1
            while row:
                bit = row & -row
                a, b = root(first), root(bit.bit_length() - 1)
                if a != b:
                    parent[b] = a
                row ^= bit
        return len({root(t) for t in range(width)}) == 1

    def cycle_energy(rows, width):
        four = 0
        adjacency = [0] * len(rows)
        for i, row in enumerate(rows):
            for j in range(i):
                overlap = (row & rows[j]).bit_count()
                four += overlap * (overlap - 1) // 2
                if overlap:
                    adjacency[i] |= 1 << j
                    adjacency[j] |= 1 << i
        triangles = 0
        for i in range(len(rows)):
            lower = adjacency[i] & ((1 << i) - 1)
            while lower:
                bit = lower & -lower
                j = bit.bit_length() - 1
                triangles += (adjacency[i] & adjacency[j]
                              & ((1 << j) - 1)).bit_count()
                lower ^= bit
        stars = sum(d * (d - 1) * (d - 2) // 6
                    for d in (c.bit_count() for c in columns(rows, width)))
        # Exact six-cycles if C4=0; otherwise only a proposal heuristic.
        six = max(0, triangles - stars)
        return (80.0 * four + six / 8.0) / q, four, six

    def boundary_exposure(cols):
        """Exact small-boundary triple statistic, not a trapping certificate."""
        total = 0
        loads = [0] * (len(cols) // q)
        for i in range(len(cols)):
            for j in range(i):
                pair = cols[i] ^ cols[j]
                for h in range(j):
                    weight = (pair ^ cols[h]).bit_count()
                    if weight <= 4:
                        cost = (6 - weight) ** 2
                        total += cost
                        loads[i // q] += cost
                        loads[j // q] += cost
                        loads[h // q] += cost
        return total / len(cols), max(loads) / q

    def sampled_words(cols, side):
        """Verified fundamental/pair-XOR words; minimum is an upper bound.

        Use the same permutation stream for finalists in this slot.
        Multiplicity and block exposure refer only to sampled witnesses.
        """
        srng = random.Random(73129 + 1009 * idx + (side == "B"))
        width = len(cols)
        permutation = list(range(width))
        minimum = width + 1
        short = set()
        threshold = 14
        for trial in range(48):
            srng.shuffle(permutation)
            pivots = {}
            basis = []
            for c in permutation:
                syndrome, word = cols[c], 1 << c
                while syndrome:
                    p = syndrome.bit_length() - 1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = (syndrome, word)
                        break
                    syndrome ^= old[0]
                    word ^= old[1]
                if not syndrome:
                    basis.append(word)

            for i, word in enumerate(basis):
                candidates = [word]
                candidates.extend(word ^ other for other in basis[:i])
                for v in candidates:
                    if not v:
                        continue
                    weight = v.bit_count()
                    minimum = min(minimum, weight)
                    if weight <= threshold and len(short) < 8192:
                        short.add(v)
            # All descent moves add genuine kernel vectors.
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
                    weight = word.bit_count()
                    minimum = min(minimum, weight)
                    if weight <= threshold and len(short) < 8192:
                        short.add(word)
            if minimum <= 6:
                break

        mask = (1 << q) - 1
        loads = [0] * (width // q)
        very_short = 0
        for word in short:
            very_short += int(word.bit_count() <= 12)
            for j in range(len(loads)):
                loads[j] += int(bool((word >> (j * q)) & mask))
        return minimum, very_short, len(short), max(loads)

    def capacity_stress(rows, cols, side):
        """Common-stream component min-sum stress, not circuit pL.

        Perfect syndromes isolate iterative decoding difficulty.
        The evaluator uses a different decoder and noisy ASC extraction.
        """
        edge_columns, check_edges = [], []
        for row in rows:
            edges = []
            while row:
                bit = row & -row
                edges.append(len(edge_columns))
                edge_columns.append(bit.bit_length() - 1)
                row ^= bit
            check_edges.append(edges)

        width = len(cols)
        p = 0.040 if side == "A" else 0.030
        prior = math.log((1 - p) / p)
        srng = random.Random(160350913 + 1009 * idx + int(side == "B"))
        wrong = unresolved = effort = 0
        for trial in range(32):
            if trial < 16:
                sites = [c for c in range(width) if srng.random() < p]
            else:
                sites = srng.sample(range(width), 3 + trial % 5)
            error = syndrome = 0
            for c in sites:
                error |= 1 << c
                syndrome ^= cols[c]
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
                            0.15 * incoming[edge] + 0.85 * proposed)

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
                    messages[edge] = max(
                        -12.0, min(12.0, beliefs[c] - incoming[edge]))
            unresolved += int(not settled)
        return wrong, unresolved, effort

    def portfolio(side):
        supports = counts[side]
        positions = [(i, j) for i, row in enumerate(supports)
                     for j, w in enumerate(row) if w]
        strata = {}
        seen = set()
        serial = 0

        def remember(grid, rows, stats):
            nonlocal serial
            coarse = (stats[1], stats[2])
            deficit = len(rows) - gf2_rank(rows)
            bucket = strata.setdefault(deficit, [])
            if len(bucket) >= 2 and coarse > bucket[-1][0]:
                return
            signature = tuple(
                tuple(sorted(G.idx(g) for g in cell))
                for row in grid for cell in row)
            if signature in seen:
                return
            seen.add(signature)
            serial += 1
            bucket.append((coarse, serial, clone(grid), list(rows)))
            bucket.sort(key=lambda item: item[:2])
            del bucket[2:]

        for restart in range(3):
            grid = draw_grid(side)
            rows, width = component(side, grid)
            current = cycle_energy(rows, width)
            remember(grid, rows, current)
            for step in range(260):
                i, j = rng.choice(positions)
                old = grid[i][j]
                grid[i][j] = draw_cell(side, i, j)
                trial_rows, _ = component(side, grid)
                proposed = cycle_energy(trial_rows, width)
                delta = proposed[0] - current[0]
                temperature = 10.0 * (0.985 ** step) + 0.08
                if (delta <= 0 or rng.random() <
                        math.exp(-min(700.0, delta / temperature))):
                    rows, current = trial_rows, proposed
                    remember(grid, rows, current)
                else:
                    grid[i][j] = old

        # Preserve rank strata before refining: component deficits alone
        # do not determine complete nonabelian CSS k.
        pool = [bucket[0] for bucket in strata.values()]
        pool.sort(key=lambda item: item[:2])
        pool = pool[:6]
        extras = [entry for bucket in strata.values() for entry in bucket[1:]]
        extras.sort(key=lambda item: item[:2])
        pool.extend(extras[:max(0, 8 - len(pool))])

        finalists = []
        for coarse, serial, grid, rows in pool:
            width = len(grid[0]) * q
            cols = columns(rows, width)
            bad = int(_small_kernel_word(rows, width))
            disconnected = int(not connected(rows, width))
            d, low12, low, load = sampled_words(cols, side)
            risk, peak = boundary_exposure(cols)
            stress = capacity_stress(rows, cols, side)
            four = coarse[0]
            six = _six_cycles(rows)
            key = (disconnected, bad, max(0, 14 - d),
                   stress[0] + stress[1], stress[0],
                   low12, low, load, peak, risk, four, six, stress[2])
            finalists.append({
                "grid": grid, "rows": rows, "key": key, "bad": bad,
                "disconnected": disconnected, "d": d,
                "low12": low12, "low": low, "load": load,
                "risk": risk, "peak": peak, "stress": stress,
                "four": four, "six": six,
                "deficit": len(rows) - gf2_rank(rows),
            })
        finalists.sort(key=lambda item: item["key"])
        return finalists

    def choose_pair(apool, bpool):
        options = []
        for a in apool:
            for b in bpool:
                A, B = a["grid"], b["grid"]
                hx, hz, n = lifted_matrices(G, A, B)
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if n != 544 or not 69 <= k <= 95:
                    continue
                ga, gb = max(0, 14 - a["d"]), max(0, 14 - b["d"])
                xa, xb = a["stress"], b["stress"]
                key = (
                    a["disconnected"] + b["disconnected"],
                    max(a["bad"], b["bad"]), a["bad"] + b["bad"],
                    max(ga, gb), ga + gb,
                    max(xa[0] + xa[1], xb[0] + xb[1]),
                    xa[0] + xa[1] + xb[0] + xb[1],
                    max(xa[0], xb[0]), xa[0] + xb[0],
                    max(a["low12"], b["low12"]),
                    a["low12"] + b["low12"],
                    max(a["load"], b["load"]),
                    a["low"] + b["low"],
                    max(a["peak"], b["peak"]),
                    a["risk"] + b["risk"],
                    max(0, 84 - k), -k,
                    a["four"] + b["four"],
                    a["six"] + b["six"], xa[2] + xb[2],
                )
                options.append((key, A, B, hx, hz, n, k))
        options.sort(key=lambda item: item[0])
        for item in options:
            _, _, _, hx, hz, n, _ = item
            if (_shape_logical4(hx, hz, n)
                    or _shape_logical4(hz, hx, n)):
                continue
            return item
        return None

    winner = None
    selected_attempt = None
    for attempt in range(3):
        contract_key, delta_grid = slot_geometries[attempt]
        profile = backup_a if attempt == 1 else even_a
        counts["A"] = [list(row) for row in profile]
        assert _grid_weights(counts["A"], sb) <= min(12, W_MAX)
        apool, bpool = portfolio("A"), portfolio("B")
        winner = choose_pair(apool, bpool)
        if winner is not None:
            selected_attempt = attempt
            break

    # A fresh finite-search backup changes augmentation before building
    # candidates. It never repairs emitted cells or borrows a saved code.
    if winner is None:
        counts["A"] = [list(row) for row in backup_a]
        selected_attempt = 3
        for trial in range(256):
            A, B = draw_grid("A"), draw_grid("B")
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if n != 544 or not 69 <= k <= 95:
                continue
            ar, br = _component_rows(hx, hz, q)
            bad_a = int(_small_kernel_word(ar, 5 * q))
            bad_b = int(_small_kernel_word(br, 5 * q))
            key = (
                int(not connected(ar, 5 * q))
                + int(not connected(br, 5 * q)),
                max(bad_a, bad_b), bad_a + bad_b,
                _four_cycles(ar) + _four_cycles(br),
                max(0, 84 - k), -k, trial,
            )
            if winner is None or key < winner[0]:
                if (_shape_logical4(hx, hz, n)
                        or _shape_logical4(hz, hx, n)):
                    continue
                winner = (key, A, B, hx, hz, n, k)
        if winner is None:
            raise ValueError("Generating three-row contracts missed cell 0.150")

    key, A, B, hx, hz, n, k = winner
    sa = counts["A"]
    assert n == 544 and css_ok(hx, hz)
    assert 69 <= k <= 95 and 0.125 < k / n < 0.175
    assert max(row.bit_count() for row in hx + hz) <= min(12, W_MAX)
    assert _no_dead_lines(sa, sb)
    actual_a, actual_b = _component_rows(hx, hz, q)
    assert component("A", A)[0] == actual_a
    assert component("B", B)[0] == actual_b

    # Independently verify the actual isolated-column kernels.
    # Generating even columns have only the constant weight-16 word.
    # Singleton-containing columns are injective.
    for j in range(5):
        isolated_a = [
            (row >> (j * q)) & block_mask for row in actual_a]
        expected_a = q if all(sa[i][j] == 1 for i in range(3)) else q - 1
        assert gf2_rank(isolated_a) == expected_a
        if expected_a == q - 1:
            assert all(row.bit_count() % 2 == 0 for row in isolated_a)
        isolated_b = [
            (row >> (j * q)) & block_mask for row in actual_b]
        assert gf2_rank(isolated_b) == q

    assert all(
        len(cell) == counts[side][i][j]
        and len({G._key(g) for g in cell}) == len(cell)
        for side, grid in (("A", A), ("B", B))
        for i, row in enumerate(grid) for j, cell in enumerate(row))
    for i, row in enumerate(A):
        for j, cell in enumerate(row):
            if len(cell) == 2:
                ratio = G.idx(G.mul(G.inv(cell[0]), cell[1]))
                assert ratio == delta_grid[i][j]
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    expected_cnots = (
        (cb + rb) * sum(map(sum, sa))
        + (ca + ra) * sum(map(sum, sb))
    ) * q * 6
    assert expected_cnots in (26880, 27648, 28416)
    assert sum(row.bit_count() for row in hx + hz) * 6 == expected_cnots
    return {"A": A, "B": B, "k": k, "screen": key,
            "design": (architecture, geometry_number, selected_attempt),
            "contract": contract_key, "cnots": expected_cnots}


def CELL(rng, G, side, i, j, ctx):
    """Return a copy of the already screened full cell."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Weight-legal rolled supports with uniform labels reliably balance the two bases: the parent had 100+ X failures in 750..33250 shots versus only 0..12 Z failures in 200000 shots.
# REFUTED: Excluding component kernel words <=6 and minimizing component four/six cycles guarantees ASC measurability: current slot 14 passed these screens but its X schedule timed out after 120 seconds.
# REFUTED: Equal d2s predicts equal circuit reliability: current slots 1 and 11 both had d2s=15, but their worse bases had zero and 13 failures in 200000 shots.
# REFUTED: d2s=16 ensures near-ceiling memory in this order-36 bed: parent slot 11 had 45 Z failures/200000 and s=6.186, whereas slot 58 had one failure per basis and s=7.225.
# REFUTED: Weight-eleven extraction plus component order/boundary screens guarantees zero-failure sampling: parent slot32 failed two Z shots/200000 with 54 logical flips, while slot35 at the same k=82 had no failures.
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
