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
CONCEPT = ("For A4 x C3, order 36, shapes (2,3)x(2,4), full-generation A columns paired with lighter mixed-parity B columns should preserve rare circuit failures while raising the logical-count ceiling and fitness above 7.3187 in cell 0.150.")
GROUP = ("Keep the exact A4 x C3 construction, order 36 and encoding under L2. The parent's five measured codes had zero failures in both bases and fitness 7.3187. This supports retaining the group and A-column experiment, not claiming a causal group advantage. The same group's larger (2,3)x(2,4) bed previously reached 7.2247 with heavier fillings.")
SHAPES = ("A=(2,3), B=(2,4): q=36 gives n=576 and kb=72, guaranteed rate 1/8. The size factor sixteen allows q<=37, so no order adjustment is needed. Components have 72 checks on 108 and 144 variables, with dimensions at least 36 and 72. Require verified full ranks to give 72<=k<=100, inside cell 0.150; after protection screening prefer k>=84. The current parent has no failed-shot hotspot to repair. Its slots22/54 nevertheless retain sector-two weight-twelve witnesses, so full-generation A columns are not treated as protection certificates for both sectors.")
FILLING = ("A remains regular doubleton with complementary order-three/six ratios generating all 36 elements in every column. B has four degree-three columns, or one additional term reinforcing a column to degree four; every B column includes a monomial cell. Twenty initial contracts cross two A order architectures, two B support architectures and five exact counts of order-three B doubleton ratios. Extra slots own further canonical geometries, never relabel saved fillings. A row/column totals are six/four; B row totals are six or seven and column totals three or four. HX weights are nine/ten and HZ weights ten/eleven. Six-round CNOTs are 28512 or 29592: higher totals than the smaller parent's 25920, but lower CNOTs per data qubit. Anneal origins using helper-derived actions; retain connectivity, exact component short-kernel screening and complete quantum-logical exclusion through four. Cap sampled protection rewards at sixteen for A and twelve for B, then prioritize verified k and sampled low-weight exposure. The objective is to preserve the parent's zero-failure measurements while increasing its sample ceiling; these structural screens do not certify circuit pL.")
FAMILY_NAME = "r015_A4xC3_23x24_fullA_mixedparity_gatelean_v4"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 36

def custom_ops(q):
    from itertools import permutations
    if not valid_q(q):
        raise ValueError("A4 x C3 requires order 36")
    perms = [p for p in permutations(range(4))
             if sum(p[i] > p[j] for i in range(4)
                    for j in range(i + 1, 4)) % 2 == 0]
    els = [p + (c,) for p in perms for c in range(3)]
    pos = {g: i for i, g in enumerate(els)}

    def red(g):
        return tuple(int(g[i]) for i in range(4)) + (int(g[4]) % 3,)

    def mul(g, h):
        return tuple(g[h[i]] for i in range(4)) + ((g[4] + h[4]) % 3,)

    def inv(g):
        p = [0] * 4
        for i in range(4):
            p[g[i]] = i
        return tuple(p) + ((-g[4]) % 3,)

    def idx(g):
        return pos[red(g)]

    return dict(elements=els, ident=(0, 1, 2, 3, 0),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb)


def _orders(G):
    result = []
    for g in G.elements:
        h = G.e
        for t in range(1, G.q + 1):
            h = G.mul(h, g)
            if G._key(h) == G._key(G.e):
                result.append(t)
                break
        else:
            raise ValueError("element order missing")
    return result


def _designed_cell(rng, G, count, target_order, mode, orders):
    """Avoid order-two differences; prescribe binomial cycle lengths."""
    for _ in range(512):
        cell = rng.sample(G.elements, count)
        # Distinct projected permutations prevent pure-central pairs.
        if len({tuple(g[:4]) for g in cell}) != count:
            continue
        relative = [
            orders[G.idx(G.mul(G.inv(cell[a]), cell[b]))]
            for a in range(count) for b in range(a + 1, count)
        ]
        if min(relative) < 3:
            continue
        if count == 2:
            if relative[0] != target_order:
                continue
        else:
            central_span = len({g[4] for g in cell})
            wanted = 3 if mode == 0 else 2
            if central_span != wanted:
                continue
        return cell
    raise ValueError("could not realize cell design")


def _same_basis_cycles(rows):
    """Number of Tanner four-cycles within one check basis."""
    total = 0
    for i, x in enumerate(rows):
        for j in range(i):
            overlap = (x & rows[j]).bit_count()
            total += overlap * (overlap - 1) // 2
    return total


def NEW_CTX(rng, G, idx):
    """Unequal-component contracts with full-generation A columns."""
    import math
    from itertools import combinations, permutations

    q, e = G.q, G.e
    assert q == 36
    assert A_SHAPE == (2, 3) and B_SHAPE == (2, 4)
    design, batch = idx % 20, idx // 20
    architecture, ratio_policy = design % 4, design // 4
    geometry_number = batch
    sa = [[2, 2, 2], [2, 2, 2]]
    b_profiles = (
        ((1, 1, 2, 2), (2, 2, 1, 1)),
        ((1, 1, 2, 2), (2, 2, 2, 1)),
    )
    sb = [list(row) for row in b_profiles[architecture % 2]]
    counts = (sa, sb)
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    expected_n = (ca * cb + ra * rb) * q
    wx = max(map(sum, sa)) + max(map(sum, zip(*sb)))
    wz = max(map(sum, sb)) + max(map(sum, zip(*sa)))
    expected_cnots = (
        (cb + rb) * sum(map(sum, sa))
        + (ca + ra) * sum(map(sum, sb))
    ) * q * 6
    assert expected_n == 576 and N_MIN <= expected_n < N_MAX
    assert wx <= 10 and wz <= 11
    assert _grid_weights(sa, sb) == max(wx, wz) <= min(11, W_MAX)
    assert expected_cnots in (28512, 29592)
    assert _no_dead_lines(sa, sb)
    assert all(1 <= w <= min(2, CELL_MAX)
               for grid in counts for row in grid for w in row)

    # Complementary A ratios preserve the parent's column-generation
    # experiment. B's monomial cells make every individual column
    # injective without requiring two binomial generators.
    masks = (
        ((3, 3, 3), (6, 6, 6)),
        ((3, 3, 6), (6, 6, 3)),
    )
    a_orders = masks[architecture // 2]
    b_orders = [[1 if w == 1 else 6 for w in row] for row in sb]
    double_sites = [(i, j) for i, row in enumerate(sb)
                    for j, w in enumerate(row) if w == 2]
    assert len(double_sites) >= 4
    for i, j in double_sites[:ratio_policy]:
        b_orders[i][j] = 3
    b_orders = tuple(tuple(row) for row in b_orders)
    assert sum(t == 3 for row in b_orders for t in row) == ratio_policy
    order_targets = (a_orders, b_orders)

    orders = _orders(G)
    six = [G.idx(g) for g in G.elements if orders[G.idx(g)] == 6]
    three = [G.idx(g) for g in G.elements
             if orders[G.idx(g)] == 3 and tuple(g[:4]) != tuple(e[:4])]
    assert six and three
    palettes = {3: three, 6: six}
    inverse = [G.idx(G.inv(g)) for g in G.elements]
    multiplication = [
        [G.idx(G.mul(g, h)) for h in G.elements]
        for g in G.elements
    ]
    identity_index = G.idx(e)
    generation_cache = {}

    def generation_size(a, b):
        key = tuple(sorted((min(a, inverse[a]), min(b, inverse[b]))))
        if key not in generation_cache:
            generators = (a, inverse[a], b, inverse[b])
            reached, todo = {identity_index}, [identity_index]
            while todo:
                current = todo.pop()
                for generator in generators:
                    image = multiplication[current][generator]
                    if image not in reached:
                        reached.add(image)
                        todo.append(image)
            generation_cache[key] = len(reached)
        return generation_cache[key]

    # Automorphisms: S4 conjugation on A4, C3 inversion, and the
    # three homomorphism twists from A4/V4 into the central C3.
    # This is geometry canonicalization, not a replacement lift.
    def compose(p, h):
        return tuple(p[h[i]] for i in range(4))

    klein = (
        (0, 1, 2, 3), (1, 0, 3, 2),
        (2, 3, 0, 1), (3, 2, 1, 0),
    )
    charge = {}
    power = (0, 1, 2, 3)
    cycle = (1, 2, 0, 3)
    for value in range(3):
        for v in klein:
            charge[compose(v, power)] = value
        power = compose(power, cycle)
    assert len(charge) == 12

    automorphisms = []
    for p in permutations(range(4)):
        pinv = tuple(p.index(i) for i in range(4))
        for unit in (1, 2):
            for twist in range(3):
                image = []
                for g in G.elements:
                    projection = tuple(g[:4])
                    conjugate = compose(compose(p, projection), pinv)
                    central = (unit * g[4] + twist * charge[projection]) % 3
                    image.append(G.idx(conjugate + (central,)))
                automorphisms.append(tuple(image))
    assert len(set(automorphisms)) == 144

    def geometry_key(grids):
        """Remove obvious joint relative-geometry orbits."""
        images = []
        for mapping in automorphisms:
            sides = []
            for grid in grids:
                columns_ = [
                    tuple(min(mapping[grid[i][j]],
                              mapping[inverse[grid[i][j]]])
                          for i in range(2))
                    for j in range(len(grid[0]))
                ]
                direct = tuple(sorted(columns_))
                swapped = tuple(sorted((b, a) for a, b in columns_))
                sides.append(min(direct, swapped))
            images.append(tuple(sorted(sides)))
        return min(images)

    grng = random.Random(
        36042491 + 1009 * architecture + 65537 * ratio_policy)

    def relative_grid(side):
        grid = []
        for i, support_row in enumerate(counts[side]):
            used, row = set(), []
            for j, count in enumerate(support_row):
                if count == 1:
                    row.append(identity_index)
                    continue
                available = [
                    d for d in palettes[order_targets[side][i][j]]
                    if min(d, inverse[d]) not in used
                ]
                if not available:
                    return None
                delta = grng.choice(available)
                used.add(min(delta, inverse[delta]))
                row.append(delta)
            grid.append(row)
        if side == 0 and any(
                generation_size(grid[0][j], grid[1][j]) != q
                for j in range(ca)):
            return None
        return grid

    # A slot owns three fresh canonical geometries for finite-search
    # retries. Their disjoint index ranges prevent retries from borrowing
    # another slot's relative contract. No saved filling is transformed.
    wanted = 3 * geometry_number
    book, seen_geometry = [], set()
    for proposal in range(30000):
        grids = [relative_grid(0), relative_grid(1)]
        if any(grid is None for grid in grids):
            continue
        key = geometry_key(grids)
        if key in seen_geometry:
            continue
        seen_geometry.add(key)
        book.append(grids)
        if len(book) >= wanted + 3:
            break
    if len(book) < wanted + 3:
        raise ValueError("Canonical full-column geometry book exhausted")
    slot_geometries = book[wanted:wanted + 3]
    delta_geometry = slot_geometries[0]

    def spectral_error(grid, side):
        return sum(
            orders[G.idx(G.mul(G.inv(cell[0]), cell[1]))]
            != order_targets[side][i][j]
            for i, row in enumerate(grid) for j, cell in enumerate(row)
            if counts[side][i][j] == 2)

    def draw_cell(side, i, j):
        count = counts[side][i][j]
        base = G.pick(rng)
        if count == 1:
            return [base]
        assert count == 2
        delta = G.elements[delta_geometry[side][i][j]]
        cell = [base, G.mul(base, delta)]
        assert len({G._key(g) for g in cell}) == count
        return cell

    def verify_contract(A, B):
        for side, grid in enumerate((A, B)):
            assert [[len(cell) for cell in row] for row in grid] == counts[side]
            assert spectral_error(grid, side) == 0
            ratios = [
                [G.idx(G.mul(G.inv(cell[0]), cell[1]))
                 if len(cell) == 2 else identity_index for cell in row]
                for row in grid
            ]
            if side == 0:
                assert all(
                    generation_size(ratios[0][j], ratios[1][j]) == q
                    for j in range(ca))
            else:
                assert all(any(len(grid[i][j]) == 1 for i in range(rb))
                           for j in range(cb) if architecture % 2 == 0
                           or j != 2)
            assert ratios == delta_geometry[side]

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def signature(grid):
        return tuple(tuple(sorted(G.idx(g) for g in cell))
                     for row in grid for cell in row)

    def columns(rows, width):
        out = [0] * width
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= tag
                row ^= bit
        return out

    # These cached blocks are extracted from the verified helper.
    # No separately implemented group action is used for screening.
    mask = (1 << q) - 1
    actions = [{}, {}]
    for g in G.elements:
        hx, _, _ = lifted_matrices(G, [[[g]]], [[[]]])
        _, hz, _ = lifted_matrices(G, [[[]]], [[[g]]])
        actions[0][G.idx(g)] = tuple(row & mask for row in hx)
        actions[1][G.idx(g)] = tuple(row & mask for row in hz)

    def component(grid, side):
        rows = []
        for grid_row in grid:
            out = [0] * q
            for j, cell in enumerate(grid_row):
                for g in cell:
                    block = actions[side][G.idx(g)]
                    for t, bits in enumerate(block):
                        out[t] ^= bits << (j * q)
            rows.extend(out)
        return rows

    def geometry(rows, width):
        """Exact Tanner four/six cycles and check-graph connectivity."""
        cols = columns(rows, width)
        adjacency = [dict() for _ in rows]
        neighborhoods = []
        for col in cols:
            ns = []
            while col:
                bit = col & -col
                ns.append(bit.bit_length() - 1)
                col ^= bit
            neighborhoods.append(ns)
            for a, b in combinations(ns, 2):
                adjacency[a][b] = adjacency[a].get(b, 0) + 1
                adjacency[b][a] = adjacency[b].get(a, 0) + 1
        reached, todo = {0}, [0]
        while todo:
            a = todo.pop()
            for b in adjacency[a]:
                if b not in reached:
                    reached.add(b)
                    todo.append(b)
        four = sum(v * (v - 1) // 2
                   for a, neighbors in enumerate(adjacency)
                   for b, v in neighbors.items() if a < b)
        six_cycles = 0
        for a, neighbors in enumerate(adjacency):
            for b, ab in neighbors.items():
                if b <= a:
                    continue
                for c, bc in adjacency[b].items():
                    if c > b and c in neighbors:
                        six_cycles += ab * bc * neighbors[c]
        # Correct triangles that reuse one variable on two/three edges.
        for ns in neighborhoods:
            for a, b, c in combinations(ns, 3):
                six_cycles -= (adjacency[a][b] + adjacency[a][c]
                               + adjacency[b][c] - 2)
        duplicates = width - len(set(cols)) + int(0 in cols)
        return (len(rows) - len(reached), duplicates,
                four, six_cycles), cols

    def short4(cols):
        if 0 in cols or len(set(cols)) != len(cols):
            return (1, 10**9, 10**9)
        singles, pairs = set(cols), {}
        three_words = four_words = 0
        for i, x in enumerate(cols):
            for y in cols[:i]:
                value = x ^ y
                three_words += value in singles
                four_words += pairs.get(value, 0)
                pairs[value] = pairs.get(value, 0) + 1
        assert three_words % 3 == 0 and four_words % 3 == 0
        return (0, three_words // 3, four_words // 3)

    def short6_boundary(cols):
        """Exact <=6 existence; triple boundary is an exposure surrogate."""
        seen = {0}
        for x in cols:
            if x in seen:
                return True, 10**9
            seen.add(x)
        for i, x in enumerate(cols):
            for y in cols[:i]:
                value = x ^ y
                if value in seen:
                    return True, 10**9
                seen.add(value)
        boundary = 0
        for i, x in enumerate(cols):
            for j in range(i):
                pair = x ^ cols[j]
                for h in range(j):
                    value = pair ^ cols[h]
                    if value in seen:
                        return True, 10**9
                    seen.add(value)
                    weight = value.bit_count()
                    if weight <= 4:
                        boundary += (5 - weight) ** 2
        return False, boundary / len(cols)

    def sampled_protection(cols, side):
        """Actual kernel witnesses; minimum is an upper bound only.

        Finalists in this slot use common permutation streams. Counts
        and block loads describe sampled words, not the complete spectrum.
        """
        srng = random.Random(360023003 + 1009 * idx + side)
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

        for trial in range(64):
            srng.shuffle(order)
            pivots, basis = {}, []
            for c in order:
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
                    offer(word)
            for i, word in enumerate(basis):
                for other in basis[:i]:
                    offer(word ^ other)

            # Descent uses only genuine kernel-vector additions.
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

        block_mask = (1 << q) - 1
        exposure_threshold = 14 if side == 0 else 10
        vulnerable = [
            word for word in short
            if word.bit_count() <= exposure_threshold
        ]
        loads = [
            sum(bool((word >> (j * q)) & block_mask) for word in vulnerable)
            for j in range(len(cols) // q)
        ]
        return minimum, len(vulnerable), max(loads, default=0)

    def optimize(side, extra):
        width = len(counts[side][0]) * q
        sites = [(i, j) for i, row in enumerate(counts[side])
                 for j, count in enumerate(row) if count]
        strata, seen = {}, set()
        serial = 0

        def offer(grid, rows, cost, cols):
            nonlocal serial
            if cost[0] or cost[1]:
                return
            sig = signature(grid)
            if sig in seen:
                return
            seen.add(sig)
            low = short4(cols)
            deficit = len(rows) - gf2_rank(rows)
            serial += 1
            key = low + (cost[2], cost[3], spectral_error(grid, side))
            entry = (key, serial, clone(grid), list(rows), list(cols), deficit)
            bucket = strata.setdefault(deficit, [])
            bucket.append(entry)
            bucket.sort(key=lambda item: item[:2])
            del bucket[2:]

        def energy(cost):
            return (5000 * cost[0] + 2000 * cost[1]
                    + 160 * cost[2] + cost[3]) / q

        for restart in range(3):
            grid = [[draw_cell(side, i, j) for j in range(len(row))]
                    for i, row in enumerate(counts[side])]
            rows = component(grid, side)
            cost, cols = geometry(rows, width)
            offer(grid, rows, cost, cols)
            for step in range(220 + extra):
                i, j = rng.choice(sites)
                old = grid[i][j]
                grid[i][j] = draw_cell(side, i, j)
                proposed_rows = component(grid, side)
                proposed, proposed_cols = geometry(proposed_rows, width)
                delta = energy(proposed) - energy(cost)
                temperature = 15.0 * (0.978 ** step) + 0.12
                if delta <= 0 or rng.random() < math.exp(
                        -min(700.0, delta / temperature)):
                    grid, rows, cost, cols = (
                        grid, proposed_rows, proposed, proposed_cols)
                    if step % 10 == 0 or delta < 0:
                        offer(grid, rows, cost, cols)
                else:
                    grid[i][j] = old
            offer(grid, rows, cost, cols)

        # Retain rank diversity because only complete CSS ranks decide
        # which cell a pair occupies. Component rank is not substituted.
        candidates = [bucket[0] for bucket in strata.values()]
        candidates.sort(key=lambda item: item[:2])
        candidates = candidates[:5]
        if len(candidates) < 5:
            remaining = [entry for bucket in strata.values()
                         for entry in bucket[1:]]
            remaining.sort(key=lambda item: item[:2])
            candidates.extend(remaining[:5 - len(candidates)])

        refined = []
        target = 16 if side == 0 else 12
        for coarse, serial, grid, rows, cols, deficit in candidates:
            bad, boundary = short6_boundary(cols)
            d, short, peak = sampled_protection(cols, side)
            key = (coarse[0], coarse[1], coarse[2], int(bad),
                   max(0, target - d), short, peak, boundary,
                   coarse[3], coarse[4], coarse[5], -min(d, target))
            refined.append((key, serial, grid, rows, deficit))
        refined.sort(key=lambda item: item[:2])
        return refined

    def small_logical(checks, stabilizers, n):
        """Exact existence of a quantum logical of weight at most four."""
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

        cols = columns(checks, n)
        seen = {0: 0}
        for i, syndrome in enumerate(cols):
            word = 1 << i
            previous = seen.get(syndrome)
            if previous is None:
                seen[syndrome] = word
            elif nontrivial(word ^ previous):
                return True
        for i, x in enumerate(cols):
            for j in range(i):
                syndrome = x ^ cols[j]
                word = (1 << i) | (1 << j)
                previous = seen.get(syndrome)
                if previous is None:
                    seen[syndrome] = word
                elif nontrivial(word ^ previous):
                    return True
        return False

    for attempt in range(3):
        delta_geometry = slot_geometries[attempt]
        apool = optimize(0, 60 * attempt)
        bpool = optimize(1, 60 * attempt)
        options = []
        for ak, at, A, ar, ad in apool:
            for bk, bt, B, br, bd in bpool:
                hx, hz, n = lifted_matrices(G, A, B)
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if n != expected_n or not 72 <= k <= 100:
                    continue
                # Full ranks, not component deficits, decide admission.
                # Sampled protection rewards have side-specific caps.
                paired = tuple(v for t in range(len(ak))
                               for v in (max(ak[t], bk[t]), ak[t] + bk[t]))
                edges = sum(row.bit_count() for row in hx + hz)
                # Above the protection floors, pursue the higher
                # logical-count ceiling without leaving cell 0.150.
                key = (paired[:10] + (max(0, 84 - k), -k)
                       + paired[10:] + (edges, at, bt))
                options.append((key, A, B, hx, hz, k, ar, br))
        options.sort(key=lambda item: item[0])
        for key, A, B, hx, hz, k, ar, br in options:
            if (small_logical(hx, hz, expected_n)
                    or small_logical(hz, hx, expected_n)):
                continue
            verify_contract(A, B)
            assert css_ok(hx, hz)
            assert max(row.bit_count() for row in hx + hz) == max(wx, wz)
            assert max(wx, wz) <= min(11, W_MAX)
            assert 6 * sum(row.bit_count() for row in hx + hz) == expected_cnots

            # Confirm cached actions against the complete verified lift.
            actual_a = []
            for i in range(ra):
                for t in range(q):
                    row = hx[(i * cb) * q + t]
                    actual_a.append(sum(
                        ((row >> (j * cb * q)) & mask) << (j * q)
                        for j in range(ca)))
            actual_b = [row & ((1 << (cb * q)) - 1)
                        for row in hz[:rb * q]]
            assert actual_a == ar and actual_b == br
            assert 0.125 <= k / expected_n < 0.175
            return {"idx": idx, "design": design, "batch": batch,
                    "k": k, "screen": key, "cnots": expected_cnots,
                    "contract": (architecture, ratio_policy, geometry_number,
                                 attempt, geometry_key(delta_geometry)),
                    "cells": {"A": A, "B": B}}

    # Finite annealing can miss a useful complete-rank stratum. Search
    # fresh origins under the same support and relative-order contract;
    # never transform a saved code or weaken A-column generation.
    fallback = []
    for trial in range(128):
        A, B = [
            [[draw_cell(side, i, j) for j in range(len(row))]
             for i, row in enumerate(counts[side])]
            for side in range(2)
        ]
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != expected_n or not 72 <= k <= 100:
            continue
        ar, br = component(A, 0), component(B, 1)
        ag, ac = geometry(ar, ca * q)
        bg, bc = geometry(br, cb * q)
        coarse = (
            ag[0] + bg[0], ag[1] + bg[1],
            _same_basis_cycles(ar) + _same_basis_cycles(br),
            spectral_error(A, 0) + spectral_error(B, 1),
            max(0, 84 - k), -k, trial,
        )
        fallback.append((coarse, A, B, hx, hz, k, ac, bc))
        fallback.sort(key=lambda item: item[0])
        del fallback[8:]

    refined = []
    for coarse, A, B, hx, hz, k, ac, bc in fallback:
        if (small_logical(hx, hz, expected_n)
                or small_logical(hz, hx, expected_n)):
            continue
        bad_a, boundary_a = short6_boundary(ac)
        bad_b, boundary_b = short6_boundary(bc)
        da, low_a, peak_a = sampled_protection(ac, 0)
        db, low_b, peak_b = sampled_protection(bc, 1)
        key = (
            coarse[0], coarse[1], int(bad_a) + int(bad_b),
            max(0, 16 - da) + max(0, 12 - db),
            max(0, 84 - k), -k,
            max(low_a, low_b), low_a + low_b,
            max(peak_a, peak_b), max(boundary_a, boundary_b),
            coarse,
        )
        refined.append((key, A, B, hx, hz, k))
    if refined:
        refined.sort(key=lambda item: item[0])
        key, A, B, hx, hz, k = refined[0]
        verify_contract(A, B)
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx + hz) == max(wx, wz)
        assert max(wx, wz) <= min(11, W_MAX)
        assert 6 * sum(row.bit_count() for row in hx + hz) == expected_cnots
        assert 0.125 <= k / expected_n < 0.175
        return {"idx": idx, "design": design, "batch": batch,
                "k": k, "screen": key, "cnots": expected_cnots,
                "contract": (architecture, ratio_policy, geometry_number, 2,
                             geometry_key(delta_geometry)),
                "cells": {"A": A, "B": B}}
    raise ValueError("no unequal-component safe filling in cell 0.150")


def CELL(rng, G, side, i, j, ctx):
    """Return the selected full cell without further randomization."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Parent Z17 slots 31/50 had d2s=5 but worse-basis s=3.691/3.892 and large basis gaps; that screen value does not establish balanced circuit protection.
# REFUTED: Excluding involutory differences and minimizing four-cycles suffices for reliable r045 memory: fitness was 3.2178, measured Z failures were 112..137/250 shots, and every measured code had weight-three X-type logical witnesses.
# REFUTED: One-row syndrome-budget and exact short-logical selection suffices for reliable A4 x C3 memory: fitness was 3.3437; measured codes retained weight-three Z witnesses and failed 113..186 X shots in 250..500 despite predominantly stage-1 settlement.
# REFUTED: Regular degree-four A columns prevent block-localized short logicals: current-parent slot32 had three weight-twelve X-type witnesses confined to Acol2/Bcol1 and four Z failures/200000 flipping 165 logical qubits.
# REFUTED: Regular weight-ten extraction and first-stage syndrome settlement certify saturation: current slot6 settled all 200000 X shots at stage 1 but failed seven, flipping 55 logical qubits and scoring 6.740.
# REFUTED: The two-second screen finds every weight-twelve logical in this bed: current slot54 reported d2s=15, but the thirty-second search verified seven sector-two weight-twelve logical operators.
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
