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
CONCEPT = ("C9 semidirect C4 with inversion action, order 36, shapes (2,3)x(2,4): zero-containing B rewiring, mixed A ratio orders and stressed component-BP selection should improve worse-basis circuit fitness above 7.3634 without increasing extraction cost.")
GROUP = ("Frozen exact order-36 group under L1. This bed scores 7.3634; four measured codes had zero failures in both bases, and the fifth had only two Z failures in 200000 shots. Keep the group to isolate a filling experiment, not to claim a causal advantage for its presentation.")
SHAPES = ("Frozen (2,3)x(2,4): n=576, kb=72 and guaranteed rate 1/8; q<=37 fits the size window and this group fixes q=36. Components have 108/144 variables and 72 checks, hence dimensions at least 36/72. Complete verified ranks retain 0.125<k/n<0.175 in cell 0.150. Parent slot32 implicated Acol0/Bcol2 at 1.5x and several Acol/Brow checks, but two failed shots do not establish a persistent hotspot.")
FILLING = ("Twenty contracts cross four upper-A order-nine/eighteen patterns with five zero-containing B support profiles. A stays regular doubleton; B has 13/14 terms, column degrees three/four and row totals <=7, giving CSS weight <=11 and 29592/30672 six-round CNOTs. Trinomial relative-order spectra distinguish further harness phases. Optimize helper-derived components; retain connectivity, exact kernel screening through six and sampled protection capped at sixteen for A and twelve for B. Then prefer common-random-number classical BP probes with degree-weighted data noise and correlated pairs, prioritizing the Z-limiting A component. Parent slot32 collectively flipped 54 logicals in two Z failures, while slot35 saturated at the same k=82. Probes omit ASC ordering, measurement histories and the decoder cascade; they are uncalibrated selection surrogates, not circuit-pL estimates. Verify final full ranks, CSS and cached component equality.")
FAMILY_NAME = "r015_dic9_23x24_zero_mixedorder_bp_v3"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 36

def custom_ops(q):
    # C9 semidirect C4: y*x*y^-1 = x^-1, y^4 = 1.
    # The parity action is a homomorphism C4 -> Aut(C9).
    m, r = 9, 4
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

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb)


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


def NEW_CTX(rng, G, idx):
    """Zero-containing support contracts and decoder-aware fresh voltages."""
    import math
    from itertools import combinations

    q = G.q
    assert q == 36
    sa = [[2, 2, 2], [2, 2, 2]]
    # The first four profiles have row totals 7/7 and column totals
    # 3/3/4/4. The fifth has row totals 6/7 and columns 3/3/3/4.
    # Unlike the parent, every contract contains a genuine zero cell.
    b_profiles = (
        ((0, 2, 2, 3), (3, 1, 2, 1)),
        ((0, 1, 3, 3), (3, 2, 1, 1)),
        ((0, 3, 1, 3), (3, 0, 3, 1)),
        ((0, 3, 2, 2), (3, 0, 2, 2)),
        ((0, 2, 2, 2), (3, 1, 1, 2)),
    )
    phase, design = divmod(idx, 20)
    if phase >= 4:
        raise ValueError("No unused zero/order contract for this slot")
    a_model, b_model = divmod(design, 5)
    sb = [list(row) for row in b_profiles[b_model]]
    counts = {"A": sa, "B": sb}
    assert _grid_weights(sa, sb) <= min(11, W_MAX)
    assert _no_dead_lines(sa, sb)
    assert all(0 <= w <= min(3, CELL_MAX)
               for grid in counts.values() for row in grid for w in row)
    assert all(3 <= sum(row[j] for row in sb) <= 4 for j in range(4))

    # Mixed upper-row orders change relative geometry, not just labels
    # or origins. The zero-containing B profiles also change incidence.
    # Downstream arbitrary Tanner-isomorphism testing remains definitive.
    a_orders = (
        ((9, 9, 9), (4, 4, 4)),
        ((9, 9, 18), (4, 4, 4)),
        ((9, 18, 18), (4, 4, 4)),
        ((18, 18, 18), (4, 4, 4)),
    )[a_model]
    b_double_orders = ((4, 9, 18), (4,), (18,), (9,))[phase]
    long_orders = (4, 9, 18)

    orders = []
    for g in G.elements:
        power = G.e
        for order in range(1, q + 1):
            power = G.mul(power, g)
            if G._key(power) == G._key(G.e):
                orders.append(order)
                break
        else:
            raise ValueError("Element order exceeded the group order")
    relative = [
        [orders[G.idx(G.mul(G.inv(g), h))] for h in G.elements]
        for g in G.elements
    ]

    def allowed(side, i, j):
        if side == "A":
            return (a_orders[i][j],)
        return b_double_orders if sb[i][j] == 2 else long_orders

    # Enumerate relative triangles once, removing the common cell origin.
    # Their order spectra are invariant under group automorphisms.
    triangle_book = {}
    nonidentity = [t for t in range(q) if G.elements[t] != G.e]
    for target in ((4, 4, 9), (4, 4, 18),
                   (9, 18, 18), (9, 9, 9)):
        book = []
        for u, v in combinations(nonidentity, 2):
            spectrum = tuple(sorted((orders[u], orders[v], relative[u][v])))
            if spectrum == target:
                book.append((G.e, G.elements[u], G.elements[v]))
        assert book
        triangle_book[target] = book

    def triangle_target(i, j):
        if phase == 0:
            return (4, 4, 9)
        if phase == 1:
            return (4, 4, 18)
        if phase == 2:
            return (4, 4, 9) if (i + j) % 2 else (9, 18, 18)
        return (4, 4, 18) if (i + j) % 2 else (9, 9, 9)

    def draw_cell(weight, palette, target=None):
        if not weight:
            return []
        if weight == 3:
            assert target in triangle_book
            origin = G.pick(rng)
            cell = [G.mul(origin, g)
                    for g in rng.choice(triangle_book[target])]
            assert len({G._key(g) for g in cell}) == 3
            return cell
        for _ in range(64):
            chosen = [rng.randrange(q)]
            for t in range(1, weight):
                available = [
                    h for h in range(q)
                    if all(relative[g][h] in palette for g in chosen)
                ]
                if not available:
                    break
                chosen.append(rng.choice(available))
            if len(chosen) == weight:
                return [G.elements[t] for t in chosen]
        raise ValueError("Unable to realize the prescribed relative orders")

    def draw_grid(side):
        return [
            [draw_cell(w, allowed(side, i, j),
                       triangle_target(i, j) if side == "B" and w == 3 else None)
             for j, w in enumerate(row)]
            for i, row in enumerate(counts[side])
        ]

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    # Cache only blocks obtained from the verified construction helper.
    # This makes the additional decoder screening affordable without
    # introducing a separately implemented left/right action.
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
                    block = actions[side][G.idx(g)]
                    for t, word in enumerate(block):
                        out[t] ^= word << (j * q)
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
        threshold = 16 if side == "A" else 12
        for trial in range(40):
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

    def bp_probe(rows, cols, side):
        """Stressed classical component BP, NOT circuit memory simulation.

        Equal latent random streams compare finalists within this slot.
        Every settled nonzero residual is a classical kernel failure.
        Unresolved shots are ambiguity penalties, not measured pL.
        """
        import numpy as np

        width, nr, shots = len(cols), len(rows), 192
        neighborhoods = []
        for row in rows:
            neighbors = []
            while row:
                bit = row & -row
                neighbors.append(bit.bit_length() - 1)
                row ^= bit
            neighborhoods.append(neighbors)

        lengths = np.asarray([len(row) for row in neighborhoods],
                             dtype=np.int32)
        weight = int(lengths.max())
        incidence = np.full((nr, weight), width, dtype=np.int32)
        for r, neighbors in enumerate(neighborhoods):
            incidence[r, :len(neighbors)] = neighbors
        flat = incidence.ravel()
        valid = flat != width
        edges = len(flat)
        degrees = np.asarray([col.bit_count() for col in cols],
                             dtype=np.int32)
        assert np.all(degrees > 0)
        variable_edges = np.full(
            (width, int(degrees.max())), edges, dtype=np.int32)
        used = np.zeros(width, dtype=np.int32)
        for edge in np.flatnonzero(valid):
            variable = flat[edge]
            variable_edges[variable, used[variable]] = edge
            used[variable] += 1

        channel = np.random.default_rng(
            360731 + 1009 * idx + int(side == "B"))
        levels = np.asarray(
            (0.025, 0.045, 0.065) if side == "A"
            else (0.020, 0.035, 0.050))
        normalization = 4.0 if side == "A" else 3.0
        rates = (levels[np.arange(shots) % 3, None]
                 * degrees[None, :] / normalization)
        error = channel.random((shots, width)) < rates
        error = np.concatenate(
            (error, np.zeros((shots, 1), dtype=bool)), axis=1)

        hook_rows = channel.integers(nr, size=shots)
        hooks = channel.random(shots) < 0.15
        for shot in np.flatnonzero(hooks):
            r = int(hook_rows[shot])
            first = int(channel.integers(int(lengths[r])))
            second = int(channel.integers(int(lengths[r]) - 1))
            second += second >= first
            error[shot, incidence[r, first]] ^= True
            error[shot, incidence[r, second]] ^= True

        prior_rates = np.clip(
            rates + 0.30 * degrees[None, :] / int(valid.sum()),
            1e-6, 0.25)
        prior = np.log((1.0 - prior_rates) / prior_rates)
        prior = np.concatenate((prior, np.zeros((shots, 1))), axis=1)
        target = (
            error[:, flat].reshape(shots, nr, weight).sum(axis=2) & 1
        ).astype(bool)
        target_sign = np.where(target, -1.0, 1.0)
        messages = np.where(valid[None, :], prior[:, flat], 24.0)
        failed = iteration_sum = 0

        for iteration in range(1, 33):
            count = len(error)
            shaped = messages.reshape(count, nr, weight)
            signs = np.where(shaped < 0, -1.0, 1.0)
            logarithms = np.log(np.clip(
                np.tanh(np.abs(shaped) / 2.0), 1e-12, 1.0 - 1e-7))
            logarithms = np.where(
                valid.reshape(1, nr, weight), logarithms, 0.0)
            products = np.minimum(np.exp(
                logarithms.sum(axis=2, keepdims=True) - logarithms),
                1.0 - 1e-7)
            magnitude = np.log1p(products) - np.log1p(-products)
            outgoing = (
                magnitude * signs * signs.prod(axis=2, keepdims=True)
                * target_sign[:, :, None]
            ).reshape(count, edges)
            outgoing *= valid[None, :]

            padded = np.zeros((count, edges + 1))
            padded[:, :edges] = outgoing
            posterior = (prior[:, :width]
                         + padded[:, variable_edges].sum(axis=2))
            posterior = np.concatenate(
                (posterior, np.zeros((count, 1))), axis=1)
            hard = posterior < 0
            syndrome = (
                hard[:, flat].reshape(count, nr, weight).sum(axis=2) & 1
            ).astype(bool)
            settled = np.all(syndrome == target, axis=1)
            if np.any(settled):
                failed += int(np.count_nonzero(np.any(
                    error[settled, :width] ^ hard[settled, :width],
                    axis=1)))
                iteration_sum += iteration * int(settled.sum())

            keep = ~settled
            if not np.any(keep):
                return failed, 0, iteration_sum
            messages = np.where(
                valid[None, :],
                np.clip(0.35 * messages
                        + 0.65 * (posterior[:, flat] - outgoing), -24, 24),
                24.0)[keep]
            prior = prior[keep]
            error = error[keep]
            target = target[keep]
            target_sign = target_sign[keep]

        unresolved = len(error)
        return (failed + unresolved, unresolved,
                iteration_sum + 33 * unresolved)

    def portfolio(side):
        supports = counts[side]
        positions = [(i, j) for i, row in enumerate(supports)
                     for j, w in enumerate(row) if w]
        pool = []
        seen = set()
        serial = 0

        def remember(grid, rows, stats):
            nonlocal serial
            coarse = (stats[1], stats[2])
            if len(pool) >= 6 and coarse > pool[-1][0]:
                return
            signature = tuple(
                tuple(sorted(G.idx(g) for g in cell))
                for row in grid for cell in row)
            if signature in seen:
                return
            seen.add(signature)
            serial += 1
            pool.append((coarse, serial, clone(grid), list(rows)))
            pool.sort(key=lambda item: item[:2])
            del pool[6:]

        for restart in range(3):
            grid = draw_grid(side)
            rows, width = component(side, grid)
            current = cycle_energy(rows, width)
            remember(grid, rows, current)
            for step in range(260):
                i, j = rng.choice(positions)
                old = grid[i][j]
                w = supports[i][j]
                target = (triangle_target(i, j)
                          if side == "B" and w == 3 else None)
                grid[i][j] = draw_cell(w, allowed(side, i, j), target)
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

        finalists = []
        for coarse, serial, grid, rows in pool:
            width = len(grid[0]) * q
            cols = columns(rows, width)
            bad = int(_small_kernel_word(rows, width))
            disconnected = int(not connected(rows, width))
            d, low12, low, load = sampled_words(cols, side)
            risk, peak = boundary_exposure(cols)
            four = coarse[0]
            six = _six_cycles(rows)
            target = 16 if side == "A" else 12
            probe = bp_probe(rows, cols, side)
            # Preserve the short-word/protection priorities. Once those
            # floors are met, decoder ambiguity precedes uncalibrated
            # witness multiplicity and boundary counts.
            key = (disconnected, bad, max(0, target - d),
                   probe[0], probe[1], probe[2],
                   low12, low, load, peak, risk, four, six)
            finalists.append({
                "grid": grid, "key": key, "bad": bad,
                "disconnected": disconnected, "d": d,
                "probe": probe,
                "low12": low12, "low": low, "load": load,
                "risk": risk, "peak": peak,
                "four": four, "six": six,
                "deficit": len(rows) - gf2_rank(rows),
            })
        finalists.sort(key=lambda item: item["key"])
        return finalists

    apool, bpool = portfolio("A"), portfolio("B")
    winner = None
    for a in apool:
        for b in bpool:
            A, B = a["grid"], b["grid"]
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (k > 0 and 8 * k > n and 40 * k < 7 * n):
                continue
            # Do not chase an arbitrarily optimistic sampled distance.
            # Stronger A protection is motivated by the measured Z limit.
            key = (
                a["disconnected"] + b["disconnected"],
                2 * a["bad"] + b["bad"],
                max(0, 16 - a["d"]) + max(0, 12 - b["d"]),
                a["probe"][0],
                max(a["probe"][0], b["probe"][0]),
                a["probe"][0] + b["probe"][0],
                a["probe"][1], b["probe"][1],
                a["probe"][2] + b["probe"][2],
                a["low12"], a["low"], a["load"],
                a["peak"], a["risk"],
                b["low12"], b["low"], b["load"],
                b["peak"], b["risk"],
                max(a["four"], b["four"]),
                a["four"] + b["four"],
                a["six"] + b["six"],
                -k,
            )
            if winner is None or key < winner[0]:
                winner = (key, A, B, hx, hz, n, k)

    # A finite search can miss the target ranks. Search fresh candidates
    # with the SAME legal count/order card, rather than alter submitted
    # cells or assume component deficits predict nonabelian CSS ranks.
    if winner is None:
        for trial in range(96):
            A, B = draw_grid("A"), draw_grid("B")
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (k > 0 and 8 * k > n and 40 * k < 7 * n):
                continue
            ar, br = _component_rows(hx, hz, q)
            key = (_four_cycles(ar) + _four_cycles(br),)
            if winner is None or key < winner[0]:
                winner = (key, A, B, hx, hz, n, k)
        if winner is None:
            raise ValueError("Count/order card missed verified cell 0.150")

    key, A, B, hx, hz, n, k = winner
    assert n == 576 and css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) <= min(11, W_MAX)
    # Independently confirm that cached screening components are exactly
    # the projections of the final helper-built submitted matrices.
    actual_a, actual_b = _component_rows(hx, hz, q)
    assert component("A", A)[0] == actual_a
    assert component("B", B)[0] == actual_b
    assert all(len({G._key(g) for g in cell}) == len(cell)
               for grid in (A, B) for row in grid for cell in row)
    expected_cnots = (72 + 5 * sum(map(sum, sb))) * q * 6
    assert sum(row.bit_count() for row in hx + hz) * 6 == expected_cnots
    return {"A": A, "B": B, "design": (a_model, b_model, phase),
            "k": k, "screen": key, "cnots": expected_cnots}


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
# REFUTED: Weight-eleven extraction and component boundary selection certify saturation: current slot32 had two Z failures/200000 flipping 54 logicals and s=7.134, while slot35 at the same k=82 had zero failures and s=7.426.
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
