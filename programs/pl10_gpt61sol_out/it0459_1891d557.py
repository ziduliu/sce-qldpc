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
CONCEPT = ("C4 semidirect C4 with inversion action, order 16, shapes (3,5)x(3,5): balanced components with injective singleton columns and full-group-generated binomial columns should improve circuit fitness beyond the parent's 7.3634 in cell 0.150 while permitting lower extraction exposure.")
GROUP = ("Retain the exact semidirect inversion construction rule and acting factor C4; change only m from nine to four. Size factor 34 requires a smaller lift, and q=16 is the largest admissible multiple of four, giving n=544. The shown C4:C4 equal-shape program scored 7.2524 and its slots5/8 had zero failures in both bases. This motivates the bed without establishing a causal group advantage or reusing that program's fillings.")
SHAPES = ("Equal (3,5) components each have 80 variables and 48 checks, hence dimension at least 32. At q=16, n=544 and kb=64 give guaranteed rate 2/17. Complete helper-built ranks must give 69<=k<=95, entirely inside cell0.150; component deficits are not substituted for quantum ranks. Parent slot32's two Z failures implicated Acol0/Bcol2 at1.5x and multiple Acol/Brow checks, while four other measured codes had no failures. Sparse attribution motivates balanced protection, not a claimed persistent hotspot.")
FILLING = ("Canonical unordered support pairs interleave lean 18-term and stronger 20-term components, supplying 66 distinct count contracts rather than saved-code labeling orbits. Every component has augmentation rank one in direction(1,1,0). Lean row totals are(6,6,6), with one degree-two singleton column and four degree-four columns; stronger totals are(7,7,6), with all columns degree four. Both CSS row-weight sums are<=11. Six-round CNOT budgets are27648/29184/30720, versus the parent's measured30672. Doubleton-only columns use complementary nonzero Frattini classes that generate all16 elements; actual isolated ranks verify their sole nonzero local kernel is the weight16 constant word. Singleton-bearing columns are verified injective. Fresh helper-derived voltage searches preserve rank diversity, prefer component connectivity and exclusion through six, and balance capped sampled protection, short-word multiplicity and triple-boundary exposure. Full matrices additionally exclude logicals through four. Target the parent's rare Z failures and54 collective flips; structural screens and smaller gate budgets do not certify ASC scheduling or circuit pL.")
FAMILY_NAME = "r015_C4byC4_35x35_evenrow_gatelean_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 16

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C4 semidirect C4 requires order 16")
    # Same inversion-semidir­ect construction rule, with new parameters.
    # x^4=y^4=1 and y*x*y^-1=x^-1.
    # The parity action is a homomorphism C4 -> Aut(C4).
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


# Odd columns augment to (1,1,0); even columns augment to zero.
# The degree-two type contains two singleton blocks, not a fragile
# binomial-only column. All other columns have degree four.
_EVENROW_COLUMN_TYPES = (
    (1, 1, 0), (1, 1, 2), (1, 3, 0), (3, 1, 0),
    (0, 2, 2), (2, 0, 2), (2, 2, 0),
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
    """Canonical even-row support contracts and fresh verified voltages."""
    import math
    from itertools import combinations_with_replacement

    q = G.q
    assert q == 16 and A_SHAPE == B_SHAPE == (3, 5)

    # Canonicalize count architectures under the two odd-row exchanges
    # and column permutations. This is not full Tanner canonization.
    support_keys = set()
    for entries in combinations_with_replacement(_EVENROW_COLUMN_TYPES, 5):
        totals = tuple(sum(col[i] for col in entries) for i in range(3))
        lean = (totals == (6, 6, 6)
                and entries.count((1, 1, 0)) == 1)
        strong = (totals == (7, 7, 6)
                  and entries.count((1, 1, 0)) == 0)
        if not (lean or strong):
            continue
        direct = tuple(sorted(entries))
        swapped = tuple(sorted((b, a, c) for a, b, c in entries))
        support_keys.add(min(direct, swapped))

    keys = sorted(support_keys, key=lambda key: (
        sum(map(sum, key)),
        -key.count((1, 1, 2)),
        sum(3 in col for col in key),
        key,
    ))
    assert len(keys) == 11
    profiles = [
        [[column[i] for column in key] for i in range(3)]
        for key in keys
    ]
    sizes = [sum(map(sum, grid)) for grid in profiles]
    tiers = {36: [], 38: [], 40: []}
    for a in range(len(profiles)):
        for b in range(a, len(profiles)):
            tiers[sizes[a] + sizes[b]].append((a, b))
    for tier in tiers.values():
        tier.sort(key=lambda pair: (max(pair), pair))

    # Interleave extraction budgets so the initial twenty slots already
    # contain lean/lean, lean/strong and strong/strong experiments.
    cards = []
    for position in range(max(map(len, tiers.values()))):
        for total in (36, 38, 40):
            if position < len(tiers[total]):
                cards.append(tiers[total][position])
    assert len(cards) == 66
    if not 0 <= idx < len(cards):
        raise ValueError("No unused canonical even-row support contract")
    pa, pb = cards[idx]
    sa, sb = [
        [list(row) for row in profiles[p]] for p in (pa, pb)
    ]
    counts = {"A": sa, "B": sb}
    assert _grid_weights(sa, sb) <= min(11, W_MAX)
    assert _no_dead_lines(sa, sb)
    for grid in counts.values():
        assert all(0 <= w <= min(3, CELL_MAX)
                   for row in grid for w in row)
        assert gf2_rank([
            sum((w & 1) << j for j, w in enumerate(row))
            for row in grid]) == 1
        assert all(sum(column) in (2, 4) for column in zip(*grid))

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

    def color(g):
        # Quotient by <x^2,y^2>; nonzero independent classes generate G.
        return ((g[0] & 1) << 1) | (g[1] & 1)

    palettes = {
        c: [g for g in G.elements
            if color(g) == c and orders[G.idx(g)] == 4]
        for c in (1, 2, 3)
    }
    assert all(len(pool) == 4 for pool in palettes.values())
    ratio_colors = {}
    pair_number = 0
    for side_number, side in enumerate(("A", "B")):
        grid = counts[side]
        assigned = [[None] * 5 for _ in range(3)]
        for j in range(5):
            column = tuple(grid[i][j] for i in range(3))
            if 1 not in column:
                rows = [i for i, weight in enumerate(column) if weight == 2]
                assert len(rows) == 2 and sum(column) == 4
                pair = ((2, 1), (1, 3), (2, 3))[
                    (idx + pair_number) % 3]
                pair_number += 1
                for i, c in zip(rows, pair):
                    assigned[i][j] = c
            else:
                for i, weight in enumerate(column):
                    if weight == 2:
                        assigned[i][j] = (2, 1, 3)[
                            (idx + side_number + j) % 3]
        ratio_colors[side] = assigned

    def draw_cell(side, i, j):
        weight = counts[side][i][j]
        if not weight:
            return []
        base = G.pick(rng)
        if weight == 1:
            return [base]
        if weight == 2:
            delta = rng.choice(palettes[ratio_colors[side][i][j]])
            cell = [base, G.mul(base, delta)]
        else:
            # Independent quotient classes give a noncommuting triangle;
            # all three pairwise relative orders are four.
            u, v = rng.choice(palettes[2]), rng.choice(palettes[1])
            assert G._key(G.mul(u, v)) != G._key(G.mul(v, u))
            cell = [base, G.mul(base, u), G.mul(base, v)]
        assert len({G._key(g) for g in cell}) == weight
        return cell

    def draw_grid(side):
        return [[draw_cell(side, i, j) for j in range(5)]
                for i in range(3)]

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    # Cache only singleton blocks obtained from the verified helper.
    # Final full-code projections independently verify this assembly.
    block_mask = (1 << q) - 1
    probe = [[[g] for g in G.elements]]
    ax, _, _ = lifted_matrices(G, probe, [[[]]])
    _, bz, _ = lifted_matrices(G, [[[]]], probe)
    actions = {"A": {}, "B": {}}
    for t, g in enumerate(G.elements):
        actions["A"][G.idx(g)] = tuple(
            (row >> (t * q)) & block_mask for row in ax)
        actions["B"][G.idx(g)] = tuple(
            (row >> (t * q)) & block_mask for row in bz)

    def component(side, grid):
        rows = []
        for grid_row in grid:
            out = [0] * q
            for j, cell in enumerate(grid_row):
                for g in cell:
                    for t, word in enumerate(actions[side][G.idx(g)]):
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

    def logical4(checks, stabilizers, width):
        """Exact existence of a nontrivial quantum logical through four."""
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

        cols = columns(checks, width)
        seen = {0: 0}
        for i, syndrome in enumerate(cols):
            word = 1 << i
            old = seen.get(syndrome)
            if old is None:
                seen[syndrome] = word
            elif nontrivial(word ^ old):
                return True
        for i, x in enumerate(cols):
            for j in range(i):
                syndrome = x ^ cols[j]
                word = (1 << i) | (1 << j)
                old = seen.get(syndrome)
                if old is None:
                    seen[syndrome] = word
                elif nontrivial(word ^ old):
                    return True
        return False

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
        for trial in range(64):
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
            # Bounded descent adds only genuine kernel vectors.
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

        # Preserve complete-rank opportunities rather than retain only
        # one component rank stratum with the smallest cycle count.
        pool = [bucket[0] for bucket in strata.values() if bucket]
        pool.sort(key=lambda item: item[:2])
        pool = pool[:8]
        if len(pool) < 8:
            extras = [entry for bucket in strata.values()
                      for entry in bucket[1:]]
            extras.sort(key=lambda item: item[:2])
            pool.extend(extras[:8 - len(pool)])

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
            target = 14
            key = (disconnected, bad, max(0, target - d),
                   low12, low, load, peak, risk, four, six)
            finalists.append({
                "grid": grid, "key": key, "bad": bad,
                "disconnected": disconnected, "d": d,
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
            # Equal-length components receive balanced protection.
            # Sampled minima are upper bounds, with no reward above14.
            da, db = max(0, 14 - a["d"]), max(0, 14 - b["d"])
            key = (
                a["disconnected"] + b["disconnected"],
                max(a["bad"], b["bad"]), a["bad"] + b["bad"],
                max(da, db), da + db,
                max(a["low12"], b["low12"]),
                a["low12"] + b["low12"],
                max(0, 84 - k), -k,
                max(a["peak"], b["peak"]),
                max(a["risk"], b["risk"]), a["risk"] + b["risk"],
                max(a["load"], b["load"]),
                max(a["low"], b["low"]), a["low"] + b["low"],
                max(a["four"], b["four"]),
                a["four"] + b["four"],
                a["six"] + b["six"],
            )
            if winner is None or key < winner[0]:
                if logical4(hx, hz, n) or logical4(hz, hx, n):
                    continue
                winner = (key, A, B, hx, hz, n, k)

    # A finite search can miss the target ranks. Search fresh candidates
    # with the SAME legal count/order card, rather than alter submitted
    # cells or assume component deficits predict nonabelian CSS ranks.
    if winner is None:
        for trial in range(192):
            A, B = draw_grid("A"), draw_grid("B")
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (k > 0 and 8 * k > n and 40 * k < 7 * n):
                continue
            ar, br = _component_rows(hx, hz, q)
            key = (_four_cycles(ar) + _four_cycles(br),
                   max(0, 84 - k), -k)
            if winner is None or key < winner[0]:
                if logical4(hx, hz, n) or logical4(hz, hx, n):
                    continue
                winner = (key, A, B, hx, hz, n, k)
        if winner is None:
            raise ValueError("Count/order card missed verified cell 0.150")

    key, A, B, hx, hz, n, k = winner
    assert n == 544 and css_ok(hx, hz)
    assert 69 <= k <= 95 and 0.125 < k / n < 0.175
    assert max(row.bit_count() for row in hx + hz) <= min(11, W_MAX)
    actual_a, actual_b = _component_rows(hx, hz, q)
    assert component("A", A)[0] == actual_a
    assert component("B", B)[0] == actual_b

    # Singleton-bearing columns are injective. In a doubleton-only
    # column, complementary Frattini classes generate all of G.
    # The verified rank, not that algebraic prediction, certifies the
    # sole nonzero column-local word is the weight16 constant word.
    for side, rows, grid in (("A", actual_a, A), ("B", actual_b, B)):
        for j in range(5):
            isolated = [
                (row >> (j * q)) & block_mask for row in rows]
            weights = [counts[side][i][j] for i in range(3)]
            expected_rank = q if 1 in weights else q - 1
            assert gf2_rank(isolated) == expected_rank
            if expected_rank == q - 1:
                assert all(row.bit_count() % 2 == 0 for row in isolated)
            for i in range(3):
                cell = grid[i][j]
                assert len(cell) == weights[i]
                assert len({G._key(g) for g in cell}) == len(cell)
                if len(cell) == 2:
                    delta = G.mul(G.inv(cell[0]), cell[1])
                    assert color(delta) == ratio_colors[side][i][j]
                    assert orders[G.idx(delta)] == 4
                elif len(cell) == 3:
                    assert all(
                        orders[G.idx(G.mul(G.inv(cell[a]), cell[b]))] == 4
                        for a in range(3) for b in range(a + 1, 3))

    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    expected_cnots = (
        (cb + rb) * sum(map(sum, sa))
        + (ca + ra) * sum(map(sum, sb))
    ) * q * 6
    assert expected_cnots in (27648, 29184, 30720)
    assert sum(row.bit_count() for row in hx + hz) * 6 == expected_cnots
    return {"A": A, "B": B, "design": (pa, pb),
            "support_contract": (keys[pa], keys[pb]),
            "ratio_classes": ratio_colors,
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
# REFUTED: Weight-eleven extraction and component order/boundary screening guarantee zero-failure memory: current-parent slot32 had two Z failures/200000 and54 collective logical flips, while slot35 at the same k=82 had none.
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
