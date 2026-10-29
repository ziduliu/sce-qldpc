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
CONCEPT = ("C9 semidirect C4 with inversion action, order 36, shapes (2,3)x(2,4): mixed-order regular A doubletons and lighter B components, selected by verified fault-bundle boundaries after capped protection screening, should improve worse-basis circuit fitness above the parent's 6.7896.")
GROUP = ("Frozen exact order-36 group and encoding under L1. This bed has reached about 7.4. Parent slots52/58 scored 7.101/7.225 with weight-eleven extraction, versus 6.413/6.186 for heavier slots5/11. These readings motivate changing filling and extraction exposure, not a group change or a causal claim about its presentation.")
SHAPES = ("Frozen A=(2,3), B=(2,4): n=576, kb=72 and guaranteed rate 1/8; the size window permits q<=37 and the fixed group uses q=36. Components have 108/144 variables and 72 checks, hence dimensions at least 36/72. Complete verified ranks require 73<=k<=100 inside cell0.150. Parent Z failures implicated several Acol/Bcol data and Acol/Brow check blocks; sparse enrichment maps do not establish one persistent hotspot.")
FILLING = ("Cross canonical mixed-order A contracts with four B incidence architectures. All A cells are doubletons with column degree four; each A row contains order-four and rotational relative differences, and the joint contract contains orders nine and eighteen. B has 12,13 or14 terms, column degrees three/four and row totals <=7, giving CSS weight <=11 and 28512/29592/30672 six-round CNOTs rather than32832. Optimize fresh voltages using helper-derived actions and retain component rank diversity. Exact kernel screening through six and common-stream sampled protection floors15/12 shortlist pairs; above those floors, compare complete-code pair/triple syndrome boundaries within opposite checks, short-witness burden and normalized block exposure. These bundles are possible correlated-fault opportunities, not ASC-certified hooks. Target predominantly Z-limited collective failures: slot11 failed45/200000 and flipped1620 logicals, often including logical50, whereas the lower-exposure slots had only1..2 failures. All full ranks, CSS and final cached projections are verified; surrogates remain uncalibrated hypotheses about circuit pL.")
FAMILY_NAME = "r015_dic9_23x24_lightB_mixedorder_faultbundle_v3"

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
    """Lighter incidence contracts and fresh mixed-order voltage searches."""
    import math
    from itertools import product, combinations

    q = G.q
    assert q == 36
    assert A_SHAPE == (2, 3) and B_SHAPE == (2, 4)
    sa = [[2, 2, 2], [2, 2, 2]]
    b_profiles = (
        ((1, 1, 2, 2), (2, 2, 1, 1)),
        ((0, 2, 2, 2), (3, 1, 1, 1)),
        ((1, 1, 2, 2), (2, 2, 2, 1)),
        ((1, 1, 2, 3), (2, 2, 2, 1)),
    )
    mask_number, architecture = divmod(idx, len(b_profiles))
    sb = [list(row) for row in b_profiles[architecture]]
    counts = {"A": sa, "B": sb}
    assert _grid_weights(sa, sb) <= min(11, W_MAX)
    assert _no_dead_lines(sa, sb)
    assert all(0 <= w <= CELL_MAX
               for support in counts.values() for row in support for w in row)

    # Canonical ratio-order contracts remove obvious row/column copies.
    # Each row mixes quotient-changing and rotational differences.
    # These are not claimed to classify arbitrary Tanner isomorphism.
    keys = set()
    for values in product((4, 9, 18), repeat=6):
        rows = (values[:3], values[3:])
        if any(4 not in row or all(t == 4 for t in row) for row in rows):
            continue
        if 9 not in values or 18 not in values:
            continue
        direct = tuple(sorted(zip(*rows)))
        reverse = tuple(sorted((b, a) for a, b in direct))
        keys.add(min(direct, reverse))
    masks = sorted(keys, key=lambda key: (
        abs(sum(a == 4 for a, b in key) - sum(b == 4 for a, b in key)),
        sum(a == b for a, b in key), key))
    if mask_number >= len(masks):
        raise ValueError("Unused mixed-order contract book exhausted")
    chosen = masks[mask_number]
    a_orders = tuple(tuple(column[i] for column in chosen) for i in range(2))

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

    # Noncommuting triangles preserve distinctness and exclude short
    # relative orders. Additional contracts also change B ratio geometry.
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
            length = (a_orders[i][j] if side == "A" else
                      (4, 9, 18)[(i + 2*j + mask_number) % 3])
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
        target = 15 if side == "A" else 12
        for cost, number, grid, rows, cols in candidates:
            bad = int(_small_kernel_word(rows, width))
            d, low12, burden, load = sampled_words(cols, side)
            boundary, peak = triple_exposure(cols)
            item = {
                "grid": grid, "rows": rows, "cost": cost, "bad": bad,
                "d": d, "gap": max(0, target-d), "low12": low12,
                "burden": burden, "load": load,
                "boundary": boundary, "peak": peak,
            }
            key = (cost[0], cost[1], bad, item["gap"], low12,
                   peak, boundary, burden, load, cost[2], cost[3], number)
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
                )
                tail = (
                    max(a["peak"], b["peak"]),
                    a["boundary"] + b["boundary"],
                    a["burden"] + b["burden"],
                    max(a["load"], b["load"]),
                    a["cost"][2] + b["cost"][2],
                    a["cost"][3] + b["cost"][3], -k,
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
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) <= min(11, W_MAX)
    expected_cnots = (72 + 5*sum(map(sum, sb)))*q*6
    assert 6*sum(row.bit_count() for row in hx + hz) == expected_cnots
    assert expected_cnots in (28512, 29592, 30672)
    assert 0.125 < k/576 < 0.175
    assert all(len(cell) == counts[side][i][j]
               and len({G._key(g) for g in cell}) == len(cell)
               for side, grid in (("A", A), ("B", B))
               for i, row in enumerate(grid) for j, cell in enumerate(row))
    return {"A": A, "B": B, "k": k, "cnots": expected_cnots,
            "contract": (chosen, architecture, triangle_length),
            "fault_bundle_surrogate": exposure, "screen": key}


def CELL(rng, G, side, i, j, ctx):
    """Emit the verified complete cell without further randomization."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Weight-legal rolled supports with uniform labels reliably balance the two bases: the parent had 100+ X failures in 750..33250 shots versus only 0..12 Z failures in 200000 shots.
# REFUTED: Excluding component kernel words <=6 and minimizing component four/six cycles guarantees ASC measurability: current slot 14 passed these screens but its X schedule timed out after 120 seconds.
# REFUTED: Equal d2s predicts equal circuit reliability: current slots 1 and 11 both had d2s=15, but their worse bases had zero and 13 failures in 200000 shots.
# REFUTED: d2s=16 ensures near-ceiling memory on this order-36 bed: parent slot11 had45 Z failures/200000 and s=6.186, whereas slot58 had one failure per basis and s=7.225.
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
