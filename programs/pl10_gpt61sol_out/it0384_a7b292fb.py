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
CONCEPT = ("For C5 semidirect C8 with inversion action, order 40, shapes (1,2)x(2,6), a unit A trinomial, long-order B ratios and lower extraction exposure should improve worse-basis circuit reliability in cell 0.300 beyond its approximately 6.2 incumbent.")
GROUP = ("Jump from cyclic C29 to nonabelian C5:C8, x^5=y^8=1 and yxy^-1=x^-1. The related order-40 centre-four/16-class bed reached about 5.9 on these shapes; other nonabelian (1,2)x(2,6) beds reached 6.2. These readings motivate a fresh filling, not a group-only advantage. Three distinct powers in an order-eight subgroup have odd augmentation and form a unit in its characteristic-two group algebra; actual helper-built ranks are also required.")
SHAPES = ("Target the weaker cell 0.300 with (1,2)x(2,6): q40 gives n560, kb160 and guaranteed rate 2/7=0.285714. A has 80 variables/40 checks and B has 240 variables/80 checks. Require actual complete k160, so rank excess cannot silently move the program. Parent Z failures reached ten/200000 in slot1 and nine/200000 in slot16; Acol/Bcol enrichments varied between slots, while Acol/Brow check enrichments were generally weak. These sparse maps do not identify one persistent block defect.")
FILLING = ("Twenty deliberate contracts cross B budgets 15,16,17,18 with five A-partner triangle/order policies. A0 is an order-eight-subgroup unit trinomial; A1 explores a different subgroup or mixed long-order geometry. B columns have degree two or three, never one. Weak matching holonomies use order twenty and noncommuting order eight; internal degree-two cells use order twenty. Additional harness batches change zero-cell incidence rather than replaying saved fillings. Both CSS weight sums are checked, with maxima eleven or twelve and 22320..24480 six-round CNOTs instead of the parent's 33408. Helper-derived component optimization screens exact kernels through four, weak-column kernels through six, capped sampled witnesses and boundary concentration. Complete ranks, CSS and quantum logicals through four guard joint two-interval noisy-syndrome BP selection with an independent holdout. The probe omits ASC ordering, six-round histories and relay-BP/LSD: it is not measured pL. The hypothesis targets fewer collective failures and decoder fall-throughs through reduced exposure and balanced protection.")
FAMILY_NAME = "r030_C5byC8inv_q40_12x26_unit_longweak_holdout_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 40

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C5 semidirect C8 requires order 40")
    # x^5=y^8=1, y*x*y^-1=x^-1.
    # The inversion action has order two, dividing eight.
    els = [(a, b) for b in range(8) for a in range(5)]

    def red(g):
        return (int(g[0]) % 5, int(g[1]) % 8)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b % 2 == 0 else -c)) % 5,
                (b + d) % 8)

    def inv(g):
        a, b = red(g)
        return ((-a if b % 2 == 0 else a) % 5, (-b) % 8)

    def idx(g):
        a, b = red(g)
        return a + 5*b

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (2, 6)               # (rb, cb)


def _retired_C29_ctx(rng, G, idx):
    """Historical C29 search; not called on the new group or shapes."""
    import math
    from itertools import combinations

    q = G.q
    assert q == 29
    assert A_SHAPE == B_SHAPE == (2, 4)
    sa = [[2] * 4 for _ in range(2)]
    sb = [[2] * 4 for _ in range(2)]

    # HX: eight A terms plus four B terms.
    # HZ: eight B terms plus four A terms.
    assert _grid_weights(sa, sb) == 12
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)
    assert CELL_MAX >= 2

    def direction(value):
        value %= q
        return min(value, q - value)

    def row_clean(gaps):
        signed = tuple(v for gap in gaps for v in (gap, -gap))
        # Repeated directions are included in this test. It excludes
        # same-check-orbit triangles, not all component six-cycles.
        return all((a + b + c) % q
                   for a in signed for b in signed for c in signed)

    row_book = [
        gaps for gaps in combinations(range(1, (q + 1) // 2), 4)
        if row_clean(gaps)
    ]
    assert len(row_book) >= 4

    def geometry_key(grids):
        """Remove obvious joint gap orbits, not arbitrary Tanner equivalence."""
        images = []
        for u in range(1, (q + 1) // 2):
            sides = []
            for grid in grids:
                pairs = [
                    (direction(u * grid[0][j]),
                     direction(u * grid[1][j]))
                    for j in range(4)
                ]
                forward = tuple(sorted(pairs))
                reverse = tuple(sorted((b, a) for a, b in pairs))
                sides.append(min(forward, reverse))
            images.append(tuple(sorted(sides)))
        return min(images)

    # Pure deterministic catalogue: slots differ in prescribed relative
    # geometry before any voltage search. Extra slots consume new cards.
    # No earlier selected code is transformed into another submitted slot.
    catalogue_rng = random.Random(290419731)
    book, seen = [], set()
    for attempt in range(10000):
        grids = []
        for side in range(2):
            upper = list(catalogue_rng.choice(row_book))
            lower = list(catalogue_rng.choice(row_book))
            for shuffle in range(64):
                catalogue_rng.shuffle(lower)
                if all(a != b for a, b in zip(upper, lower)):
                    break
            else:
                break
            grids.append([upper, list(lower)])
        if len(grids) != 2:
            continue
        signature = geometry_key(grids)
        if signature in seen:
            continue
        seen.add(signature)
        book.append((signature, grids))
        if len(book) > idx:
            break
    if len(book) <= idx:
        raise ValueError("No unused sum-free joint geometry contract")

    contract, grids = book[idx]
    geometry = dict(zip(("A", "B"), grids))
    for grid in geometry.values():
        assert all(row_clean(row) for row in grid)
        assert all(len(set(row)) == 4 for row in grid)
        assert all(grid[0][j] != grid[1][j] for j in range(4))

    def draw_cell(side, i, j):
        origin = G.pick(rng)
        gap = geometry[side][i][j]
        return [origin, G.mul(origin, (gap,))]

    def draw_grid(side):
        return [[draw_cell(side, i, j) for j in range(4)]
                for i in range(2)]

    def copy_grid(grid):
        return [[list(cell) for cell in row] for row in grid]

    # Cache only singleton blocks returned by the verified construction.
    # Final component projections are independently checked below.
    actions = {"A": {}, "B": {}}
    block_mask = (1 << q) - 1
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
                    block = actions[side][G.idx(g)]
                    for t, bits in enumerate(block):
                        out[t] ^= bits << (j * q)
            rows.extend(out)
        return rows, len(grid[0]) * q

    def columns(rows, nc):
        out = [0] * nc
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= tag
                row ^= bit
        return out

    def cycle_energy(rows, nc):
        nr = len(rows)
        adj = [0] * nr
        four = 0
        for i in range(nr):
            for j in range(i):
                common = (rows[i] & rows[j]).bit_count()
                four += common * (common - 1) // 2
                if common:
                    adj[i] |= 1 << j
                    adj[j] |= 1 << i
        triangles = 0
        for i in range(nr):
            lower = adj[i] & ((1 << i) - 1)
            while lower:
                bit = lower & -lower
                j = bit.bit_length() - 1
                triangles += (adj[i] & adj[j]
                              & ((1 << j) - 1)).bit_count()
                lower ^= bit
        stars = 0
        for col in columns(rows, nc):
            d = col.bit_count()
            stars += d * (d - 1) * (d - 2) // 6
        # Exact six-cycle count without four-cycles; otherwise a heuristic.
        six = max(0, triangles - stars)
        return (80.0 * four + six / 4.0) / q, four, six

    def boundary_scan(cols):
        """Exact existence test for kernel words of weights 1..6.

        Collisions between subsets of size at most three cover all six
        weights. Boundary statistics are only trapping-set heuristics.
        """
        nc = len(cols)
        seen = {0: 0}
        bad = False
        risk = 0
        loads = [0] * (nc // q)

        def record(syndrome, word):
            nonlocal bad
            old = seen.get(syndrome)
            if old is None:
                seen[syndrome] = word
            elif old != word:
                bad = True

        for i, col in enumerate(cols):
            record(col, 1 << i)
        for i in range(nc):
            for j in range(i):
                record(cols[i] ^ cols[j], (1 << i) | (1 << j))
        for i in range(nc - 2):
            for j in range(i + 1, nc - 1):
                pair = cols[i] ^ cols[j]
                word = (1 << i) | (1 << j)
                for t in range(j + 1, nc):
                    syndrome = pair ^ cols[t]
                    record(syndrome, word | (1 << t))
                    weight = syndrome.bit_count()
                    if weight <= 6:
                        cost = (8 - weight) ** 2
                        risk += cost
                        loads[i // q] += cost
                        loads[j // q] += cost
                        loads[t // q] += cost
        return int(bad), risk / nc, max(loads) / q

    def kernel_exposure(cols, trials, side):
        """Sample actual kernel words; protection remains an upper bound.

        Common permutation streams compare finalists in a slot.
        Fundamental words, pair XORs and bounded descent expose ambiguity;
        no extra protection reward is given above weight sixteen.
        """
        srng = random.Random(290791 + 1009 * idx + int(side == "B"))
        nc = len(cols)
        order = list(range(nc))
        distance = nc + 1
        small = set()

        def offer(word):
            nonlocal distance
            if not word:
                return
            weight = word.bit_count()
            distance = min(distance, weight)
            if weight <= 16:
                small.add(word)

        for trial in range(trials):
            srng.shuffle(order)
            pivots = {}
            basis = []
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

            for word in sorted(basis, key=int.bit_count)[:6]:
                for step in range(3):
                    best, best_weight = word, word.bit_count()
                    for other in basis:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < best_weight:
                            best, best_weight = candidate, candidate.bit_count()
                    if best == word:
                        break
                    word = best
                    offer(word)
            if distance <= 6:
                break

        # Quotient only the verified cyclic translation action on words.
        # This reduces uneven sampling of the same short-word orbit.
        mask = (1 << q) - 1
        for j in range(nc // q):
            seed = cols[j * q]
            for shift in range(q):
                translated = 0
                for r in range(2):
                    block = (seed >> (r * q)) & mask
                    translated |= (
                        ((block << shift) | (block >> (q - shift))) & mask
                    ) << (r * q)
                assert cols[j * q + shift] == translated

        orbits = set()
        for word in small:
            blocks = [(word >> (j * q)) & mask for j in range(nc // q)]
            images = []
            for shift in range(q):
                images.append(sum(
                    (((block << shift) | (block >> (q - shift))) & mask)
                    << (j * q) for j, block in enumerate(blocks)))
            orbits.add(min(images))

        loads = [0] * (nc // q)
        burden = 0
        for word in orbits:
            cost = (18 - word.bit_count()) ** 2
            burden += cost
            for j in range(len(loads)):
                loads[j] += cost * ((word >> (j * q)) & mask).bit_count()
        return distance, burden, max(loads)

    def design_component(supports, side):
        positions = [(i, j) for i, row in enumerate(supports)
                     for j, w in enumerate(row) if w]
        locals_found = []
        for restart in range(6):
            grid = draw_grid(side)
            rows, nc = component(grid, side)
            current = cycle_energy(rows, nc)
            local = copy_grid(grid)
            local_stats = current

            for step in range(260):
                i, j = rng.choice(positions)
                old = list(grid[i][j])
                # Change the whole-cell origin, preserving its gap.
                grid[i][j] = draw_cell(side, i, j)
                trial_rows, _ = component(grid, side)
                proposed = cycle_energy(trial_rows, nc)
                delta = proposed[0] - current[0]
                temperature = 12.0 * (0.985 ** step) + 0.08
                if (delta <= 0 or rng.random() <
                        math.exp(-min(700.0, delta / temperature))):
                    current = proposed
                    if (current[0] < local_stats[0] or
                            (current[0] == local_stats[0]
                             and rng.random() < 0.06)):
                        local = copy_grid(grid)
                        local_stats = current
                else:
                    grid[i][j] = old

            rows, nc = component(local, side)
            # Zero augmentation forces two missing ranks. Since
            # ord_29(2)=28, reject any additional character collapse.
            if gf2_rank(rows) != 56:
                continue
            locals_found.append((local_stats, restart, local, rows, nc))

        locals_found.sort(key=lambda item: (item[0], item[1]))
        finalists = []
        for stats, restart, grid, rows, nc in locals_found[:4]:
            cols = columns(rows, nc)
            assert all(col.bit_count() == 4 for col in cols)
            bad, risk, peak = boundary_scan(cols)
            distance, low, word_peak = kernel_exposure(cols, 48, side)
            key = (bad, max(0, 16 - distance),
                   low, word_peak, peak, risk, stats[1], stats[2])
            finalists.append({
                "grid": grid, "key": key, "d": distance,
                "low": low, "word_peak": word_peak,
                "bad": bad, "risk": risk,
                "peak": peak, "four": stats[1], "six": stats[2],
            })
        finalists.sort(key=lambda item: item["key"])
        if not finalists:
            raise ValueError("No rank-safe component filling")
        return finalists

    apool = design_component(sa, "A")
    bpool = design_component(sb, "B")
    winner = None
    winning_key = None
    for a in apool:
        for b in bpool:
            A, B = a["grid"], b["grid"]
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            # The augmentation prediction is never substituted for the
            # evaluator-convention ranks of the complete CSS matrices.
            if n != 580 or k != 132:
                continue
            da, db = max(0, 16 - a["d"]), max(0, 16 - b["d"])
            key = (
                max(a["bad"], b["bad"]),
                max(da, db), da + db,
                max(a["low"], b["low"]),
                a["low"] + b["low"],
                max(a["word_peak"], b["word_peak"]),
                a["word_peak"] + b["word_peak"],
                max(a["peak"], b["peak"]),
                max(a["risk"], b["risk"]),
                a["risk"] + b["risk"],
                max(a["four"], b["four"]),
                a["four"] + b["four"],
                a["six"] + b["six"],
            )
            if winning_key is None or key < winning_key:
                winning_key = key
                winner = (A, B, hx, hz)

    if winner is None:
        raise ValueError("No finalist retained verified k=132 in cell 0.250")
    A, B, hx, hz = winner
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) == 12
    assert max(row.bit_count() for row in hx + hz) <= W_MAX
    assert sum(row.bit_count() for row in hx + hz) * 6 == 33408
    assert 40 * 132 > 9 * 580 and 40 * 132 < 11 * 580

    # Check cached helper-derived actions against the final complete lift.
    actual_a = []
    for i in range(2):
        for t in range(q):
            row = hx[(i * 4) * q + t]
            actual_a.append(sum(
                ((row >> (j * 4 * q)) & block_mask) << (j * q)
                for j in range(4)))
    actual_b = [
        row & ((1 << (4 * q)) - 1) for row in hz[:2 * q]
    ]
    assert component(A, "A")[0] == actual_a
    assert component(B, "B")[0] == actual_b
    assert all(len(cell) == 2 and
               len({G._key(g) for g in cell}) == 2
               for grid in (A, B) for row in grid for cell in row)
    return {"idx": idx, "architecture": contract, "k": 132,
            "screen": winning_key, "cnots": 33408,
            "cells": {"A": A, "B": B}}


def NEW_CTX(rng, G, idx):
    """Unit/long-ratio contracts with verified ranks and joint holdouts."""
    import math
    import random as local_random

    assert G.q == 40
    assert A_SHAPE == (1, 2) and B_SHAPE == (2, 6)
    q = G.q
    budget, policy = divmod(idx % 20, 5)
    batch = idx // 20
    weak_number = 3 - budget
    strong_number = 6 - weak_number

    sa = [[3, 3]]
    bcolumns = [(1, 1)] * weak_number
    bcolumns += ([(2, 1)] * (strong_number // 2)
                 + [(1, 2)] * (strong_number - strong_number // 2))

    # Additional batches change count/incidence architecture, not just
    # the origin of a previously selected voltage configuration.
    if batch % 3 == 1:
        upper = next(j for j, c in enumerate(bcolumns) if c == (2, 1))
        lower = next(j for j, c in enumerate(bcolumns) if c == (1, 2))
        bcolumns[upper], bcolumns[lower] = (3, 0), (0, 3)
    elif batch % 3 == 2:
        if weak_number >= 2:
            bcolumns[0], bcolumns[1] = (2, 0), (0, 2)
        elif weak_number == 1:
            bcolumns[0] = (2, 0)
        else:
            upper = [j for j, c in enumerate(bcolumns) if c == (2, 1)]
            lower = [j for j, c in enumerate(bcolumns) if c == (1, 2)]
            for j in upper[:2]:
                bcolumns[j] = (3, 0)
            for j in lower[:2]:
                bcolumns[j] = (0, 3)

    sb = [[c[i] for c in bcolumns] for i in range(2)]
    supports = {"A": sa, "B": sb}
    assert sum(map(sum, sb)) == 15 + budget
    assert all(sum(c) in (2, 3) for c in bcolumns)
    assert _no_dead_lines(sa, sb)
    wx = sum(sa[0]) + max(map(sum, bcolumns))
    wz = max(map(sum, sb)) + max(sa[0])
    assert max(wx, wz) == _grid_weights(sa, sb)
    assert max(wx, wz) <= min(12, W_MAX)

    expected_cnots = 6*q*(8*sum(sa[0]) + 3*sum(map(sum, sb)))
    assert expected_cnots in (22320, 23040, 23760, 24480)

    orders = {}
    for g in G.elements:
        value = G.e
        for t in range(1, q + 1):
            value = G.mul(value, g)
            if value == G.e:
                orders[G.idx(g)] = t
                break
        else:
            raise ValueError("validated group has no element order")
    by_order = {
        t: [g for g in G.elements if orders[G.idx(g)] == t]
        for t in (4, 5, 8, 10, 20)
    }
    assert all(by_order.values())

    def power(g, t):
        out = G.e
        for _ in range(t):
            out = G.mul(out, g)
        return out

    def relative_order(g, h):
        return orders[G.idx(G.mul(G.inv(g), h))]

    # A1 contracts have genuinely different relative-order spectra.
    partner_books = {}
    partner_books[0] = [(h, power(h, 3)) for h in by_order[8]]
    partner_books[1] = [
        (h, z) for h in by_order[8] for z in by_order[8]
        if relative_order(h, z) == 20
    ]
    for mode, terminal in ((2, 5), (3, 10), (4, 20)):
        partner_books[mode] = [
            (h, z) for h in by_order[8] for z in by_order[terminal]
            if relative_order(h, z) == 8
        ]
    assert all(partner_books.values())
    palettes = ((20,), (20, 8), (20, 10),
                (20, 5), (20, 10, 8))

    def unit_triangle():
        h = rng.choice(by_order[8])
        # Three distinct powers in a finite two-group algebra: a unit.
        return [G.e, h, power(h, 3)]

    def draw_A():
        first = unit_triangle()
        for _ in range(128):
            h, z = rng.choice(partner_books[policy])
            second = [G.e, h, z]
            if any(G.mul(a, b) != G.mul(b, a)
                   for a in first[1:] for b in second[1:]):
                return [[first, second]]
        raise ValueError("A contract failed to couple noncommuting directions")

    def draw_B():
        grid = [[[] for _ in range(6)] for _ in range(2)]
        matching = [j for j, c in enumerate(bcolumns) if c == (1, 1)]
        holonomies = [G.e]
        if len(matching) >= 2:
            holonomies.append(rng.choice(by_order[20]))
        if len(matching) >= 3:
            # This does not commute with an order-20 element having
            # nonzero x coordinate. All pair differences have long order.
            holonomies.append(rng.choice(by_order[8]))
        for j, h in zip(matching, holonomies):
            grid[0][j], grid[1][j] = [G.e], [h]

        for j, column in enumerate(bcolumns):
            if column == (1, 1):
                continue
            for i, count in enumerate(column):
                if count == 0:
                    continue
                base = G.pick(rng)
                if count == 1:
                    cell = [base]
                elif count == 2:
                    internal = column[1-i] == 0
                    palette = (20,) if internal else palettes[policy]
                    order = palette[(j + 2*i + batch) % len(palette)]
                    candidates = by_order[order]
                    if order in (5, 10, 20):
                        direction = 1 + ((j + i*j + policy + batch) % 2)
                        candidates = [
                            g for g in candidates
                            if min(g[0], 5-g[0]) == direction
                        ]
                    delta = rng.choice(candidates)
                    cell = [base, G.mul(base, delta)]
                else:
                    relative = unit_triangle()
                    cell = [G.mul(base, g) for g in relative]
                assert len(cell) == count
                assert len({G._key(g) for g in cell}) == count
                grid[i][j] = cell
        return grid

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    # Cache ONLY singleton actions returned by the verified helper.
    mask = (1 << q) - 1
    actions = {"A": {}, "B": {}}
    for g in G.elements:
        x, _, _ = lifted_matrices(G, [[[g]]], [[[]]])
        _, z, _ = lifted_matrices(G, [[[]]], [[[g]]])
        actions["A"][G.idx(g)] = tuple(row & mask for row in x)
        actions["B"][G.idx(g)] = tuple(row & mask for row in z)

    def component(grid, side):
        out = []
        for row in grid:
            block_rows = [0] * q
            for j, cell in enumerate(row):
                for g in cell:
                    term = actions[side][G.idx(g)]
                    for t, bits in enumerate(term):
                        block_rows[t] ^= bits << (j*q)
            out.extend(block_rows)
        return out

    def columns(rows, width):
        out = [0] * width
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                out[bit.bit_length()-1] |= tag
                row ^= bit
        return out

    def graph_stats(rows):
        """Exact Tanner C4/C6 and check-graph connectivity."""
        size = len(rows)
        adjacency = [0] * size
        overlap = [dict() for _ in rows]
        four = 0
        for i, row in enumerate(rows):
            for j in range(i):
                common = (row & rows[j]).bit_count()
                four += common*(common-1)//2
                if common:
                    adjacency[i] |= 1 << j
                    adjacency[j] |= 1 << i
                    overlap[i][j] = overlap[j][i] = common
        six = 0
        for i, neighbors in enumerate(adjacency):
            lower = neighbors & ((1 << i)-1)
            while lower:
                bit = lower & -lower
                j = bit.bit_length()-1
                common = neighbors & adjacency[j] & ((1 << j)-1)
                shared = rows[i] & rows[j]
                while common:
                    third = common & -common
                    h = third.bit_length()-1
                    a, b, c = overlap[i][j], overlap[i][h], overlap[j][h]
                    reused = (shared & rows[h]).bit_count()
                    six += a*b*c - reused*(a+b+c) + 2*reused
                    common ^= third
                lower ^= bit
        assert six >= 0
        unseen = (1 << size)-1
        parts = 0
        while unseen:
            todo = unseen & -unseen
            unseen ^= todo
            parts += 1
            while todo:
                bit = todo & -todo
                todo ^= bit
                fresh = adjacency[bit.bit_length()-1] & unseen
                unseen ^= fresh
                todo |= fresh
        return four, six, parts

    def kernel_small(cols, through_six=False):
        """Exact existence test through four, optionally through six."""
        seen = {0}
        for s in cols:
            if s in seen:
                return True
            seen.add(s)
        for i, s in enumerate(cols):
            for j in range(i):
                value = s ^ cols[j]
                if value in seen:
                    return True
                seen.add(value)
        if through_six:
            for i, s in enumerate(cols):
                for j in range(i):
                    pair = s ^ cols[j]
                    for h in range(j):
                        value = pair ^ cols[h]
                        if value in seen:
                            return True
                        seen.add(value)
        return False

    def sampled_exposure(cols, side):
        """Verified sampled kernel words; minima remain upper bounds."""
        srng = local_random.Random(40560271 + 1009*idx + (side == "B"))
        order = list(range(len(cols)))
        minimum = len(cols)+1
        words = set()

        def record(word):
            nonlocal minimum
            if not word:
                return
            weight = word.bit_count()
            minimum = min(minimum, weight)
            if weight <= 12 and len(words) < 4096:
                words.add(word)

        for trial in range(40 if side == "A" else 28):
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
                    record(word)
            for sweep in range(2):
                srng.shuffle(basis)
                for i, word in enumerate(basis):
                    weight = word.bit_count()
                    for other in basis[:i]:
                        candidate = word ^ other
                        record(candidate)
                        w = candidate.bit_count()
                        if candidate and w < weight:
                            word, weight = candidate, w
                    basis[i] = word
            if minimum <= 4:
                break

        loads = [0] * (len(cols)//q)
        burden = 0
        for word in words:
            cost = (14-word.bit_count())**2
            burden += cost
            for j in range(len(loads)):
                loads[j] += cost*((word >> (j*q)) & mask).bit_count()
        return minimum, burden, max(loads, default=0)

    def boundary(cols):
        """Orbit-anchored triple boundaries, not a trapping-set census."""
        loads = []
        for anchor in range(0, len(cols), q):
            cost = 0
            for i, s in enumerate(cols):
                if i == anchor:
                    continue
                pair = cols[anchor] ^ s
                for j in range(i):
                    if j == anchor:
                        continue
                    weight = (pair ^ cols[j]).bit_count()
                    if weight <= 4:
                        cost += (6-weight)**2
            loads.append(cost)
        return max(loads), sum(loads)/len(cols)

    def portfolio(side):
        draw = draw_A if side == "A" else draw_B
        width = (2 if side == "A" else 6)*q
        pool, seen = [], set()

        def assess(grid):
            rows = component(grid, side)
            four, six, parts = graph_stats(rows)
            energy = (60*four + six/8.0)/q + 40*(parts-1)
            return energy, rows, four, six, parts

        def retain(grid, state):
            _, rows, four, six, parts = state
            if gf2_rank(rows) != len(rows):
                return
            identity = tuple(
                tuple(sorted(G.idx(g) for g in cell))
                for row in grid for cell in row)
            if identity in seen:
                return
            seen.add(identity)
            pool.append(((parts-1, four, six), identity,
                         clone(grid), list(rows)))
            pool.sort(key=lambda entry: entry[:2])
            del pool[6:]

        for restart in range(5):
            grid = draw()
            state = assess(grid)
            retain(grid, state)
            for step in range(170):
                proposal = clone(grid)
                fresh = draw()
                if side == "A":
                    j = rng.randrange(2)
                    proposal[0][j] = fresh[0][j]
                else:
                    j = rng.randrange(6)
                    proposal[0][j], proposal[1][j] = fresh[0][j], fresh[1][j]
                proposed = assess(proposal)
                difference = proposed[0]-state[0]
                temperature = 12.0*(0.977**step) + 0.10
                if difference <= 0 or rng.random() < math.exp(
                        -min(700.0, difference/temperature)):
                    grid, state = proposal, proposed
                    if step % 16 == 0:
                        retain(grid, state)
            retain(grid, state)

        if not pool:
            for trial in range(192):
                grid = draw()
                retain(grid, assess(grid))
                if len(pool) >= 3:
                    break
        if not pool:
            raise ValueError("no full-rank component in this contract")

        finalists = []
        for coarse, identity, grid, rows in pool:
            probe = (lifted_matrices(G, grid, [[[]]])[0]
                     if side == "A"
                     else lifted_matrices(G, [[[]]], grid)[1])
            assert rows == [r & ((1 << width)-1) for r in probe]
            cols = columns(rows, width)
            bad = int(kernel_small(cols))
            weak = [c for c in cols if c.bit_count() == 2]
            weak_bad = int(kernel_small(weak, True)) if weak else 0
            d, burden, peak_words = sampled_exposure(cols, side)
            peak, mean = boundary(cols)
            key = (bad, weak_bad, coarse[0], max(0, 10-d),
                   peak, mean, burden, peak_words, coarse[1], coarse[2])
            finalists.append(dict(
                key=key, grid=grid, rows=rows, d=d, bad=bad,
                weak_bad=weak_bad, parts=coarse[0], burden=burden,
                word_peak=peak_words, peak=peak, boundary=mean,
                four=coarse[1], six=coarse[2]))
        finalists.sort(key=lambda entry: entry["key"])
        return finalists[:4]

    def pivots_of(rows):
        pivots = {}
        for word in rows:
            while word:
                p = word.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = word
                    break
                word ^= old
        return pivots

    def nontrivial(word, pivots):
        while word:
            old = pivots.get(word.bit_length()-1)
            if old is None:
                return True
            word ^= old
        return False

    def small_logical(checks, stabilizers, n):
        """Exact quantum logical existence through weight four."""
        cols = columns(checks, n)
        pivots = pivots_of(stabilizers)
        seen = {0: 0}
        for i, s in enumerate(cols):
            word = 1 << i
            previous = seen.get(s)
            if previous is None:
                seen[s] = word
            elif nontrivial(previous ^ word, pivots):
                return True
        for i, s in enumerate(cols):
            for j in range(i):
                syndrome = s ^ cols[j]
                word = (1 << i) | (1 << j)
                previous = seen.get(syndrome)
                if previous is None:
                    seen[syndrome] = word
                elif nontrivial(previous ^ word, pivots):
                    return True
        return False

    apool, bpool = portfolio("A"), portfolio("B")
    prepared = []
    for a in apool:
        for b in bpool:
            A, B = a["grid"], b["grid"]
            hx, hz, n = lifted_matrices(G, A, B)
            k = n-gf2_rank(hx)-gf2_rank(hz)
            if n != 560 or k != 160:
                continue
            assert max(r.bit_count() for r in hx+hz) <= min(12, W_MAX)
            assert 6*sum(r.bit_count() for r in hx+hz) == expected_cnots
            actual_a = [
                sum(((hx[t] >> (j*6*q)) & mask) << (j*q)
                    for j in range(2)) for t in range(q)]
            actual_b = [r & ((1 << (6*q))-1) for r in hz[:2*q]]
            assert actual_a == a["rows"] and actual_b == b["rows"]
            key = (
                a["bad"]+b["bad"], a["weak_bad"]+b["weak_bad"],
                a["parts"]+b["parts"],
                max(0, 10-a["d"])+max(0, 10-b["d"]),
                max(a["peak"], b["peak"]),
                a["boundary"]+b["boundary"],
                max(a["burden"], b["burden"]),
                a["burden"]+b["burden"],
                max(a["word_peak"], b["word_peak"]),
                a["four"]+b["four"], a["six"]+b["six"])
            prepared.append((key, A, B, hx, hz))
    prepared.sort(key=lambda entry: entry[0])
    if not prepared:
        raise ValueError("no helper-built k160 pair in cell 0.300")

    safe = []
    for entry in prepared[:8]:
        _, A, B, hx, hz = entry
        if small_logical(hx, hz, 560) or small_logical(hz, hx, 560):
            continue
        safe.append(entry)
        if len(safe) == 4:
            break
    if not safe:
        raise ValueError("all joint finalists had quantum logicals through four")

    def temporal_probe(checks, stabilizers, seed, shots):
        """Two noisy intervals and one noisy intermediate syndrome.

        This is NOT ASC, a six-round history, or relay-BP/LSD. Spread
        pairs are approximate opportunities, not scheduled hook faults.
        Settled residuals are classified modulo actual stabilizers.
        """
        try:
            import numpy as np
        except ImportError:
            return (0, 0, 0)

        n, m = 560, len(checks)
        width = 2*n+m
        augmented = [
            row | (1 << (2*n+r)) for r, row in enumerate(checks)
        ] + [row | (row << n) for row in checks]
        nr = len(augmented)
        ev, ce_lists = [], []
        ve_lists = [[] for _ in range(width)]
        for row in augmented:
            edges = []
            while row:
                bit = row & -row
                c = bit.bit_length()-1
                edge = len(ev)
                ev.append(c)
                edges.append(edge)
                ve_lists[c].append(edge)
                row ^= bit
            ce_lists.append(edges)
        ne = len(ev)
        ce = np.full((nr, max(map(len, ce_lists))), ne, dtype=np.int32)
        ve = np.full((width, max(map(len, ve_lists))), ne, dtype=np.int32)
        for r, edges in enumerate(ce_lists):
            ce[r, :len(edges)] = edges
        for c, edges in enumerate(ve_lists):
            ve[c, :len(edges)] = edges
        real = ce != ne
        ev = np.asarray(ev, dtype=np.int32)
        sensed = np.asarray(list(ev)+[width], dtype=np.int32)[ce]

        channel = np.random.default_rng(seed)
        intensity = np.asarray((0.35, 0.60, 0.85))[np.arange(shots) % 3]
        cx, cz = columns(checks, n), columns(stabilizers, n)
        degree = np.asarray([
            x.bit_count()+z.bit_count() for x, z in zip(cx, cz)], dtype=float)
        data_p = np.clip(
            0.5*(1-np.exp(-0.008*intensity[:, None]*degree)), 0.003, 0.18)
        measurement_p = np.clip(
            0.5*(1-np.exp(-0.008*intensity[:, None]*np.asarray([
                row.bit_count()+2 for row in checks]))), 0.003, 0.18)
        probabilities = np.concatenate(
            (data_p, data_p, measurement_p), axis=1)
        errors = np.zeros((shots, width+1), dtype=bool)
        errors[:, :width] = channel.random((shots, width)) < probabilities

        opposite = []
        for row in stabilizers:
            sites = []
            while row:
                bit = row & -row
                sites.append(bit.bit_length()-1)
                row ^= bit
            opposite.append(sites)
        for interval in range(2):
            for shot in range(shots):
                if channel.random() < 0.25*intensity[shot]:
                    sites = opposite[int(channel.integers(len(opposite)))]
                    if len(sites) >= 2:
                        pair = channel.choice(sites, 2, replace=False)
                        errors[shot, interval*n+pair] ^= True

        target = np.logical_xor.reduce(errors[:, sensed], axis=2)
        signs = np.where(target, -1.0, 1.0)
        prior = np.log((1-probabilities)/probabilities)
        incoming = np.empty((shots, ne+1), dtype=float)
        incoming[:, :ne] = prior[:, ev]
        incoming[:, ne] = 30.0
        outgoing = np.zeros_like(incoming)
        active = np.ones(shots, dtype=bool)
        pivots = pivots_of(stabilizers)
        failures = work = 0
        ones = np.ones((shots, nr, 1), dtype=float)

        for iteration in range(40):
            work += int(active.sum())
            values = np.tanh(np.clip(incoming[:, ce], -18, 18)/2)
            values[:, ~real] = 1.0
            left = np.concatenate(
                (ones, np.cumprod(values, axis=2)[:, :, :-1]), axis=2)
            suffix = np.cumprod(values[:, :, ::-1], axis=2)[:, :, ::-1]
            right = np.concatenate((suffix[:, :, 1:], ones), axis=2)
            update = 2*np.arctanh(np.clip(
                signs[:, :, None]*left*right, -0.999999, 0.999999))
            outgoing[:, ce[real]] = update[:, real]
            outgoing[:, ne] = 0.0
            posterior = prior + outgoing[:, ve].sum(axis=2)
            hard = np.concatenate(
                (posterior < 0, np.zeros((shots, 1), dtype=bool)), axis=1)
            observed = np.logical_xor.reduce(hard[:, sensed], axis=2)
            settled = active & np.all(observed == target, axis=1)
            for shot in np.flatnonzero(settled):
                residual = (errors[shot, :n] ^ errors[shot, n:2*n]
                            ^ hard[shot, :n] ^ hard[shot, n:2*n])
                word = sum(1 << int(c) for c in np.flatnonzero(residual))
                failures += int(nontrivial(word, pivots))
            active[settled] = False
            if not active.any():
                break
            damping = 0.25 if iteration < 12 else 0.50
            proposal = posterior[:, ev]-outgoing[:, :ne]
            incoming[:, :ne] = (
                damping*incoming[:, :ne]
                + (1-damping)*np.clip(proposal, -24, 24))
        unresolved = int(active.sum())
        return failures+unresolved, unresolved, work

    def probe_key(x, z):
        return (max(x[0], z[0]), x[0]+z[0],
                max(x[1], z[1]), x[1]+z[1],
                max(x[2], z[2]), x[2]+z[2])

    # Identical streams compare finalists; the holdout is independently
    # seeded. No result is interpreted as a calibrated circuit rate.
    seed = 40560317+1009*idx
    coarse = []
    for number, entry in enumerate(safe):
        key, A, B, hx, hz = entry
        x = temporal_probe(hz, hx, seed, 48)
        z = temporal_probe(hx, hz, seed+1, 48)
        quality = key[:4]+probe_key(x, z)+key[4:]+(number,)
        coarse.append((quality, number, entry, x, z))
    coarse.sort(key=lambda entry: entry[:2])

    best = None
    for _, number, entry, cx, cz in coarse[:2]:
        key, A, B, hx, hz = entry
        x = temporal_probe(hz, hx, seed+1000003, 96)
        z = temporal_probe(hx, hz, seed+1000004, 96)
        tx = tuple(a+b for a, b in zip(cx, x))
        tz = tuple(a+b for a, b in zip(cz, z))
        quality = key[:4]+probe_key(tx, tz)+key[4:]+(number,)
        if best is None or quality < best[0]:
            best = quality, A, B, hx, hz, tx, tz

    quality, A, B, hx, hz, tx, tz = best
    assert css_ok(hx, hz)
    assert lifted_k(G, A, B) == 160
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert max(r.bit_count() for r in hx+hz) <= min(12, W_MAX)
    assert 6*sum(r.bit_count() for r in hx+hz) == expected_cnots
    assert all(
        0 <= len(cell) <= CELL_MAX
        and len({G._key(g) for g in cell}) == len(cell)
        for grid in (A, B) for row in grid for cell in row)
    assert 0.275 <= 160/560 < 0.325
    return {
        "idx": idx, "contract": (budget, policy, batch),
        "k": 160, "cnots": expected_cnots, "screen": quality,
        "two_interval_surrogate": {"X": tx, "Z": tz},
        "cells": {"A": A, "B": B},
    }


def CELL(rng, G, side, i, j, ctx):
    """Emit the complete selected cell without redraws or relabeling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal rolled supports and uniform labels suffice for basis-balanced memory on this bed: parent slot 4 failed 102/1500 X shots versus 1/200000 Z shots, with widespread weight-six witnesses.
# REFUTED: First-stage syndrome settlement certifies logical success: parent slot 12 settled 749/750 X shots at stage 1 yet failed 109/750 scored shots.
# REFUTED: Equal k and d2s imply equal circuit reliability: current slots 1 and 6 both had k=98 and d2s=12, but X failed 11 versus 24 times in 200000 shots and scored 6.776 versus 6.518.
# REFUTED: Increasing d2s guarantees better memory on the parent C17 bed: slot53 had d2s=14, eight X failures/200000 and s=6.806, while slot10 had d2s=13, one failure per basis and s=7.258.
# REFUTED: Sum-free regular C29 directions guarantee ceiling-level memory: all five measured codes had nonzero failures; slots4/16 both had k132,w12,d2s14,depth16,CNOTs33408 but two/nine Z failures per 200000 and s7.341/6.967.
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
