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
CONCEPT = ("For (C5 semidirect C4, action 2) x C2, order 40, shapes (1,2)x(2,6), canonical noncommuting A trinomials and balanced B components with lower extraction exposure should improve worse-basis memory in frontier cell 0.300; circuit measurements can refute this.")
GROUP = ("Jump from C29 to the nonabelian direct product (C5:C4) x C2. The action 2 has order four modulo five; the last factor is central. Relative orders 4,5,10 permit noncommuting triangle geometries without relative involutions. Related order-40 beds reached approximately 5.9-6.1, but their reported centre/class invariants do not identify this presentation uniquely or establish its performance.")
SHAPES = ("Use (1,2)x(2,6) at q40: n560, kb160 and guaranteed rate 2/7=0.285714 in cell0.300. A has 80 variables/40 checks; B has 240 variables/80 checks. Require helper-built component ranks40/80 and complete k160. Parent failures implicated different Acol/Bcol blocks, including Bcol1/2, with mostly weak Acol/Brow check enrichment; these small samples do not justify repairing one persistent hotspot.")
FILLING = ("Twenty contracts combine five canonical noncommuting A geometries with four B count architectures; additional slots consume additional canonical A pairs. A cells are trinomials. B rows total8/9 or9/9 and columns total2..4, so HX<=10 and HZ<=12. Six-round CNOT counts are23760/24480 instead of33408. Canonicalization removes explicit A column gauges, group automorphisms and column exchange, not arbitrary Tanner equivalence. Verified singleton actions screen ranks, connectivity, cycles, A kernel words through six, B words through four, sampled ambiguity and anchored boundaries. Complete CSS matrices prefer no logical through four and compare fixed-bank two-interval BP stress tests with noisy intermediate syndromes and opposite-check pair faults. This is not ASC scheduling, six-round memory, relay-BP/LSD or calibrated pL. Target fewer collective failures and decoder fall-throughs, not a larger distance upper bound.")
FAMILY_NAME = "r030_F20xC2_q40_12x26_canonical_triples_twotime_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 40

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("(C5:C4) x C2 requires order 40")
    # x^5=y^4=z^2=1, y*x*y^-1=x^2, z central.
    # Elements are x^a*y^b*z^c.
    els = [(a, b, c) for b in range(4)
           for c in range(2) for a in range(5)]
    action = (1, 2, 4, 3)

    def red(g):
        return (int(g[0]) % 5, int(g[1]) % 4, int(g[2]) % 2)

    def mul(g, h):
        a, b, c = red(g)
        d, e, f = red(h)
        return ((a + action[b]*d) % 5, (b+e) % 4, (c+f) % 2)

    def inv(g):
        a, b, c = red(g)
        bi = (-b) % 4
        return ((-action[bi]*a) % 5, bi, (-c) % 2)

    def idx(g):
        a, b, c = red(g)
        return 10*b + 5*c + a

    return dict(elements=els, ident=(0, 0, 0),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (2, 6)               # (rb, cb)


def _retired_C29_CTX(rng, G, idx):
    """Historical C29 search; never called on the new group/shape bed."""
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
    """Canonical A contracts, optimized B fillings and complete-CSS probes."""
    import math
    from itertools import combinations

    q = G.q
    assert q == 40 and A_SHAPE == (1, 2) and B_SHAPE == (2, 6)
    sa = [[3, 3]]
    profiles = (
        ((1, 1, 1, 2, 2, 2), (2, 2, 2, 1, 1, 1)),
        ((0, 1, 1, 2, 2, 3), (3, 2, 2, 1, 1, 0)),
        ((1, 1, 1, 2, 2, 2), (1, 2, 2, 1, 2, 1)),
        ((1, 1, 1, 2, 2, 2), (1, 2, 2, 1, 1, 1)),
    )
    acard, architecture = divmod(idx, 4)
    sb = [list(row) for row in profiles[architecture]]
    wx = sum(sa[0]) + max(map(sum, zip(*sb)))
    wz = max(map(sum, sb)) + 3
    assert max(wx, wz) == _grid_weights(sa, sb)
    assert wx <= min(12, W_MAX) and wz <= min(12, W_MAX)
    assert _no_dead_lines(sa, sb)
    assert all(0 <= t <= CELL_MAX
               for grid in (sa, sb) for row in grid for t in row)

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    # Cache ONLY actual singleton actions from the verified helper.
    mask = (1 << q) - 1
    actions = {"A": {}, "B": {}}
    for g in G.elements:
        hx, _, _ = lifted_matrices(G, [[[g]]], [[[]]])
        _, hz, _ = lifted_matrices(G, [[[]]], [[[g]]])
        actions["A"][G.idx(g)] = tuple(row & mask for row in hx)
        actions["B"][G.idx(g)] = tuple(row & mask for row in hz)

    def component(grid, side):
        rows = []
        for row in grid:
            out = [0] * q
            for j, cell in enumerate(row):
                for g in cell:
                    block = actions[side][G.idx(g)]
                    for t, bits in enumerate(block):
                        out[t] ^= bits << (j*q)
            rows.extend(out)
        return rows

    def columns(rows, width):
        out = [0] * width
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                out[bit.bit_length()-1] |= 1 << r
                row ^= bit
        return out

    def graph(rows):
        """Exact Tanner C4/C6 and check-graph connectivity."""
        adj = [0] * len(rows)
        overlaps = [dict() for _ in rows]
        four = 0
        for i, row in enumerate(rows):
            for j in range(i):
                v = (row & rows[j]).bit_count()
                four += v*(v-1)//2
                if v:
                    adj[i] |= 1 << j
                    adj[j] |= 1 << i
                    overlaps[i][j] = overlaps[j][i] = v
        six = 0
        for i, neighbors in enumerate(adj):
            lower = neighbors & ((1 << i)-1)
            while lower:
                bit = lower & -lower
                j = bit.bit_length()-1
                common = neighbors & adj[j] & ((1 << j)-1)
                ab = overlaps[i][j]
                shared = rows[i] & rows[j]
                while common:
                    third = common & -common
                    h = third.bit_length()-1
                    ac, bc = overlaps[i][h], overlaps[j][h]
                    reused = (shared & rows[h]).bit_count()
                    six += ab*ac*bc - reused*(ab+ac+bc) + 2*reused
                    common ^= third
                lower ^= bit
        assert six >= 0
        unseen, parts = (1 << len(rows))-1, 0
        while unseen:
            todo = unseen & -unseen
            unseen ^= todo
            parts += 1
            while todo:
                bit = todo & -todo
                todo ^= bit
                fresh = adj[bit.bit_length()-1] & unseen
                unseen ^= fresh
                todo |= fresh
        return four, six, parts

    def short_kernel(cols, through_six):
        """Exact component existence test through four or six."""
        seen = {0}
        for syndrome in cols:
            if syndrome in seen:
                return True
            seen.add(syndrome)
        for i, syndrome in enumerate(cols):
            for j in range(i):
                value = syndrome ^ cols[j]
                if value in seen:
                    return True
                seen.add(value)
        if through_six:
            for i, syndrome in enumerate(cols):
                for j in range(i):
                    pair = syndrome ^ cols[j]
                    for h in range(j):
                        value = pair ^ cols[h]
                        if value in seen:
                            return True
                        seen.add(value)
        return False

    def ambiguity(cols, side, trials):
        """Actual sampled kernel words; minima remain upper bounds."""
        srng = random.Random(400731 + 1009*idx + (side == "B"))
        order = list(range(len(cols)))
        minimum, short = len(cols)+1, set()

        def record(word):
            nonlocal minimum
            if not word:
                return
            weight = word.bit_count()
            minimum = min(minimum, weight)
            if weight <= 12 and len(short) < 4096:
                short.add(word)

        for trial in range(trials):
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
            if minimum <= 6:
                break
        burden, loads = 0, [0] * (len(cols)//q)
        for word in short:
            cost = (14-word.bit_count())**2
            burden += cost
            for j in range(len(loads)):
                loads[j] += cost*((word >> (j*q)) & mask).bit_count()
        return minimum, burden, max(loads, default=0)

    def boundary(cols):
        """Orbit-anchored exposure, not an exact trapping-set spectrum."""
        loads = []
        for anchor in range(0, len(cols), q):
            cost = 0
            for i, syndrome in enumerate(cols):
                if i == anchor:
                    continue
                pair = syndrome ^ cols[anchor]
                for j in range(i):
                    if j == anchor:
                        continue
                    weight = (pair ^ cols[j]).bit_count()
                    if weight <= 4:
                        cost += (6-weight)**2
            loads.append(cost)
        return max(loads), sum(loads)

    def refine(grid, side, rows, coarse):
        width = len(grid[0])*q
        # Check cached composition against a complete verified zero probe.
        if side == "A":
            verified = lifted_matrices(G, grid, [[[]]])[0]
        else:
            verified = lifted_matrices(G, [[[]]], grid)[1]
        assert rows == [row & ((1 << width)-1) for row in verified]
        cols = columns(rows, width)
        bad = int(short_kernel(cols, side == "A"))
        d, burden, peak = ambiguity(cols, side, 32 if side == "A" else 24)
        bp, bt = boundary(cols)
        key = (bad, coarse[0], max(0, 10-d),
               burden, peak, bp, bt, coarse[1], coarse[2])
        return dict(grid=grid, rows=rows, key=key, bad=bad,
                    parts=coarse[0], d=d, burden=burden,
                    peak=peak, boundary_peak=bp, boundary_total=bt,
                    four=coarse[1], six=coarse[2])

    orders = {}
    for g in G.elements:
        power = G.e
        for t in range(1, q+1):
            power = G.mul(power, g)
            if power == G.e:
                orders[G.idx(g)] = t
                break
        assert G.idx(g) in orders
    palettes = {
        t: [g for g in G.elements if orders[G.idx(g)] == t]
        for t in (4, 5, 10)
    }
    assert all(palettes.values())

    def normalize(cell):
        # Right multiplication of an A cell is a column gauge.
        return min(tuple(sorted(G.mul(g, G.inv(origin)) for g in cell))
                   for origin in cell)

    triples = set()
    nonidentity = [g for g in G.elements if g != G.e]
    for h, z in combinations(nonidentity, 2):
        if G.mul(h, z) == G.mul(z, h):
            continue
        cell = (G.e, h, z)
        if any(orders[G.idx(G.mul(G.inv(a), b))] < 3
               for a, b in combinations(cell, 2)):
            continue
        triples.add(normalize(cell))
    triples = sorted(triples)
    assert triples

    def canonical_pair(cells):
        """Remove explicit A gauges and a valid automorphism subgroup.

        x->x^u, y->x^v*y*z^flip, z->z are automorphisms.
        This is not a complete arbitrary Tanner-isomorphism classifier.
        """
        images = []
        for u in range(1, 5):
            for v in range(5):
                for flip in range(2):
                    transformed = []
                    for cell in cells:
                        image = [
                            ((u*a + v*(pow(2, b, 5)-1)) % 5,
                             b, (c + flip*b) % 2)
                            for a, b, c in cell
                        ]
                        transformed.append(normalize(image))
                    images.append(tuple(sorted(transformed)))
        return min(images)

    # Reconstructed deterministically in every pure call. Different A
    # cards are canonical relative geometries, not random cell origins.
    crng = random.Random(405124731)
    needed = max(24, (N_SLOTS+3)//4 + 8)
    target = max(64, 3*needed)
    catalogue, signatures = [], set()
    for attempt in range(20000):
        signature = canonical_pair(
            (crng.choice(triples), crng.choice(triples)))
        if signature in signatures:
            continue
        signatures.add(signature)
        grid = [[list(cell) for cell in signature]]
        rows = component(grid, "A")
        if gf2_rank(rows) != q:
            continue
        four, six, parts = graph(rows)
        catalogue.append(((parts-1, four, six), signature, grid, rows))
        if len(catalogue) >= target:
            break
    catalogue.sort(key=lambda entry: (entry[0], entry[1]))
    if len(catalogue) <= acard:
        raise ValueError("canonical full-rank A catalogue exhausted")
    abook = []
    for coarse, signature, grid, rows in catalogue[:needed]:
        entry = refine(grid, "A", rows, coarse)
        entry["signature"] = signature
        abook.append(entry)
    abook.sort(key=lambda entry: (entry["key"], entry["signature"]))
    if acard >= len(abook):
        raise ValueError("no unused refined canonical A card")
    a = abook[acard]
    A = clone(a["grid"])

    def draw_cell(i, j):
        count = sb[i][j]
        if count == 0:
            return []
        base = G.pick(rng)
        if count == 1:
            return [base]
        if count == 2:
            order = (10, 4, 5)[(i + j + acard) % 3]
            return [base, G.mul(base, rng.choice(palettes[order]))]
        # Whole-cell redraws change internal noncommuting geometry too.
        cell = rng.choice(triples)
        return [G.mul(base, g) for g in cell]

    def draw_B():
        return [[draw_cell(i, j) for j in range(6)] for i in range(2)]

    sites = [(i, j) for i in range(2) for j in range(6) if sb[i][j]]
    pool, seen = [], set()

    def analyze_B(grid):
        rows = component(grid, "B")
        four, six, parts = graph(rows)
        deficit = 2*q-gf2_rank(rows)
        energy = (100*four + six/8.0)/q + 60*deficit + 40*(parts-1)
        return energy, rows, four, six, parts, deficit

    def retain_B(grid, state):
        _, rows, four, six, parts, deficit = state
        if deficit:
            return
        signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                          for row in grid for cell in row)
        if signature in seen:
            return
        seen.add(signature)
        pool.append(((parts-1, four, six), signature,
                     clone(grid), list(rows)))
        pool.sort(key=lambda entry: (entry[0], entry[1]))
        del pool[6:]

    for restart in range(4):
        grid = draw_B()
        state = analyze_B(grid)
        retain_B(grid, state)
        for step in range(220):
            i, j = rng.choice(sites)
            old = grid[i][j]
            grid[i][j] = draw_cell(i, j)
            proposed = analyze_B(grid)
            delta = proposed[0]-state[0]
            temperature = 12.0*(0.982**step)+0.10
            if delta <= 0 or rng.random() < math.exp(
                    -min(700.0, delta/temperature)):
                state = proposed
                if step % 16 == 0:
                    retain_B(grid, state)
            else:
                grid[i][j] = old
        retain_B(grid, state)
    if not pool:
        for attempt in range(256):
            grid = draw_B()
            retain_B(grid, analyze_B(grid))
            if len(pool) >= 3:
                break
    if not pool:
        raise ValueError("B contract missed verified full component rank")
    bpool = [
        refine(grid, "B", rows, coarse)
        for coarse, signature, grid, rows in pool[:4]
    ]
    bpool.sort(key=lambda entry: entry["key"])

    def low4(checks, stabilizers, n):
        """Exact full quantum-logical existence test through weight four."""
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
                old = pivots.get(word.bit_length()-1)
                if old is None:
                    return True
                word ^= old
            return False

        cols = columns(checks, n)
        seen_words = {0: 0}
        for i, syndrome in enumerate(cols):
            word = 1 << i
            old = seen_words.get(syndrome)
            if old is None:
                seen_words[syndrome] = word
            elif nontrivial(word ^ old):
                return True
        for i, syndrome in enumerate(cols):
            for j in range(i):
                value, word = syndrome ^ cols[j], (1 << i) | (1 << j)
                old = seen_words.get(value)
                if old is None:
                    seen_words[value] = word
                elif nontrivial(word ^ old):
                    return True
        return False

    def stress(checks, stabilizers, n, basis):
        """Two fault intervals and a perfect final syndrome, NOT ASC memory."""
        try:
            import numpy as np
        except ImportError:
            return (0, 0, 0)
        m, shots = len(checks), 64
        width = 2*n+m
        augmented = [
            row | (1 << (2*n+r)) for r, row in enumerate(checks)
        ] + [row | (row << n) for row in checks]
        srng = np.random.default_rng(405991 + 1009*idx + 7919*basis)
        intensity = np.repeat(np.asarray((0.25, 0.45)), shots//2)
        hc, sc = columns(checks, n), columns(stabilizers, n)
        degree = np.asarray([
            h.bit_count()+s.bit_count() for h, s in zip(hc, sc)
        ], dtype=float)
        opposite_degree = np.asarray(
            [s.bit_count() for s in sc], dtype=float)
        data_p = 0.5*(1-np.exp(
            -2*0.004*degree[:, None]*intensity[None, :]))
        errors = np.zeros((width+1, shots), dtype=np.uint8)
        errors[:n] = srng.random((n, shots)) < data_p
        errors[n:2*n] = srng.random((n, shots)) < data_p
        for row in stabilizers:
            positions = []
            while row:
                bit = row & -row
                positions.append(bit.bit_length()-1)
                row ^= bit
            event_p = intensity*0.002*(len(positions)+1)
            for interval in range(2):
                events = np.flatnonzero(srng.random(shots) < event_p)
                for shot in events:
                    # Actual opposite-check support, but no claimed
                    # ASC ordering or certified ancilla-hook model.
                    pair = srng.choice(positions, 2, replace=False)
                    errors[interval*n+pair, shot] ^= 1
        measurement_mu = 0.004*np.asarray(
            [row.bit_count()+2 for row in checks], dtype=float)
        measurement_p = 0.5*(1-np.exp(
            -2*measurement_mu[:, None]*intensity[None, :]))
        errors[2*n:width] = (
            srng.random((m, shots)) < measurement_p)
        marginal = 0.5*(1-np.exp(
            -2*0.004*(degree+opposite_degree)[:, None]
            * intensity[None, :]))
        probabilities = np.concatenate(
            (marginal, marginal, measurement_p), axis=0)
        probabilities = np.clip(probabilities, 0.003, 0.18)
        prior = np.log((1-probabilities)/probabilities).astype(np.float32)

        edge_variables, redges, rsites = [], [], []
        vedges = [[] for _ in range(width)]
        for row in augmented:
            edges, positions = [], []
            while row:
                bit = row & -row
                c = bit.bit_length()-1
                row ^= bit
                edge = len(edge_variables)
                edge_variables.append(c)
                edges.append(edge)
                positions.append(c)
                vedges[c].append(edge)
            redges.append(edges)
            rsites.append(positions)
        ne = len(edge_variables)
        rd, vd = max(map(len, redges)), max(map(len, vedges))
        rmap = np.full((len(augmented), rd), ne, dtype=np.intp)
        cmap = np.full((len(augmented), rd), width, dtype=np.intp)
        vmap = np.full((width, vd), ne, dtype=np.intp)
        for r, edges in enumerate(redges):
            rmap[r, :len(edges)] = edges
            cmap[r, :len(edges)] = rsites[r]
        for c, edges in enumerate(vedges):
            vmap[c, :len(edges)] = edges
        valid = rmap != ne
        ev = np.asarray(edge_variables, dtype=np.intp)

        def syndrome(bits):
            return np.bitwise_xor.reduce(bits[cmap], axis=1)

        target = syndrome(errors)
        target_sign = np.where(target, -1.0, 1.0).astype(np.float32)
        incoming = np.empty((ne+1, shots), dtype=np.float32)
        incoming[:ne] = prior[ev]
        incoming[ne] = 30
        messages = np.zeros_like(incoming)
        final = np.zeros_like(errors)
        hard = np.zeros_like(errors)
        active = np.ones(shots, dtype=bool)
        slots = np.arange(rd)[None, :, None]
        late = 0
        for iteration in range(36):
            gathered = incoming[rmap]
            signs = np.where(gathered < 0, -1.0, 1.0)
            magnitude = np.abs(gathered)
            at = np.argmin(magnitude, axis=1)
            first = np.min(magnitude, axis=1)
            second = np.partition(magnitude, 1, axis=1)[:, 1, :]
            excluded = np.where(
                slots == at[:, None, :],
                second[:, None, :], first[:, None, :])
            parity = np.prod(signs, axis=1)*target_sign
            update = 0.80*excluded*signs*parity[:, None, :]
            messages[rmap[valid]] = update[valid]
            messages[ne] = 0
            posterior = prior + messages[vmap].sum(axis=1)
            hard[:width] = posterior < 0
            if iteration % 4 == 3:
                settled = active & np.all(syndrome(hard) == target, axis=0)
                final[:, settled] = hard[:, settled]
                if iteration >= 15:
                    late += int(settled.sum())
                active[settled] = False
                if not active.any():
                    break
            proposal = posterior[ev]-messages[:ne]
            incoming[:ne] = (
                0.25*incoming[:ne]+0.75*np.clip(proposal, -20, 20))
            incoming[ne] = 30
        final[:, active] = hard[:, active]

        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = row
                    break
                row ^= old
        residual = errors[:n] ^ errors[n:2*n] ^ final[:n] ^ final[n:2*n]
        packed = np.packbits(residual.T, axis=1, bitorder="little")
        failures = 0
        for shot in range(shots):
            word = int.from_bytes(packed[shot].tobytes(), "little")
            while word:
                old = pivots.get(word.bit_length()-1)
                if old is None:
                    break
                word ^= old
            failures += int(bool(active[shot]) or bool(word))
        return failures, int(active.sum()), late

    prepared = []
    for b in bpool:
        B = b["grid"]
        hx, hz, n = lifted_matrices(G, A, B)
        k = n-gf2_rank(hx)-gf2_rank(hz)
        if n != 560 or k != 160:
            continue
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx) == wx
        assert max(row.bit_count() for row in hz) == wz
        actual_a = [
            sum(((hx[t] >> (j*6*q)) & mask) << (j*q)
                for j in range(2)) for t in range(q)
        ]
        actual_b = [row & ((1 << (6*q))-1) for row in hz[:2*q]]
        assert actual_a == a["rows"] and actual_b == b["rows"]
        unsafe = int(low4(hx, hz, n) or low4(hz, hx, n))
        structural = (
            unsafe, a["bad"]+b["bad"], a["parts"]+b["parts"],
            max(0, 10-a["d"])+max(0, 10-b["d"]),
            max(a["burden"], b["burden"]), a["burden"]+b["burden"],
            max(a["peak"], b["peak"]),
            max(a["boundary_peak"], b["boundary_peak"]),
            a["boundary_total"]+b["boundary_total"],
            a["four"]+b["four"], a["six"]+b["six"])
        prepared.append((structural, B, hx, hz))
    if not prepared:
        raise ValueError("no complete helper-built k160 pair")
    prepared.sort(key=lambda entry: entry[0])

    winner = None
    for structural, B, hx, hz in prepared[:3]:
        sx, sz = stress(hx, hz, 560, 0), stress(hz, hx, 560, 1)
        # Exact short-logical evidence and coarse protection precede the
        # decoder surrogate. Longer sampled distance earns no extra reward.
        quality = structural[:4] + (
            max(sx[0], sz[0]), sx[0]+sz[0],
            max(sx[1], sz[1]), sx[1]+sz[1], sx[2]+sz[2],
        ) + structural[4:]
        if winner is None or quality < winner[0]:
            winner = quality, B, hx, hz, sx, sz
    quality, B, hx, hz, sx, sz = winner
    expected = 6*q*(8*6 + 3*sum(map(sum, sb)))
    assert expected in (23760, 24480)
    assert 6*sum(row.bit_count() for row in hx+hz) == expected
    assert max(row.bit_count() for row in hx+hz) <= W_MAX
    assert css_ok(hx, hz)
    assert all(len(cell) <= CELL_MAX and
               len({G._key(g) for g in cell}) == len(cell)
               for grid in (A, B) for row in grid for cell in row)
    return {
        "idx": idx, "contract": (acard, architecture),
        "a_geometry": a["signature"], "k": 160,
        "screen": quality, "cnots": expected,
        "two_time_stress": (sx, sz),
        "cells": {"A": A, "B": B},
    }


def CELL(rng, G, side, i, j, ctx):
    """Emit the complete selected cell without subsequent random draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal rolled supports and uniform labels suffice for basis-balanced memory on this bed: parent slot 4 failed 102/1500 X shots versus 1/200000 Z shots, with widespread weight-six witnesses.
# REFUTED: First-stage syndrome settlement certifies logical success: parent slot 12 settled 749/750 X shots at stage 1 yet failed 109/750 scored shots.
# REFUTED: Equal k and d2s imply equal circuit reliability: current slots 1 and 6 both had k=98 and d2s=12, but X failed 11 versus 24 times in 200000 shots and scored 6.776 versus 6.518.
# REFUTED: Increasing d2s guarantees better memory on the parent C17 bed: slot53 had d2s=14, eight X failures/200000 and s=6.806, while slot10 had d2s=13, one failure per basis and s=7.258.
# REFUTED: Sum-free C29 gaps and identical measured n,k,d2s,depth,CNOTs imply uniform memory reliability: slots4/16 both had n580,k132,d2s14,depth16,CNOTs33408, but two/nine Z failures per200000 and worse-basis scores7.341/6.967.
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
