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
CONCEPT = ("For cyclic Z20 in the frozen metacyclic encoding, order 20, shapes (2,5)x(2,5), sparse unit-trinomial columns and reduced exact weight-seven/eight component multiplicity should improve worse-basis circuit fitness above 5.7715 while keeping checks at weight at most twelve.")
GROUP = ("Keep the exact custom Z20 group and order for L1. This bed reaches approximately 5.8, including the current 5.7715 program. The experiment isolates filling effects. Odd trinomials are units in F2[Z20]: x^20-1=(x+1)^4*(x^4+x^3+x^2+x+1)^4, and an odd polynomial with at most three terms cannot vanish at either factor.")
SHAPES = ("Frozen equal (2,5) components have 100 variables and 40 checks each. The size factor 29 permits q<=20; q=20 gives n=580, kb=180 and guaranteed rate 9/29=0.310345 in cell 0.300. Require actual 180<=k<189 through complete verified ranks. Parent failures collectively affected all 180 logicals, with mostly 1.1-1.2x enrichment across changing Acol/Bcol data and Acol/Brow check blocks; neither the maps nor worst-logical indices establish one persistent repair target.")
FILLING = ("Twenty indexed contracts cross four sparse support pairs with five triangle/binomial relative-order policies. Every code contains zero and trinomial cells. Rows total seven/eight or eight/eight, columns total three/four, checks weigh at most twelve, and six-round extraction uses 25200, 26040 or 26880 CNOTs. This changes geometry, not necessarily gate exposure favorably. A deterministic relative-support catalogue removes obvious multiplier, cell-origin, row/column and side-exchange copies, not arbitrary Tanner isomorphisms. Anneal origins using cached verified singleton actions. Finalists screen connectivity, rank, exact kernels through six, exact weight-seven/eight multiplicities under an explicitly checked translation action, and absorbing-triple exposure. Complete helper-built CSS matrices retain the rate band and prefer no quantum logical through four. Common-noise joint BP probes include opposite-check pairs, gate-weighted data noise, and stabilizer classification of settled residuals; they omit ASC scheduling and measurement histories. All five parent codes reached failure caps; slot53 had d2s=9,s=5.811 versus slot2's d2s=8,s=5.829, motivating decoder-risk selection rather than distance-only ordering.")
FAMILY_NAME = "r030_m1r20s1_25x25_sparse_exact8_jointbp_v3"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 20

def custom_ops(q):
    # Same metacyclic rule; order resized because 29*21 exceeds the window.
    m, r, s = 1, 20, 1
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
B_SHAPE = (2, 5)               # (rb, cb)


# Columns are prescribed term-count pairs, not random support draws.
# Every profile has five columns of weight 3 or 4 and row sums <=8.
_SUPPORT_PROFILES = (
    ((2,1),(2,1),(1,2),(1,2),(1,2)),
    ((2,1),(2,1),(1,2),(1,2),(2,2)),
    ((2,1),(1,2),(1,2),(1,2),(3,1)),
    ((3,0),(2,1),(1,2),(1,2),(1,2)),
    ((3,0),(1,2),(1,2),(1,2),(1,2)),
    ((3,0),(0,3),(2,1),(1,2),(1,2)),
    ((3,0),(0,3),(2,1),(1,2),(2,2)),
    ((3,0),(1,2),(1,2),(1,2),(2,2)),
)
_PROFILE_PAIRS = (
    (0,0),(0,1),(0,2),(1,1),(1,2),(2,2),
    (0,3),(0,4),(1,3),(1,4),(2,3),(2,4),
    (3,3),(3,4),(4,4),(0,5),(1,5),(2,5),
    (0,6),(0,7),
)


def _support_profile(rng, number):
    columns = list(_SUPPORT_PROFILES[number])
    rng.shuffle(columns)
    swap = rng.randrange(2)
    return [[column[i ^ swap] for column in columns] for i in range(2)]


def _component_rows(G, side, grid):
    """Extract the component action from the verified construction.

    The opposite side is a zero 1x1 probe, not a submitted component.
    This preserves the evaluator's left/right conventions exactly.
    """
    if side == "A":
        rows = lifted_matrices(G, grid, [[[]]])[0]
    else:
        rows = lifted_matrices(G, [[[]]], grid)[1]
    mask = (1 << (len(grid[0]) * G.q)) - 1
    return [row & mask for row in rows]


def _cycle_counts(rows):
    """Exact component Tanner four- and six-cycle counts."""
    size = len(rows)
    intersections = [[0] * size for _ in range(size)]
    counts = [[0] * size for _ in range(size)]
    four = 0
    for i in range(size):
        for j in range(i + 1, size):
            common = rows[i] & rows[j]
            number = common.bit_count()
            intersections[i][j] = common
            counts[i][j] = number
            four += number * (number - 1) // 2
    six = 0
    for i in range(size):
        for j in range(i + 1, size):
            a = counts[i][j]
            if not a:
                continue
            for k in range(j + 1, size):
                b, c = counts[i][k], counts[j][k]
                if b and c:
                    common = (intersections[i][j] & rows[k]).bit_count()
                    # Count distinct variable choices, excluding stars.
                    six += a*b*c - common*(a+b+c) + 2*common
    return four, six


def _short_collision_counts(rows, columns):
    """Cheap component-codeword screen through weight four.

    Pair collisions with overlapping supports can overcount when
    weight-one/two words exist; those are already penalized first.
    """
    syndromes = [0] * columns
    for check, row in enumerate(rows):
        while row:
            bit = row & -row
            syndromes[bit.bit_length() - 1] |= 1 << check
            row ^= bit
    singles = {}
    one = two = 0
    for syndrome in syndromes:
        one += int(syndrome == 0)
        two += singles.get(syndrome, 0)
        singles[syndrome] = singles.get(syndrome, 0) + 1
    pairs = {}
    three = four = 0
    for i in range(columns):
        for j in range(i + 1, columns):
            syndrome = syndromes[i] ^ syndromes[j]
            three += singles.get(syndrome, 0)
            four += pairs.get(syndrome, 0)
            pairs[syndrome] = pairs.get(syndrome, 0) + 1
    return one, two, three, four


def _copy_cells(grid):
    return [[list(cell) for cell in row] for row in grid]


def _design_component(rng, G, side, support, idx):
    """Independent multistart label search; no orbit-copy construction."""
    import math
    positions = [(i,j) for i in range(2) for j in range(5)
                 if support[i][j]]
    selected = None
    selected_quality = None
    # Extra harness slots use different search pressure as well as labels.
    cycle_weight = 200 + 20 * (idx // len(_PROFILE_PAIRS))
    for restart in range(4):
        grid = [[rng.sample(G.elements, support[i][j])
                 for j in range(5)] for i in range(2)]
        rows = _component_rows(G, side, grid)
        cycles = _cycle_counts(rows)
        score = cycle_weight * cycles[0] + cycles[1]
        local_best = _copy_cells(grid)
        local_score = score

        for step in range(180):
            i, j = rng.choice(positions)
            cell = grid[i][j]
            position = rng.randrange(len(cell))
            old = cell[position]
            occupied = {G._key(g) for t,g in enumerate(cell)
                        if t != position}
            replacement = G.pick(rng)
            while G._key(replacement) in occupied:
                replacement = G.pick(rng)
            if G._key(replacement) == G._key(old):
                continue
            cell[position] = replacement
            trial_cycles = _cycle_counts(_component_rows(G, side, grid))
            trial_score = (cycle_weight * trial_cycles[0]
                           + trial_cycles[1])
            temperature = 35.0 * (1.0 - step / 180.0) + 0.3
            delta = trial_score - score
            if delta <= 0 or rng.random() < math.exp(-delta / temperature):
                score = trial_score
                if score < local_score:
                    local_score = score
                    local_best = _copy_cells(grid)
            else:
                cell[position] = old

        rows = _component_rows(G, side, local_best)
        quality = (
            len(rows) - gf2_rank(rows),
        ) + _short_collision_counts(rows, 5 * G.q) + _cycle_counts(rows)
        if selected_quality is None or quality < selected_quality:
            selected_quality = quality
            selected = local_best
    return selected


def _retired_generatorgap_ctx(rng, G, idx):
    """Inactive historical binomial experiment; new slots do not call it."""
    import math

    # U, E, F denote gaps of orders 20, 10, 5 respectively.
    # Top-row doubletons occupy columns 0,1; bottom ones occupy 2,3,4.
    signatures = (
        (("U","U"), ("U","U","U")),
        (("U","U"), ("U","U","E")),
        (("U","U"), ("U","U","F")),
        (("U","E"), ("U","U","U")),
        (("U","F"), ("U","U","U")),
        (("U","E"), ("U","U","E")),
        (("U","F"), ("U","U","F")),
        (("U","E"), ("U","U","F")),
        (("U","F"), ("U","U","E")),
        (("U","U"), ("U","E","F")),
    )
    designs = (
        (0,0),(0,1),(0,2),(0,3),(0,4),
        (1,1),(1,2),(1,3),(1,4),
        (2,2),(2,3),(2,4),
        (3,3),(3,4),(4,4),
        (0,5),(0,6),(0,7),(0,8),(0,9),
    )
    gapsets = {"U": (1,3,7,9), "E": (2,6), "F": (4,8)}
    pa, pb = designs[idx % 20]
    mode = (idx // 20) % 3
    q = G.q
    mask = (1 << q) - 1

    def architecture(number, reinforced):
        counts = [[2,2,1,1,1], [1,1,2,2,2]]
        kinds = {(0,0): signatures[number][0][0],
                 (0,1): signatures[number][0][1]}
        kinds.update({(1,j+2): kind
                      for j, kind in enumerate(signatures[number][1])})
        if reinforced:
            counts[0][4] = 2
            kinds[0,4] = "U"
        return counts, kinds

    sa, ka = architecture(pa, mode >= 1)
    sb, kb = architecture(pb, mode >= 2)
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)

    def draw(counts, kinds):
        grid = [[None] * 5 for _ in range(2)]
        for i in range(2):
            used = set()
            for j in range(5):
                g = G.pick(rng)
                if counts[i][j] == 1:
                    grid[i][j] = [g]
                    continue
                choices = [d for d in gapsets[kinds[i,j]] if d not in used]
                d = rng.choice(choices)
                used.add(d)
                grid[i][j] = [g, G.mul(g, (0,d))]
        return grid

    def columns(rows):
        out = [0] * (5 * q)
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return out

    def fast_cycles(rows, cols):
        # Four-cycles are exact. The triangle statistic is exact for
        # six-cycles when four-cycles are absent; otherwise it is a proxy.
        adjacency = [0] * len(rows)
        four = 0
        for i, row in enumerate(rows):
            for j in range(i):
                overlap = (row & rows[j]).bit_count()
                four += overlap * (overlap - 1) // 2
                if overlap:
                    adjacency[i] |= 1 << j
                    adjacency[j] |= 1 << i
        triangles = 0
        for i in range(len(rows)):
            later = adjacency[i] >> (i + 1)
            while later:
                bit = later & -later
                j = i + 1 + bit.bit_length() - 1
                triangles += ((adjacency[i] & adjacency[j])
                              >> (j + 1)).bit_count()
                later ^= bit
        stars = sum(w * (w-1) * (w-2) // 6
                    for w in (c.bit_count() for c in cols))
        return four, max(0, triangles - stars)

    def assess(side, grid):
        rows = _component_rows(G, side, grid)
        cols = columns(rows)
        four, six = fast_cycles(rows, cols)
        deficit = len(rows) - gf2_rank(rows)
        energy = 400 * deficit + 60 * four / q + six / (8 * q)
        return energy, rows, cols, deficit, four, six

    def triple_statistics(cols):
        # With no words of weight <=4, equal triple syndromes count
        # weight-six words (ten partitions per word).
        seen = {}
        collisions = boundary = 0
        for i in range(len(cols)):
            for j in range(i):
                pair = cols[i] ^ cols[j]
                for h in range(j):
                    syndrome = pair ^ cols[h]
                    weight = syndrome.bit_count()
                    if weight < 3:
                        boundary += (3 - weight) ** 2
                    old = seen.get(syndrome, 0)
                    collisions += old
                    seen[syndrome] = old + 1
        return collisions, boundary

    def orbit_key(word):
        # Regular translations only canonicalize verified witnesses;
        # they do not construct or screen a different lift.
        best = word
        shifted = word
        for _ in range(q - 1):
            rotated = 0
            for j in range(5):
                block = (shifted >> (j*q)) & mask
                block = ((block << 1) & mask) | (block >> (q-1))
                rotated |= block << (j*q)
            shifted = rotated
            best = min(best, shifted)
        return best

    def sampled_kernel(rows):
        """Sample verified component words, not a distance certificate."""
        witnesses = {}
        minimum = 5*q + 1
        for _ in range(64):
            rr = list(rows)
            order = list(range(5*q))
            rng.shuffle(order)
            pivots = []
            rank = 0
            for c in order:
                bit = 1 << c
                hit = next((t for t in range(rank, len(rr))
                            if rr[t] & bit), None)
                if hit is None:
                    continue
                rr[rank], rr[hit] = rr[hit], rr[rank]
                pivot = rr[rank]
                for t in range(len(rr)):
                    if t != rank and rr[t] & bit:
                        rr[t] ^= pivot
                pivots.append(c)
                rank += 1
                if rank == len(rr):
                    break
            occupied = set(pivots)
            for c in order:
                if c in occupied:
                    continue
                word = 1 << c
                for p, row in zip(pivots, rr[:rank]):
                    if (row >> c) & 1:
                        word |= 1 << p
                weight = word.bit_count()
                minimum = min(minimum, weight)
                if weight <= 10:
                    witnesses[orbit_key(word)] = weight

        eight = [word for word, weight in witnesses.items() if weight <= 8]
        exposure = [sum(bool((word >> (j*q)) & mask) for word in eight)
                    for j in range(5)]
        return (min(minimum, 12), len(eight), max(exposure, default=0),
                len(witnesses))

    def select(side, counts, kinds):
        pool = []
        positions = [(i,j) for i in range(2) for j in range(5)]
        for restart in range(6):
            grid = draw(counts, kinds)
            current = assess(side, grid)
            best_grid = _copy_cells(grid)
            best = current

            for step in range(180):
                i, j = rng.choice(positions)
                old = grid[i][j]
                g = G.pick(rng)
                if counts[i][j] == 1:
                    proposed_cell = [g]
                else:
                    used = set()
                    for other in range(5):
                        if other != j and counts[i][other] == 2:
                            difference = G.mul(G.inv(grid[i][other][0]),
                                               grid[i][other][1])[1]
                            used.add(min(difference, q-difference))
                    choices = [d for d in gapsets[kinds[i,j]]
                               if d not in used]
                    d = rng.choice(choices)
                    proposed_cell = [g, G.mul(g, (0,d))]
                grid[i][j] = proposed_cell
                proposed = assess(side, grid)
                delta = proposed[0] - current[0]
                temperature = 12 * (1 - step / 180) + 0.15
                if delta <= 0 or rng.random() < math.exp(
                        -min(700.0, delta / temperature)):
                    current = proposed
                    if current[0] < best[0]:
                        best = current
                        best_grid = _copy_cells(grid)
                else:
                    grid[i][j] = old

            _, rows, cols, deficit, four, six = best
            short = _short_collision_counts(rows, 5*q)
            pool.append((deficit, short, four, six, restart,
                         best_grid, rows, cols))

        pool.sort(key=lambda item: item[:5])
        finalists = []
        for deficit, short, four, six, serial, grid, rows, cols in pool[:4]:
            collisions, boundary = triple_statistics(cols)
            distance, eight, worst, ten = sampled_kernel(rows)
            # Remove actual short words before optimizing multiplicity.
            # Mode 0 emphasizes total exposure; reinforcement modes also
            # emphasize the most frequently touched component block.
            exposure = (eight, worst) if mode == 0 else (worst, eight)
            key = (deficit, short, four, collisions, boundary,
                   -distance, exposure, ten, six, serial)
            finalists.append((key, grid))
        finalists.sort(key=lambda item: item[0])
        return finalists

    # Keep alternatives rather than voiding a slot when one optimized
    # component has a small rank deficit.
    for attempt in range(2):
        aa = select("A", sa, ka)
        bb = select("B", sb, kb)
        combinations = [(ia + ib, ia, ib, A, B)
                        for ia, (_, A) in enumerate(aa)
                        for ib, (_, B) in enumerate(bb)]
        combinations.sort(key=lambda item: item[:3])
        for _, ia, ib, A, B in combinations:
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (k > 0 and 11*n <= 40*k < 13*n):
                continue
            assert css_ok(hx, hz)
            assert max(row.bit_count() for row in hx + hz) <= W_MAX
            return {"idx": idx, "cells": {"A": A, "B": B},
                    "k": k, "screen": (aa[ia][0], bb[ib][0])}
    raise ValueError("no generator-gap filling in rate cell 0.300")


def NEW_CTX(rng, G, idx):
    """Sparse indexed contracts, exact short spectra, and joint BP selection."""
    import math
    from itertools import combinations
    from math import gcd
    import numpy as np

    q = G.q
    assert q == 20 and A_SHAPE == B_SHAPE == (2, 5)
    mask = (1 << q) - 1

    profiles = (
        ((3,0),(0,3),(1,2),(1,2),(2,1)),
        ((3,0),(1,2),(1,2),(1,2),(1,2)),
        ((3,0),(0,3),(1,2),(2,1),(2,2)),
        ((3,1),(0,3),(1,2),(2,1),(2,1)),
    )
    support_pairs = (
        ((0,0),(0,1),(0,2),(2,3)),
        ((1,1),(1,2),(1,3),(3,3)),
        ((0,3),(2,2),(1,3),(0,2)),
    )
    triangle_spectra = (
        (5,20,20), (10,20,20), (5,20,20),
        (5,20,20), (4,5,20),
    )
    short_orders = ((), (10,), (5,), (10,10), (10,5))
    units = [u for u in range(1, q) if gcd(u, q) == 1]
    palettes = {
        order: [d for d in range(1, q) if q // gcd(d, q) == order]
        for order in (5, 10, 20)
    }

    def direction(d):
        d %= q
        return min(d, q-d)

    triangle_pools = {}
    for spectrum in set(triangle_spectra):
        available = []
        for u, v in combinations(range(1, q), 2):
            directions = (direction(u), direction(v), direction(v-u))
            measured = tuple(sorted(q // gcd(d, q)
                                    for d in (u, v, (v-u) % q)))
            if measured == spectrum and len(set(directions)) == 3:
                available.append((0, u, v))
        assert available
        triangle_pools[spectrum] = available

    def contract(number):
        architecture, policy = divmod(number % 20, 5)
        phase = number // 20
        pa, pb = support_pairs[phase % 3][architecture]
        counts = {}
        for side, p in (("A", pa), ("B", pb)):
            counts[side] = [[column[i] for column in profiles[p]]
                            for i in range(2)]
        assert _no_dead_lines(counts["A"], counts["B"])
        assert _grid_weights(counts["A"], counts["B"]) <= min(12, W_MAX)
        return counts, architecture, policy, phase

    def relative_grid(source, counts, policy):
        """Prescribed spectra and exact short-order counts, not orbit copies."""
        bins = [(i,j) for i in range(2) for j in range(5)
                if counts[i][j] == 2]
        for attempt in range(20000):
            orders = list(short_orders[policy])
            orders += [20] * (len(bins)-len(orders))
            source.shuffle(orders)
            schedule = dict(zip(bins, orders))
            grid = [[[] for _ in range(5)] for _ in range(2)]
            clean = True
            for i in range(2):
                used = set()
                # Place triangles first; binomials then consume remaining
                # row directions. This excludes same-row four-cycles.
                sites = sorted(range(5),
                               key=lambda j: -counts[i][j])
                for j in sites:
                    weight = counts[i][j]
                    if weight == 0:
                        cell = []
                    elif weight == 1:
                        cell = [0]
                    elif weight == 2:
                        choices = [d for d in palettes[schedule[i,j]]
                                   if direction(d) not in used]
                        if not choices:
                            clean = False
                            break
                        cell = [0, source.choice(choices)]
                    else:
                        candidates = [
                            cell for cell in triangle_pools[
                                triangle_spectra[policy]]
                            if not any(direction(a-b) in used
                                       for a, b in combinations(cell, 2))
                        ]
                        if not candidates:
                            clean = False
                            break
                        cell = list(source.choice(candidates))
                    for a, b in combinations(cell, 2):
                        d = direction(a-b)
                        if d in used:
                            clean = False
                        used.add(d)
                    grid[i][j] = tuple(cell)
                if not clean:
                    break
            if clean:
                return grid
        raise ValueError("sparse relative-order contract exhausted")

    def geometry_key(grids):
        """Internal-support invariant, not a full Tanner classifier."""
        images = []
        for u in units:
            sides = []
            for grid in grids:
                columns = []
                for j in range(5):
                    normalized = []
                    for i in range(2):
                        values = [u*v % q for v in grid[i][j]]
                        normalized.append(
                            min(tuple(sorted((v-a) % q for v in values))
                                for a in values) if values else ())
                    columns.append(tuple(normalized))
                sides.append(min(
                    tuple(sorted(columns)),
                    tuple(sorted((b,a) for a,b in columns))))
            images.append(tuple(sorted(sides)))
        return min(images)

    # Rebuild a pure deterministic prefix. Each slot receives a new
    # relative-support contract before any origin optimization.
    source = random.Random(203031947)
    seen_geometry = set()
    for number in range(idx+1):
        counts, architecture, policy, phase = contract(number)
        for proposal in range(4096):
            ga = relative_grid(source, counts["A"], policy)
            gb = relative_grid(source, counts["B"], policy)
            identity = geometry_key((ga, gb))
            if identity not in seen_geometry:
                seen_geometry.add(identity)
                geometry = {"A": ga, "B": gb}
                break
        else:
            raise ValueError("inequivalent sparse geometry catalogue exhausted")

    # Only scaffold-built singleton actions are cached.
    actions = {"A": [], "B": []}
    for g in G.elements:
        hx, _, _ = lifted_matrices(G, [[[g]]], [[[]]])
        _, hz, _ = lifted_matrices(G, [[[]]], [[[g]]])
        actions["A"].append(tuple(row & mask for row in hx))
        actions["B"].append(tuple(row & mask for row in hz))

    def component(side, grid):
        result = []
        for grid_row in grid:
            rows = [0] * q
            for j, cell in enumerate(grid_row):
                for g in cell:
                    block = actions[side][G.idx(g)]
                    for r, bits in enumerate(block):
                        rows[r] ^= bits << (j*q)
            result.extend(rows)
        return result

    def columns(rows, width):
        result = [0] * width
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                result[bit.bit_length()-1] |= tag
                row ^= bit
        return result

    def cell_at(side, i, j):
        origin = rng.randrange(q)
        return [(0, (origin+d) % q) for d in geometry[side][i][j]]

    def draw(side):
        return [[cell_at(side, i, j) for j in range(5)]
                for i in range(2)]

    def coarse(side, grid):
        rows = component(side, grid)
        cols = columns(rows, 5*q)
        adjacency = [0] * len(rows)
        four = 0
        for i, row in enumerate(rows):
            for j in range(i):
                overlap = (row & rows[j]).bit_count()
                four += overlap*(overlap-1)//2
                if overlap:
                    adjacency[i] |= 1 << j
                    adjacency[j] |= 1 << i
        triangles = 0
        for i, neighbors in enumerate(adjacency):
            later = neighbors >> (i+1)
            while later:
                bit = later & -later
                j = i+1+bit.bit_length()-1
                triangles += ((neighbors & adjacency[j])
                              >> (j+1)).bit_count()
                later ^= bit
        stars = sum(d*(d-1)*(d-2)//6
                    for d in (c.bit_count() for c in cols))
        # Exact C6 when C4=0; otherwise only an annealing surrogate.
        six = max(0, triangles-stars)
        unseen = (1 << len(rows))-1
        parts = 0
        while unseen:
            todo = unseen & -unseen
            unseen ^= todo
            parts += 1
            while todo:
                bit = todo & -todo
                todo ^= bit
                neighbors = adjacency[bit.bit_length()-1] & unseen
                unseen ^= neighbors
                todo |= neighbors
        deficit = len(rows)-gf2_rank(rows)
        key = (parts-1, deficit, four, six)
        energy = 80*(parts-1)+40*deficit+(80*four+six/4)/q
        return energy, key, rows, cols

    def short_and_traps(cols):
        """Exact kernel exclusion through six; absorbing triples are a surrogate."""
        seen = {0}
        for value in cols:
            if value in seen:
                return True, (10**12, 10**12)
            seen.add(value)
        for i, x in enumerate(cols):
            for y in cols[:i]:
                value = x ^ y
                if value in seen:
                    return True, (10**12, 10**12)
                seen.add(value)
        loads = [0] * 5
        total = 0
        degrees = [c.bit_count() for c in cols]
        for i, x in enumerate(cols):
            for j in range(i):
                pair = x ^ cols[j]
                for h in range(j):
                    syndrome = pair ^ cols[h]
                    if syndrome in seen:
                        return True, (10**12, 10**12)
                    seen.add(syndrome)
                    boundary = syndrome.bit_count()
                    if 0 < boundary <= 4 and all(
                            2*(syndrome & cols[v]).bit_count() < degrees[v]
                            for v in (i,j,h)):
                        cost = (6-boundary)**2
                        total += cost
                        for v in (i,j,h):
                            loads[v//q] += cost
        return False, (max(loads), total)

    def exact_seven_eight(cols):
        """Exact component counts after exclusion of kernels through six.

        Weight each four-subset by its number of lift-zero vertices.
        Equal four syndromes have disjoint supports under the precondition.
        Translation averaging gives N8=q*hits/(8*35).
        Four/three collisions similarly give N7=q*hits/(7*20).
        This includes shorter translation orbits at composite q.
        """
        def translate(value, shift):
            return sum(
                ((((value >> (r*q)) & mask) << shift
                  | (((value >> (r*q)) & mask) >> (q-shift))) & mask)
                << (r*q) for r in range(2))

        for j in range(5):
            seed = cols[j*q]
            assert all(cols[j*q+t] == translate(seed, t)
                       for t in range(q)), "unverified syndrome translation"

        anchored = {}
        width = len(cols)
        for anchor in range(0, width, q):
            allowed = [v for v in range(width) if v != anchor]
            x = cols[anchor]
            for bi in range(len(allowed)-2):
                b = allowed[bi]
                ab = x ^ cols[b]
                for ci in range(bi+1, len(allowed)-1):
                    c = allowed[ci]
                    abc = ab ^ cols[c]
                    for di in range(ci+1, len(allowed)):
                        value = abc ^ cols[allowed[di]]
                        anchored[value] = anchored.get(value, 0)+1
        lookup = anchored.get
        seven = eight = 0
        for a in range(width-2):
            x = cols[a]
            for b in range(a+1, width-1):
                ab = x ^ cols[b]
                for c in range(b+1, width):
                    seven += lookup(ab ^ cols[c], 0)
        for a in range(width-3):
            x = cols[a]
            for b in range(a+1, width-2):
                ab = x ^ cols[b]
                for c in range(b+1, width-1):
                    abc = ab ^ cols[c]
                    for d in range(c+1, width):
                        eight += lookup(abc ^ cols[d], 0)
        eight -= sum(anchored.values())
        assert eight >= 0
        assert (q*seven) % 140 == 0 and (q*eight) % 280 == 0
        return q*seven//140, q*eight//280

    def select(side):
        positions = [(i,j) for i in range(2) for j in range(5)
                     if counts[side][i][j]]
        pool, identities = [], set()
        serial = 0

        def retain(grid, result):
            nonlocal serial
            signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                              for row in grid for cell in row)
            if signature in identities:
                return
            identities.add(signature)
            serial += 1
            _, key, rows, cols = result
            pool.append((key, serial, _copy_cells(grid), rows, cols))
            pool.sort(key=lambda item: item[:2])
            del pool[6:]

        for restart in range(4):
            grid = draw(side)
            current = coarse(side, grid)
            best_grid, best = _copy_cells(grid), current
            retain(grid, current)
            for step in range(180):
                i, j = rng.choice(positions)
                old = grid[i][j]
                grid[i][j] = cell_at(side, i, j)
                proposed = coarse(side, grid)
                delta = proposed[0]-current[0]
                temperature = 8*(0.978**step)+0.08
                if delta <= 0 or rng.random() < math.exp(
                        -min(700.0, delta/temperature)):
                    current = proposed
                    if (current[0] < best[0] or
                            (current[0] == best[0] and rng.random() < 0.06)):
                        best_grid, best = _copy_cells(grid), current
                else:
                    grid[i][j] = old
                if step in (89,179):
                    retain(grid, current)
            retain(best_grid, best)

        preliminary = []
        for cost, number, grid, rows, cols in pool:
            bad, traps = short_and_traps(cols)
            key = (cost[0], int(bad), cost[1], traps, cost[2:])
            preliminary.append((key, number, grid, rows, cols, bad, traps))
        preliminary.sort(key=lambda item: item[:2])

        finalists = []
        for _, number, grid, rows, cols, bad, traps in preliminary[:2]:
            n7, n8 = ((10**12,10**12) if bad
                      else exact_seven_eight(cols))
            four, six = _cycle_counts(rows)
            key = (int(bad), n7, n8, traps[0], traps[1],
                   four, six, len(rows)-gf2_rank(rows))
            finalists.append((key, number, grid))
        finalists.sort(key=lambda item: item[:2])
        return finalists

    def span(rows):
        pivots = {}
        for row in rows:
            while row:
                p = row.bit_length()-1
                if p in pivots:
                    row ^= pivots[p]
                else:
                    pivots[p] = row
                    break
        return pivots

    def small_logical(checks, stabilizers, n):
        """Exact logical existence through four, using quotient representatives."""
        pivots = span(stabilizers)
        residues = [1 << c for c in range(n)]
        for p in sorted(pivots):
            tail = pivots[p] ^ (1 << p)
            value = 0
            while tail:
                bit = tail & -tail
                value ^= residues[bit.bit_length()-1]
                tail ^= bit
            residues[p] = value
        cols = columns(checks, n)
        seen = {0: 0}
        for c, syndrome in enumerate(cols):
            value = residues[c]
            old = seen.get(syndrome)
            if old is not None and old != value:
                return True
            seen[syndrome] = value
        for a, x in enumerate(cols):
            for b in range(a):
                syndrome = x ^ cols[b]
                value = residues[a] ^ residues[b]
                old = seen.get(syndrome)
                if old is not None and old != value:
                    return True
                seen[syndrome] = value
        return False

    def bp_probe(checks, stabilizers, n, seed):
        """Common-noise decoder-risk surrogate, NOT the ASC memory experiment."""
        def neighborhoods(rows):
            result = []
            for row in rows:
                nodes = []
                while row:
                    bit = row & -row
                    nodes.append(bit.bit_length()-1)
                    row ^= bit
                result.append(nodes)
            return result

        ns = neighborhoods(checks)
        hooks = neighborhoods(stabilizers)
        flat = np.asarray([c for row in ns for c in row], dtype=np.int32)
        edges, nr = len(flat), len(ns)
        maxweight = max(map(len, ns))
        check_edges = np.full((nr, maxweight), edges, dtype=np.int32)
        incidence = np.full((nr, maxweight), n, dtype=np.int32)
        edge_rows, edge_slots = [], []
        cursor = 0
        for r, row in enumerate(ns):
            length = len(row)
            check_edges[r, :length] = np.arange(cursor, cursor+length)
            incidence[r, :length] = row
            edge_rows.extend([r]*length)
            edge_slots.extend(range(length))
            cursor += length
        edge_rows = np.asarray(edge_rows, dtype=np.int32)
        edge_slots = np.asarray(edge_slots, dtype=np.int32)
        degree = np.bincount(flat, minlength=n)
        variable_edges = np.full(
            (n, int(degree.max())), edges, dtype=np.int32)
        used = np.zeros(n, dtype=np.int32)
        for edge, c in enumerate(flat):
            variable_edges[c, used[c]] = edge
            used[c] += 1

        other_degree = np.bincount(
            [c for row in hooks for c in row], minlength=n)
        channel = np.random.default_rng(seed)
        shots = 64
        levels = np.asarray((0.012, 0.023, 0.037))
        # No idling term: weight channel exposure by extraction incidence.
        rates = np.minimum(
            levels[np.arange(shots) % 3, None]
            * (degree+other_degree)[None, :] / 6.0, 0.22)
        error = channel.random((shots, n)) < rates
        for shot in range(shots):
            row = hooks[int(channel.integers(len(hooks)))]
            if len(row) >= 2 and channel.random() < 0.35:
                a = int(channel.integers(len(row)))
                b = int(channel.integers(len(row)-1))
                b += b >= a
                error[shot, row[a]] ^= True
                error[shot, row[b]] ^= True
        prior_rates = np.minimum(rates+0.7/n, 0.25)
        prior = np.log((1-prior_rates)/prior_rates)
        padded_error = np.pad(error, ((0,0),(0,1)), constant_values=False)
        target = (padded_error[:, incidence].sum(axis=2) & 1).astype(bool)
        target_sign = np.where(target, -1.0, 1.0)
        messages = prior[:, flat].copy()
        pivots = span(stabilizers)

        def nontrivial(word):
            while word:
                p = word.bit_length()-1
                if p not in pivots:
                    return True
                word ^= pivots[p]
            return False

        failed = iteration_sum = 0
        for iteration in range(1, 31):
            count = len(error)
            padded = np.full((count, edges+1), 24.0)
            padded[:, :edges] = messages
            shaped = padded[:, check_edges]
            signs = np.where(shaped < 0, -1.0, 1.0)
            logarithms = np.log(np.clip(
                np.tanh(np.abs(shaped)/2), 1e-12, 1-1e-9))
            products = np.minimum(np.exp(
                logarithms.sum(axis=2, keepdims=True)-logarithms),
                1-1e-7)
            magnitude = np.log1p(products)-np.log1p(-products)
            outgoing_checks = (magnitude * signs
                               * signs.prod(axis=2, keepdims=True)
                               * target_sign[:, :, None])
            outgoing = outgoing_checks[:, edge_rows, edge_slots]
            padded = np.zeros((count, edges+1))
            padded[:, :edges] = outgoing
            posterior = prior+padded[:, variable_edges].sum(axis=2)
            hard = posterior < 0
            padded_hard = np.pad(
                hard, ((0,0),(0,1)), constant_values=False)
            syndrome = (padded_hard[:, incidence].sum(axis=2) & 1).astype(bool)
            settled = np.all(syndrome == target, axis=1)
            for local in np.flatnonzero(settled):
                word = 0
                for c in np.flatnonzero(error[local] ^ hard[local]):
                    word |= 1 << int(c)
                failed += int(nontrivial(word))
                iteration_sum += iteration
            keep = ~settled
            if not np.any(keep):
                return failed, 0, iteration_sum
            messages = np.clip(
                0.35*messages+0.65*(posterior[:, flat]-outgoing),
                -24, 24)[keep]
            prior, error = prior[keep], error[keep]
            target, target_sign = target[keep], target_sign[keep]
        unresolved = len(error)
        return failed+unresolved, unresolved, iteration_sum+31*unresolved

    def finish(A, B, hx, hz, n, k, structure, probe=None):
        assert n == 580 and 180 <= k < 189
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx+hz) <= min(12, W_MAX)
        assert all(len(set(cell)) == len(cell)
                   for grid in (A,B) for row in grid for cell in row)
        assert component("A", A) == _component_rows(G, "A", A)
        assert component("B", B) == _component_rows(G, "B", B)
        expected = 7*q*6*sum(
            sum(map(sum, counts[side])) for side in ("A","B"))
        assert 6*sum(row.bit_count() for row in hx+hz) == expected
        return {
            "idx": idx, "contract": (architecture, policy, phase),
            "relative_geometry": identity, "k": k,
            "component_screen": structure, "bp_surrogate": probe,
            "cnots": expected, "cells": {"A": A, "B": B},
        }

    apool, bpool = select("A"), select("B")
    options = []
    for ak, at, A in apool:
        for bk, bt, B in bpool:
            structure = tuple(v for a,b in zip(ak,bk)
                              for v in (max(a,b), a+b))
            options.append((structure, at, bt, A, B))
    options.sort(key=lambda item: item[:3])

    finalists, fallback = [], None
    for structure, at, bt, A, B in options:
        hx, hz, n = lifted_matrices(G, A, B)
        k = n-gf2_rank(hx)-gf2_rank(hz)
        if not (n == 580 and 180 <= k < 189):
            continue
        item = (A,B,hx,hz,n,k,structure)
        if fallback is None:
            fallback = item
        if small_logical(hx, hz, n) or small_logical(hz, hx, n):
            continue
        seed = 203033071+1009*idx
        x = bp_probe(hz, hx, n, seed)
        z = bp_probe(hx, hz, n, seed+1)
        # Retain a short-protection floor, then select on worse-basis
        # decoder risk rather than maximizing a distance upper bound.
        key = (structure[0], int(structure[2] > 0),
               max(x[0],z[0]), x[0]+z[0],
               max(x[1],z[1]), x[1]+z[1],
               x[2]+z[2], structure, at, bt)
        finalists.append((key, item, (x,z)))
    if finalists:
        finalists.sort(key=lambda item: item[0])
        _, item, probe = finalists[0]
        return finish(*item, probe=probe)
    if fallback is not None:
        return finish(*fallback)

    # Rescue changes origins only, preserving the indexed support and
    # relative-order contract. Full ranks, not component predictions,
    # remain authoritative for membership in the target rate cell.
    for attempt in range(128):
        A, B = draw("A"), draw("B")
        hx, hz, n = lifted_matrices(G, A, B)
        k = n-gf2_rank(hx)-gf2_rank(hz)
        if 180 <= k < 189:
            return finish(A,B,hx,hz,n,k,(10**12,attempt))
    raise ValueError("sparse Z20 contract missed verified rate cell 0.300")


def CELL(rng, G, side, i, j, ctx):
    """Emit the complete selected sparse cell without subsequent redraws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: A d2s=6 screen does not ensure balanced circuit protection: parent slot 56 had 107/1250 Z failures but only 53/200000 X failures.
# REFUTED: Component short-cycle optimization alone certifies strong circuit memory: the current parent's five d2s=8 codes all reached the failed-shot cap, with worse-basis s=5.402..5.782 and widespread logical flips.
# REFUTED: Generator-rich binomial cards eliminate collective circuit failures on Z20 (2,5)x(2,5): all five current-parent measured codes reached failed-shot caps and affected all 180 logicals; d2s=9 slot53 scored 5.811 versus d2s=8 slot2 at 5.829.
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
