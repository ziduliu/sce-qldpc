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
CONCEPT = ("For cyclic Z20, order 20, shapes (2,5)x(2,5), controlled degree-two/four redistribution and invertible trinomial cells should reduce worse-basis circuit failures at comparable or lower extraction cost than the generator-gap parent.")
GROUP = ("Frozen under L1: retain the exact custom Z20 group and order. This bed measures fitness 5.7715. In F2[Z20], every distinct trinomial is a unit: reduction modulo x^5-1 has odd support of size one or three, hence neither the x+1 factor nor the irreducible degree-four factor divides it. This supports a new cell model, not a claim of cyclic-group superiority.")
SHAPES = ("Frozen equal (2,5) components each have 100 variables and 40 checks; n=580, kb=180, and guaranteed rate 9/29=0.310345 in cell 0.300. Order 20 is the largest admissible lift. Parent failures affect all 180 logicals, with diffuse 1.1-1.2x Acol/Bcol enrichment and mostly 1.0-1.1x Brow check enrichment. Strengthen both components, and verify actual full-CSS ranks rather than assuming augmentation predicts k.")
FILLING = ("Twenty explicit incidence pairs mix column degrees two, three and four with row totals seven or eight. Two degree-two plus three degree-four columns test stronger protection; mixed lower-exposure profiles provide controls. Constrain binomial ratios, unit-trinomial quotient supports and generator-valued relative voltages between monomial columns. Optimize verified lifts, then seed kernel screening with verified commutative minors and verified cyclic-factor divisions before information-set refinement. Compare exposure-weighted witness multiplicity on both sides; this surrogate is not measured pL. Six-round CNOT budgets are 23520..26880 versus the parent's 25200..26880. All five parent measurements reached failure caps, and slot 53's d2s=9 did not improve on slot 2's d2s=8 score; distance alone is not the proposed success criterion.")
FAMILY_NAME = "r030_m1r20s1_25x25_degree24_unit_minor_v3"

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


def NEW_CTX(rng, G, idx):
    """Search a new incidence/unit-cell contract using verified lifts."""
    import math
    from itertools import combinations

    # Columns specify term counts in the two component rows.
    # Profiles 1 and 3 have rank-one augmentation; the others rank two.
    profiles = (
        ((2,1),(2,1),(1,2),(1,2),(1,2)),
        ((1,1),(1,1),(2,2),(2,2),(2,2)),
        ((1,1),(1,1),(2,1),(1,2),(2,2)),
        ((1,1),(1,1),(3,1),(1,3),(2,2)),
        ((1,1),(1,1),(3,0),(0,3),(2,2)),
        ((1,1),(2,1),(1,2),(2,2),(2,2)),
        ((3,0),(2,1),(1,2),(1,2),(1,2)),
        ((3,0),(0,3),(2,1),(1,2),(1,2)),
    )
    designs = (
        (1,1),(1,3),(3,3),(2,2),(2,4),
        (4,4),(1,5),(3,5),(5,5),(2,5),
        (4,5),(0,1),(0,3),(0,5),(0,4),
        (5,6),(5,7),(3,6),(3,7),(6,7),
    )
    pa, pb = designs[idx % len(designs)]
    mode = (idx // len(designs)) % 3
    q = G.q
    mask = (1 << q) - 1
    assert q == 20

    def architecture(number):
        columns = profiles[number]
        counts = [[column[i] for column in columns] for i in range(2)]
        return counts, number

    sa, ka = architecture(pa)
    sb, kb = architecture(pb)
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert _no_dead_lines(sa, sb)
    assert all(2 <= sum(grid[i][j] for i in range(2)) <= 4
               for grid in (sa, sb) for j in range(5))
    expected_deficit = {
        side: 2 - gf2_rank([
            sum((w & 1) << j for j, w in enumerate(row))
            for row in counts])
        for side, counts in (("A", sa), ("B", sb))
    }

    # Extra harness slots change the actual ratio/quotient architecture.
    palettes = (
        (1,3,7,9),
        (1,2,3,4,6,7,8,9),
        (1,3,5,7,9),
    )
    generators = {1,3,7,9,11,13,17,19}

    def pivot_ok(grid, counts):
        pivots = [j for j in range(5)
                  if counts[0][j] == counts[1][j] == 1]
        if len(pivots) < 2:
            return True
        a, b = pivots[:2]
        if any(grid[i][j] is None for i in range(2) for j in (a,b)):
            return True
        relative = (
            grid[0][b][0][1] - grid[1][b][0][1]
            - grid[0][a][0][1] + grid[1][a][0][1]) % q
        return relative in generators

    def draw_cell(grid, counts, i, j):
        weight = counts[i][j]
        if not weight:
            return []
        if weight == 1:
            # The first two monomial columns have a generator-valued
            # relative voltage, not an index-two/five decomposition.
            old = grid[i][j]
            choices = []
            for g in G.elements:
                grid[i][j] = [g]
                if pivot_ok(grid, counts):
                    choices.append(g)
            grid[i][j] = old
            return [rng.choice(choices)]

        if weight == 2:
            used = set()
            for other in range(5):
                cell = grid[i][other]
                if other != j and cell is not None and len(cell) == 2:
                    d = G.mul(G.inv(cell[0]), cell[1])[1]
                    used.add(min(d, q-d))
            opposite = grid[1-i][j]
            if opposite is not None and len(opposite) == 2:
                d = G.mul(G.inv(opposite[0]), opposite[1])[1]
                used.add(min(d, q-d))
            choices = [d for d in palettes[mode] if d not in used]
            # At most two same-row gaps and one opposite-cell gap
            # are excluded; even the smallest palette has four choices.
            assert choices
            g = G.pick(rng)
            d = rng.choice(choices)
            return [g, G.mul(g, (0,d))]

        assert weight == 3
        if mode == 0:
            # Quotient-monomial unit: two terms cancel in F2[Z5].
            g = G.pick(rng)
            e = rng.choice([t for t in range(q) if t % 5])
            return [g, G.mul(g, (0,5)), G.mul(g, (0,e))]
        if mode == 1:
            # Quotient weight three is also a unit, but changes the
            # quotient incidence rather than relabeling the first model.
            residues = rng.sample(range(5), 3)
            return [(0, r + 5*rng.randrange(4)) for r in residues]
        return rng.sample(G.elements, 3)

    def draw(counts, kinds):
        grid = [[None] * 5 for _ in range(2)]
        for i in range(2):
            for j in range(5):
                grid[i][j] = draw_cell(grid, counts, i, j)
        assert pivot_ok(grid, counts)
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
        excess = max(0, deficit - expected_deficit[side])
        energy = 400 * excess + 60 * four / q + six / (8 * q)
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

    def sampled_kernel(rows, cols, grid, side, counts):
        """Verified witnesses and an exposure-weighted screening surrogate.

        Minor words and every proposed cyclic-factor division are checked
        against the helper-built rows. None is a distance lower bound.
        """
        witnesses = {}
        minimum = 5*q + 1
        lightest = None

        def record(word, verify=False):
            nonlocal minimum, lightest
            if not word:
                return
            if verify and any((row & word).bit_count() & 1 for row in rows):
                return
            weight = word.bit_count()
            if weight < minimum:
                minimum, lightest = weight, word
            if weight <= 14:
                witnesses[orbit_key(word)] = weight

        # These are group-algebra expressions, not an independent lift.
        # B's verified component action uses inverses of its cell labels.
        terms = [
            [[g if side == "A" else G.inv(g) for g in cell]
             for cell in row]
            for row in grid
        ]

        def product(left, right):
            out = 0
            for g in left:
                for h in right:
                    out ^= 1 << G.idx(G.mul(g, h))
            return out

        def minor(a, b):
            return (product(terms[0][a], terms[1][b])
                    ^ product(terms[0][b], terms[1][a]))

        def quotient(poly):
            # Solve (1+x)*v=poly cyclically, choosing v[q-1]=0.
            # The other solution differs by the all-one group polynomial.
            if poly.bit_count() & 1:
                return None
            previous = out = 0
            for t in range(q):
                previous ^= (poly >> t) & 1
                out |= previous << t
            assert previous == 0
            return out

        for a, b, c in combinations(range(5), 3):
            sites = (a,b,c)
            polys = [minor(b,c), minor(a,c), minor(a,b)]
            record(sum(p << (j*q) for j, p in zip(sites, polys)),
                   verify=True)
            # Common x+1 factors can hide shorter witnesses. Division
            # need not preserve a kernel word; verification decides.
            for _ in range(3):
                divided = [quotient(p) for p in polys]
                if any(p is None for p in divided):
                    break
                polys = divided
                for flips in range(8):
                    word = 0
                    for t, (j, p) in enumerate(zip(sites, polys)):
                        if (flips >> t) & 1:
                            p ^= mask
                        word |= p << (j*q)
                    record(word, verify=True)

        # Fundamental circuits come exclusively from verified column
        # syndromes. Pair-XOR reductions remain actual kernel words.
        order = list(range(5*q))
        for _ in range(32):
            rng.shuffle(order)
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
                    record(word)
                    basis.append(word)
            rng.shuffle(basis)
            for i, word in enumerate(basis):
                weight = word.bit_count()
                for other in basis[:i]:
                    trial = word ^ other
                    if not trial:
                        continue
                    record(trial)
                    w = trial.bit_count()
                    if w < weight:
                        word, weight = trial, w
                basis[i] = word

        assert lightest is not None
        # Avoid treating "no sampled small word" as zero estimated risk.
        witnesses.setdefault(orbit_key(lightest), minimum)
        degrees = [sum(counts[i][j] for i in range(2)) for j in range(5)]
        loads = [0.0] * 5
        risk = 0.0
        for word, weight in witnesses.items():
            exposure = [
                ((word >> (j*q)) & mask).bit_count() for j in range(5)
            ]
            # A deliberately approximate gate-exposure model. It is
            # neither the ASC circuit nor the evaluator's decoder.
            contribution = float(math.comb(weight, (weight+1)//2))
            for j, occupied in enumerate(exposure):
                eta = 0.002 * (8 + 6*degrees[j])
                contribution *= eta ** (occupied / 2.0)
            risk += contribution
            for j, occupied in enumerate(exposure):
                loads[j] += contribution * occupied / weight

        eight = sum(weight <= 8 for weight in witnesses.values())
        return (min(minimum, 14), eight, max(loads),
                len(witnesses), risk)

    def select(side, counts, kinds):
        pool = []
        positions = [(i,j) for i in range(2) for j in range(5)
                     if counts[i][j]]
        for restart in range(6):
            grid = draw(counts, kinds)
            current = assess(side, grid)
            best_grid = _copy_cells(grid)
            best = current

            for step in range(200):
                i, j = rng.choice(positions)
                old = grid[i][j]
                grid[i][j] = draw_cell(grid, counts, i, j)
                proposed = assess(side, grid)
                delta = proposed[0] - current[0]
                temperature = 12 * (1 - step / 200) + 0.15
                if delta <= 0 or rng.random() < math.exp(
                        -min(700.0, delta / temperature)):
                    current = proposed
                    if (current[0] < best[0] or
                            (current[0] == best[0]
                             and rng.random() < 0.04)):
                        best = current
                        best_grid = _copy_cells(grid)
                else:
                    grid[i][j] = old

            _, rows, cols, deficit, four, six = best
            assert pivot_ok(best_grid, counts)
            short = _short_collision_counts(rows, 5*q)
            pool.append((deficit, short, four, six, restart,
                         best_grid, rows, cols))

        pool.sort(key=lambda item: item[:5])
        finalists = []
        for deficit, short, four, six, serial, grid, rows, cols in pool[:4]:
            collisions, boundary = triple_statistics(cols)
            distance, eight, worst, total, risk = sampled_kernel(
                rows, cols, grid, side, counts)
            # Verified short dependencies precede the noisy-exposure
            # surrogate. A larger sampled distance is only a tiebreaker.
            key = (max(0, deficit-expected_deficit[side]),
                   short, collisions, risk, worst, four, boundary,
                   -distance, eight, total, six, serial)
            finalists.append((key, grid))
        finalists.sort(key=lambda item: item[0])
        return finalists

    # Keep alternatives rather than voiding a slot when one optimized
    # component has a small rank deficit.
    for attempt in range(2):
        aa = select("A", sa, ka)
        bb = select("B", sb, kb)
        options = []
        for ia, (akey, A) in enumerate(aa):
            for ib, (bkey, B) in enumerate(bb):
                short = tuple(max(x,y)
                              for x,y in zip(akey[1], bkey[1]))
                pairing = (
                    max(akey[0], bkey[0]), short,
                    max(akey[2], bkey[2]),
                    max(akey[3], bkey[3]),
                    akey[3] + bkey[3],
                    max(akey[4], bkey[4]), ia+ib,
                )
                options.append((pairing, ia, ib, A, B))
        options.sort(key=lambda item: item[:3])
        for _, ia, ib, A, B in options:
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (k > 0 and 11*n <= 40*k < 13*n):
                continue
            assert css_ok(hx, hz)
            assert max(row.bit_count() for row in hx + hz) <= W_MAX
            return {"idx": idx, "cells": {"A": A, "B": B},
                    "k": k, "screen": (aa[ia][0], bb[ib][0])}
    raise ValueError("no degree24 unit-cell filling in rate cell 0.300")


def CELL(rng, G, side, i, j, ctx):
    """Emit the selected incidence/unit-cell design without late redraws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: A d2s=6 screen does not ensure balanced circuit protection: parent slot 56 had 107/1250 Z failures but only 53/200000 X failures.
# REFUTED: Component short-cycle optimization alone certifies strong circuit memory: the current parent's five d2s=8 codes all reached the failed-shot cap, with worse-basis s=5.402..5.782 and widespread logical flips.
# REFUTED: Generator-rich doubletons eliminate the weight-eight logical families: current measured baseline slots still showed hundreds of weight-eight witnesses, failures affecting all 180 logicals, and failure-cap termination in both bases.
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
